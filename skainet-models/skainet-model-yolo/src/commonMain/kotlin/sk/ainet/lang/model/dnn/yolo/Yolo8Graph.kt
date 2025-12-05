package sk.ainet.lang.model.dnn.yolo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.types.FP32
import kotlin.math.round

/**
 * Minimal YOLOv8-style graph assembled from reusable blocks.
 *
 * This is intentionally compact: it wires the main backbone, a lightweight neck
 * with upsampling/concats, and three detection heads. The output currently returns
 * the largest-scale head; smaller heads are kept for future concatenation when
 * post-processing is added.
 */
internal class Yolo8Graph(private val executionContext: ExecutionContext, private val config: YoloConfig) {

    private val base = config.baseChannels
    private val c1 = base
    private val c2 = base * 2
    private val c3 = base * 4
    private val c4 = base * 8
    private val c5 = base * 16

    private fun scaleDepth(baseCount: Int): Int =
        kotlin.math.max(1, round(baseCount * config.depthMultiple).toInt())

    private val stem = ConvSiLU(
        inChannels = 3,
        outChannels = c1,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stem",
        executionContext = executionContext
    )

    private val stage1 = ConvSiLU(
        inChannels = c1,
        outChannels = c2,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage1_down",
        executionContext = executionContext
    )
    private val c2f1 = C2f(
        inChannels = c2,
        outChannels = c2,
        bottlenecks = scaleDepth(3),
        name = "c2f1",
        executionContext = executionContext
    )

    private val stage2 = ConvSiLU(
        inChannels = c2,
        outChannels = c3,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage2_down",
        executionContext = executionContext
    )
    private val c2f2 = C2f(
        inChannels = c3,
        outChannels = c3,
        bottlenecks = scaleDepth(6),
        name = "c2f2",
        executionContext = executionContext
    )

    private val stage3 = ConvSiLU(
        inChannels = c3,
        outChannels = c4,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage3_down",
        executionContext = executionContext
    )
    private val c2f3 = C2f(
        inChannels = c4,
        outChannels = c4,
        bottlenecks = scaleDepth(6),
        name = "c2f3",
        executionContext = executionContext
    )

    private val stage4 = ConvSiLU(
        inChannels = c4,
        outChannels = c5,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage4_down",
        executionContext = executionContext
    )
    private val c2f4 = C2f(
        inChannels = c5,
        outChannels = c5,
        bottlenecks = scaleDepth(3),
        name = "c2f4",
        executionContext = executionContext
    )

    private val sppf = Sppf(channels = c5, name = "sppf", executionContext = executionContext)

    // Neck
    private val up1 = UpsampleWrapper(scale = 2 to 2, name = "up1", executionContext = executionContext)
    private val c2fNeck1 = C2f(
        inChannels = c5 + c4,
        outChannels = c4,
        bottlenecks = scaleDepth(3),
        name = "model.12",
        executionContext = executionContext
    )

    private val up2 = UpsampleWrapper(scale = 2 to 2, name = "up2", executionContext = executionContext)
    private val c2fNeck2 = C2f(
        inChannels = c4 + c3,
        outChannels = c3,
        bottlenecks = scaleDepth(3),
        name = "model.15",
        executionContext = executionContext
    )

    // Bottom-up path
    private val down1 = ConvSiLU(
        inChannels = c3,
        outChannels = c3,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "model.16",
        executionContext = executionContext
    )
    private val c2fDown1 = C2f(
        inChannels = c3 + c4,
        outChannels = c4,
        bottlenecks = scaleDepth(3),
        name = "model.18",
        executionContext = executionContext
    )
    private val down2 = ConvSiLU(
        inChannels = c4,
        outChannels = c4,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "model.19",
        executionContext = executionContext
    )
    private val c2fDown2 = C2f(
        inChannels = c4 + c5,
        outChannels = c5,
        bottlenecks = scaleDepth(3),
        name = "model.21",
        executionContext = executionContext
    )

    private val detect = DecoupledDetectHead(
        ch = listOf(c3, c4, c5),
        numClasses = config.numClasses,
        regMax = config.regMax,
        name = "model.22",
        executionContext = executionContext
    )

    fun modules(): List<Module<FP32, Float>> = buildList {
        add(stem.module)
        add(stage1.module); add(c2f1)
        add(stage2.module); add(c2f2)
        add(stage3.module); add(c2f3)
        add(stage4.module); add(c2f4)
        add(sppf)
        add(up1)
        add(c2fNeck1)
        add(up2)
        add(c2fNeck2)
        add(down1.module)
        add(c2fDown1)
        add(down2.module)
        add(c2fDown2)
        add(detect)
    }

    fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): HeadOutputs {
        val x0 = stem.forward(input, ctx)
        val x1 = c2f1.forward(stage1.forward(x0, ctx), ctx) // 1/4
        val x2 = c2f2.forward(stage2.forward(x1, ctx), ctx) // 1/8
        val x3 = c2f3.forward(stage3.forward(x2, ctx), ctx) // 1/16
        val x4 = c2f4.forward(stage4.forward(x3, ctx), ctx) // 1/32
        val p5 = sppf.forward(x4, ctx)

        // Neck path
        val p4 = c2fNeck1.forward(
            concatAlongChannels(
                up1.forward(p5, ctx),
                x3,
                ctx
            ),
            ctx
        )
        val p3 = c2fNeck2.forward(
            concatAlongChannels(
                up2.forward(p4, ctx),
                x2,
                ctx
            ),
            ctx
        )

        // Bottom-up refinements
        val p4d = c2fDown1.forward(
            concatAlongChannels(
                down1.forward(p3, ctx),
                p4,
                ctx
            ),
            ctx
        )
        val p5d = c2fDown2.forward(
            concatAlongChannels(
                down2.forward(p4d, ctx),
                p5,
                ctx
            ),
            ctx
        )

        return detect.forward(listOf(p3, p4d, p5d), ctx)
    }

    private fun concatAlongChannels(
        a: Tensor<FP32, Float>,
        b: Tensor<FP32, Float>,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> = ctx.ops.concat(listOf(a, b), dim = 1)
}

internal class ConvBnSiLU(
    private val inChannels: Int,
    private val outChannels: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val name: String,
    executionContext: ExecutionContext
) {
    val module: Module<FP32, Float> = definition {
        network(executionContext) {
            sequential {
                stage(name) {
                    conv2d(
                        outChannels = outChannels,
                        kernelSize = kernel to kernel,
                        stride = stride to stride,
                        padding = padding to padding
                    ) {
                        this.inChannels = this@ConvBnSiLU.inChannels
                    }
                    batchNorm(numFeatures = outChannels)
                    activation { tensor -> tensor.silu() }
                }
            }
        }
    }

    fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        module.forward(input, ctx)
}

internal class ConvSiLU(
    private val inChannels: Int,
    private val outChannels: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val name: String,
    executionContext: ExecutionContext
) {
    val module: Module<FP32, Float> = definition {
        network(executionContext) {
            sequential {
                stage(name) {
                    conv2d(
                        outChannels = outChannels,
                        kernelSize = kernel to kernel,
                        stride = stride to stride,
                        padding = padding to padding
                    ) {
                        this.inChannels = this@ConvSiLU.inChannels
                        this.bias = true
                    }
                    activation { tensor -> tensor.silu() }
                }
            }
        }
    }

    fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        module.forward(input, ctx)
}

internal class C2f(
    private val inChannels: Int,
    private val outChannels: Int,
    bottlenecks: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val hidden = outChannels / 2
    private val cv1 = ConvSiLU(
        inChannels = inChannels,
        outChannels = hidden * 2,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_cv1",
        executionContext = executionContext
    )
    private val bottleneckBlocks = List(bottlenecks) { idx ->
        Bottleneck(
            channels = hidden,
            name = "${name}_m$idx",
            executionContext = executionContext
        )
    }
    private val cv2 = ConvSiLU(
        inChannels = hidden * (2 + bottlenecks),
        outChannels = outChannels,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_cv2",
        executionContext = executionContext
    )

    override val modules: List<Module<FP32, Float>> = buildList {
        add(cv1.module)
        bottleneckBlocks.forEach { add(it) }
        add(cv2.module)
    }

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val y = ctx.ops.split(cv1.forward(input, ctx), splitSize = hidden, dim = 1).toMutableList()
        var last = y.last()
        bottleneckBlocks.forEach { block ->
            last = block.forward(last, ctx)
            y += last
        }
        val concat = ctx.ops.concat(y, dim = 1)
        return cv2.forward(concat, ctx)
    }
}

private class Bottleneck(
    channels: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val cv1 = ConvSiLU(
        inChannels = channels,
        outChannels = channels,
        kernel = 3,
        stride = 1,
        padding = 1,
        name = "${name}_cv1",
        executionContext = executionContext
    )
    private val cv2 = ConvSiLU(
        inChannels = channels,
        outChannels = channels,
        kernel = 3,
        stride = 1,
        padding = 1,
        name = "${name}_cv2",
        executionContext = executionContext
    )

    override val modules: List<Module<FP32, Float>> = listOf(cv1.module, cv2.module)

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val h1 = cv1.forward(input, ctx)
        val h2 = cv2.forward(h1, ctx)
        return ctx.ops.add(input, h2)
    }
}

public data class HeadTensor(
    val reg: Tensor<FP32, Float>,
    val cls: Tensor<FP32, Float>
)

public data class HeadOutputs(
    val small: HeadTensor,
    val medium: HeadTensor,
    val large: HeadTensor
)

internal class Sppf(
    channels: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val reduce = ConvSiLU(
        inChannels = channels,
        outChannels = channels / 2,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_reduce",
        executionContext = executionContext
    )
    private val expand = ConvSiLU(
        inChannels = channels * 2,
        outChannels = channels,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_expand",
        executionContext = executionContext
    )

    override val modules: List<Module<FP32, Float>> = listOf(reduce.module, expand.module)

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val x = reduce.forward(input, ctx)
        val p1 = ctx.ops.maxPool2d(x, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
        val p2 = ctx.ops.maxPool2d(p1, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
        val p3 = ctx.ops.maxPool2d(p2, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
        val concat = ctx.ops.concat(listOf(x, p1, p2, p3), dim = 1)
        return expand.forward(concat, ctx)
    }
}

internal class UpsampleWrapper(
    private val scale: Pair<Int, Int>,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {
    private val module: Module<FP32, Float> = definition {
        network(executionContext) {
            sequential {
                stage(name) {
                    upsample2d(scale = scale, mode = UpsampleMode.Nearest)
                }
            }
        }
    }

    override val modules: List<Module<FP32, Float>> = listOf(module)

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        module.forward(input, ctx)
}

internal class DecoupledDetectHead(
    ch: List<Int>,
    private val numClasses: Int,
    private val regMax: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val regBranches = ch.mapIndexed { idx, c ->
        DecoupledBranch(
            inChannels = c,
            midChannels = 64,
            outChannels = regMax * 4,
            baseName = "$name.cv2.$idx",
            executionContext = executionContext
        )
    }
    private val clsBranches = ch.mapIndexed { idx, c ->
        DecoupledBranch(
            inChannels = c,
            midChannels = 64,
            outChannels = numClasses,
            baseName = "$name.cv3.$idx",
            executionContext = executionContext
        )
    }

    // Holder for DFL projection weights so the ONNX initializer maps cleanly.
    private val dfl = DflProjection(regMax, name = "$name.dfl", executionContext = executionContext)

    override val modules: List<Module<FP32, Float>> = buildList {
        addAll(regBranches)
        addAll(clsBranches)
        add(dfl)
    }

    fun forward(inputs: List<Tensor<FP32, Float>>, ctx: ExecutionContext): HeadOutputs {
        require(inputs.size == 3) { "Detect head expects 3 feature maps, got ${inputs.size}" }
        val reg = regBranches.mapIndexed { idx, branch -> branch.forward(inputs[idx], ctx) }
        val cls = clsBranches.mapIndexed { idx, branch -> branch.forward(inputs[idx], ctx) }
        return HeadOutputs(
            small = HeadTensor(reg = reg[0], cls = cls[0]),
            medium = HeadTensor(reg = reg[1], cls = cls[1]),
            large = HeadTensor(reg = reg[2], cls = cls[2])
        )
    }

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        // Return a placeholder tensor (small cls) to satisfy Module contract; inference uses forward(inputs).
        return clsBranches.first().forward(input, ctx)
    }
}

private class DecoupledBranch(
    private val inChannels: Int,
    midChannels: Int,
    private val outChannels: Int,
    baseName: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {
    override val name: String = baseName
    private val c1 = ConvSiLU(
        inChannels = inChannels,
        outChannels = midChannels,
        kernel = 3,
        stride = 1,
        padding = 1,
        name = "$baseName.0",
        executionContext = executionContext
    )
    private val c2 = ConvSiLU(
        inChannels = midChannels,
        outChannels = midChannels,
        kernel = 3,
        stride = 1,
        padding = 1,
        name = "$baseName.1",
        executionContext = executionContext
    )
    private val out = ConvNoAct(
        inChannels = midChannels,
        outChannels = outChannels,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "$baseName.2",
        executionContext = executionContext
    )

    override val modules: List<Module<FP32, Float>> = listOf(c1.module, c2.module, out.module)

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val h1 = c1.forward(input, ctx)
        val h2 = c2.forward(h1, ctx)
        return out.forward(h2, ctx)
    }
}

private class ConvNoAct(
    private val inChannels: Int,
    private val outChannels: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val name: String,
    executionContext: ExecutionContext
) {
    val module: Module<FP32, Float> = definition {
        network(executionContext) {
            sequential {
                stage(name) {
                    conv2d(
                        outChannels = outChannels,
                        kernelSize = kernel to kernel,
                        stride = stride to stride,
                        padding = padding to padding
                    ) {
                        this.inChannels = this@ConvNoAct.inChannels
                        this.bias = true
                    }
                }
            }
        }
    }

    fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        module.forward(input, ctx)
}

private class DflProjection(
    regMax: Int,
    override val name: String,
    executionContext: ExecutionContext
) : Module<FP32, Float>(), sk.ainet.lang.nn.topology.ModuleParameters<FP32, Float> {
    private val weightShape = sk.ainet.lang.tensor.Shape(intArrayOf(1, regMax, 1, 1))
    override val params: List<sk.ainet.lang.nn.topology.ModuleParameter<FP32, Float>> =
        listOf(
            sk.ainet.lang.nn.topology.ModuleParameter.WeightParameter(
                "$name.conv.weight",
                executionContext.zeros(weightShape, FP32::class)
            )
        )
    override val modules: List<Module<FP32, Float>> = emptyList()

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        input
}
