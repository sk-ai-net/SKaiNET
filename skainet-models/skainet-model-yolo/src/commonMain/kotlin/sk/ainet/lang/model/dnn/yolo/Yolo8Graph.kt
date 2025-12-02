package sk.ainet.lang.model.dnn.yolo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.types.FP32

/**
 * Minimal YOLOv8-style graph assembled from reusable blocks.
 *
 * This is intentionally compact: it wires the main backbone, a lightweight neck
 * with upsampling/concats, and three detection heads. The output currently returns
 * the largest-scale head; smaller heads are kept for future concatenation when
 * post-processing is added.
 */
internal class Yolo8Graph(private val executionContext: ExecutionContext, private val config: YoloConfig) {

    private val stem = ConvBnSiLU(
        inChannels = 3,
        outChannels = 32,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stem",
        executionContext = executionContext
    )

    private val stage1 = ConvBnSiLU(
        inChannels = 32,
        outChannels = 64,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage1_down",
        executionContext = executionContext
    )
    private val c2f1 = C2f(
        inChannels = 64,
        outChannels = 64,
        bottlenecks = 1,
        name = "c2f1",
        executionContext = executionContext
    )

    private val stage2 = ConvBnSiLU(
        inChannels = 64,
        outChannels = 128,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage2_down",
        executionContext = executionContext
    )
    private val c2f2 = C2f(
        inChannels = 128,
        outChannels = 128,
        bottlenecks = 1,
        name = "c2f2",
        executionContext = executionContext
    )

    private val stage3 = ConvBnSiLU(
        inChannels = 128,
        outChannels = 256,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage3_down",
        executionContext = executionContext
    )
    private val c2f3 = C2f(
        inChannels = 256,
        outChannels = 256,
        bottlenecks = 1,
        name = "c2f3",
        executionContext = executionContext
    )

    private val stage4 = ConvBnSiLU(
        inChannels = 256,
        outChannels = 512,
        kernel = 3,
        stride = 2,
        padding = 1,
        name = "stage4_down",
        executionContext = executionContext
    )
    private val c2f4 = C2f(
        inChannels = 512,
        outChannels = 512,
        bottlenecks = 1,
        name = "c2f4",
        executionContext = executionContext
    )

    private val sppf = Sppf(
        channels = 512,
        name = "sppf",
        executionContext = executionContext
    )

    // Neck
    private val up1 = UpsampleWrapper(scale = 2 to 2, name = "up1", executionContext = executionContext)
    private val c2fNeck1 = C2f(
        inChannels = 512 + 256,
        outChannels = 256,
        bottlenecks = 1,
        name = "c2f_neck1",
        executionContext = executionContext
    )

    private val up2 = UpsampleWrapper(scale = 2 to 2, name = "up2", executionContext = executionContext)
    private val c2fNeck2 = C2f(
        inChannels = 256 + 128,
        outChannels = 128,
        bottlenecks = 1,
        name = "c2f_neck2",
        executionContext = executionContext
    )

    // Heads (three scales)
    private val headSmall = DetectHead(
        inChannels = 128,
        outChannels = config.numClasses + 5,
        name = "detect_small",
        executionContext = executionContext
    )
    private val headMedium = DetectHead(
        inChannels = 256,
        outChannels = config.numClasses + 5,
        name = "detect_medium",
        executionContext = executionContext
    )
    private val headLarge = DetectHead(
        inChannels = 512,
        outChannels = config.numClasses + 5,
        name = "detect_large",
        executionContext = executionContext
    )

    fun modules(): List<Module<FP32, Float>> = listOf(
        stem.module,
        stage1.module, c2f1,
        stage2.module, c2f2,
        stage3.module, c2f3,
        stage4.module, c2f4,
        sppf,
        up1,
        c2fNeck1,
        up2,
        c2fNeck2,
        headSmall,
        headMedium,
        headLarge
    )

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

        // Heads: currently return the large-scale output; smaller scales stay available for later post-processing wiring.
        val small = headSmall.forward(p3, ctx)
        val medium = headMedium.forward(p4, ctx)
        val large = headLarge.forward(p5, ctx)
        return HeadOutputs(small = small, medium = medium, large = large)
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

internal class C2f(
    private val inChannels: Int,
    private val outChannels: Int,
    private val bottlenecks: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val hidden = outChannels / 2
    private val reduce = ConvBnSiLU(
        inChannels = inChannels,
        outChannels = hidden * 2,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_reduce",
        executionContext = executionContext
    )
    private val bottleneckBlocks = List(bottlenecks) { idx ->
        ConvBnSiLU(
            inChannels = hidden * 2,
            outChannels = hidden * 2,
            kernel = 3,
            stride = 1,
            padding = 1,
            name = "${name}_b$idx",
            executionContext = executionContext
        )
    }
    private val expand = ConvBnSiLU(
        inChannels = hidden * (2 + bottlenecks),
        outChannels = outChannels,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_expand",
        executionContext = executionContext
    )

    override val modules: List<Module<FP32, Float>> = buildList {
        add(reduce.module)
        bottleneckBlocks.forEach { add(it.module) }
        add(expand.module)
    }

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val y0 = reduce.forward(input, ctx)
        val parts = mutableListOf(y0)
        var last = y0
        bottleneckBlocks.forEach { block ->
            last = block.forward(last, ctx)
            parts += last
        }
        val concat = ctx.ops.concat(parts, dim = 1)
        return expand.forward(concat, ctx)
    }
}

internal data class HeadOutputs(
    val small: Tensor<FP32, Float>,
    val medium: Tensor<FP32, Float>,
    val large: Tensor<FP32, Float>
)

internal class Sppf(
    private val channels: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val reduce = ConvBnSiLU(
        inChannels = channels,
        outChannels = channels,
        kernel = 1,
        stride = 1,
        padding = 0,
        name = "${name}_reduce",
        executionContext = executionContext
    )
    private val expand = ConvBnSiLU(
        inChannels = channels * 4,
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

internal class DetectHead(
    private val inChannels: Int,
    private val outChannels: Int,
    override val name: String,
    private val executionContext: ExecutionContext
) : Module<FP32, Float>() {

    private val stem = ConvBnSiLU(
        inChannels = inChannels,
        outChannels = inChannels,
        kernel = 3,
        stride = 1,
        padding = 1,
        name = "${name}_stem",
        executionContext = executionContext
    )

    private val proj: Module<FP32, Float> = definition {
        network(executionContext) {
            sequential {
                stage("${name}_proj") {
                    conv2d(
                        outChannels = outChannels,
                        kernelSize = 1 to 1,
                        stride = 1 to 1,
                        padding = 0 to 0
                    ) {
                        this.inChannels = inChannels
                        this.bias = true
                    }
                }
            }
        }
    }

    override val modules: List<Module<FP32, Float>> = listOf(stem.module, proj)

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val h = stem.forward(input, ctx)
        return proj.forward(h, ctx)
    }
}
