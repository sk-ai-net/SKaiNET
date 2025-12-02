package sk.ainet.lang.model.dnn.yolo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

public class Yolo8(
    private val config: YoloConfig = YoloConfig()
) : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    private fun buildModel(executionContext: ExecutionContext): Module<FP32, Float> =
        Yolo8Module(executionContext, config)

    // Backward-compatible helper for old call sites
    public fun model(executionContext: ExecutionContext): Module<FP32, Float> = create(executionContext)

    override fun create(executionContext: ExecutionContext): Module<FP32, Float> = buildModel(executionContext)

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): Tensor<FP32, Float> {
        reportProgress(0, 1, "starting yolo8 forward")
        val out = module.forward(inputValue, executionContext)
        reportProgress(1, 1, "done")
        return out
    }

    /**
     * Runs the YOLO graph and returns all head outputs (small/medium/large) for downstream post-processing.
     */
    public fun calculateHeads(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        executionContext: ExecutionContext
    ): HeadOutputs {
        require(module is Yolo8Module) { "calculateHeads expects a Yolo8Module instance" }
        return module.forwardHeads(inputValue, executionContext)
    }

    /**
     * Full inference: forward pass + decode + NMS. Expects the input tensor to be
     * preprocessed to the configured model size (see [YoloPreprocess]).
     */
    public suspend fun infer(
        module: Module<FP32, Float>,
        input: YoloInput,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit = { _, _, _ -> }
    ): List<Detection> {
        require(module is Yolo8Module) { "infer expects a Yolo8Module instance" }
        reportProgress(0, 2, "yolo forward")
        val heads = module.forwardHeads(input.tensor, executionContext)
        reportProgress(1, 2, "decode")
        val detections = YoloDecoder.decode(heads, config, input)
        reportProgress(2, 2, "done")
        return detections
    }

    override fun modelCard(): ModelCard {
        return ModelCard(
            license = "apache-2.0",
            libraryName = "skainet",
            pipelineTag = "object-detection",
            language = listOf("en"),
            modalities = listOf("vision"),
            baseModel = "yolov8",
            contextLength = 0,
            datasets = emptyList(),
            metrics = emptyList(),
            modelIndex = emptyList(),
            intendedUse = "YOLOv8-style object detection skeleton using SKaiNET DSL blocks.",
            limitations = "Weights and image preprocessing are not included; outputs depend on provided weights and input normalization."
        )
    }
}

private class Yolo8Module(
    private val executionContext: ExecutionContext,
    private val config: YoloConfig
) : Module<FP32, Float>() {

    private val graph = Yolo8Graph(executionContext, config)

    override val modules: List<Module<FP32, Float>>
        get() = graph.modules()

    override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        graph.forward(input, ctx).large

    fun forwardHeads(input: Tensor<FP32, Float>, ctx: ExecutionContext): HeadOutputs =
        graph.forward(input, ctx)
}
