package sk.ainet.lang.model.dnn.yolo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

public class Yolo8 : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    private fun buildModel(executionContext: ExecutionContext): Module<FP32, Float> = definition {
        network(executionContext) {
            // Placeholder input definition; actual YOLOv8 graph to be wired when available
            input(1)
        }
    }

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
            intendedUse = "YOLOv8-style object detection placeholder implemented with new Model API.",
            limitations = "Architecture is a placeholder; no pretrained weights or real detection head wired yet."
        )
    }
}