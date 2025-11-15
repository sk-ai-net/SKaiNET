package sk.ainet.lang.model.compute

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.model.ModelIndexEntry
import sk.ainet.lang.model.ModelResult
import sk.ainet.lang.model.Task
import sk.ainet.lang.model.Dataset
import sk.ainet.lang.model.Metric
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Color Space Transformation: RGB → Grayscale using fixed luminance weights.
 *
 * Implements the equivalent of NumPy code:
 *   weights = [0.2989, 0.5870, 0.1140]
 *   gray = (H*W x 3) @ weights → reshape(H, W)
 *
 * In SKainNET DSL this is realized as a 1x1 Conv2D with inChannels=3 and outChannels=1
 * whose kernel is fixed to the weights above and with no bias.
 *
 * Expected input shape: (N, 3, H, W)
 * Output shape: (N, 1, H, W)
 */
public class Rgb2GrayScale : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    private val modelImpl: Module<FP32, Float> = definition {
        network {
            sequential {
                // 1x1 convolution performing per-pixel dot product with RGB weights
                conv2d(
                    outChannels = 1,
                    kernelSize = 1 to 1,
                    stride = 1 to 1,
                    padding = 0 to 0,
                    dilation = 1 to 1,
                    groups = 1,
                    bias = false,
                    id = "rgb2gray"
                ) {
                    inChannels = 3
                    // weights shape = (out=1, in=3, kH=1, kW=1)
                    weights { shape ->
                        fromArray(floatArrayOf(0.2989f, 0.5870f, 0.1140f))
                    }
                }
            }
        }
    }

    // Keep backward-compatible helper for existing tests/usages
    public fun model(executionContext: ExecutionContext): Module<FP32, Float> = create(executionContext)

    override fun create(executionContext: ExecutionContext): Module<FP32, Float> = modelImpl

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): Tensor<FP32, Float> {
        // trivial pass-through computation; the heavy work is in the module
        reportProgress(0, 1, "starting rgb2gray")
        val out = module.forward(inputValue, executionContext)
        reportProgress(1, 1, "done")
        return out
    }

    override fun modelCard(): ModelCard {
        return ModelCard(
            license = "apache-2.0",
            libraryName = "skainet",
            pipelineTag = "image-processing",
            language = listOf("en"),
            modalities = listOf("image"),
            baseModel = "n/a",
            contextLength = 0,
            datasets = emptyList(),
            metrics = listOf("functional-correctness"),
            modelIndex = listOf(
                ModelIndexEntry(
                    name = "Rgb2GrayScale",
                    results = listOf(
                        ModelResult(
                            task = Task(type = "image-to-image"),
                            dataset = Dataset(name = "n/a", type = "synthetic"),
                            metrics = listOf(Metric(name = "N/A", value = 0.0))
                        )
                    )
                )
            ),
            intendedUse = "RGB→Grayscale via fixed 1x1 Conv2D with weights [0.2989, 0.5870, 0.1140]. Input: (N,3,H,W). Output: (N,1,H,W).",
            limitations = "Assumes RGB channel order and linear color space; no gamma or color profile handling."
        )
    }
}