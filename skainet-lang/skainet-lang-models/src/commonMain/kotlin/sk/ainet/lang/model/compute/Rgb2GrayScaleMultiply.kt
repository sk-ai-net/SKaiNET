package sk.ainet.lang.model.compute

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.model.ModelIndexEntry
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.FP16

/**
 * Color Space Transformation: RGB → Grayscale using direct tensor multiplication (no conv).
 *
 * Implements the NumPy-equivalent operation per pixel:
 *   weights = [0.2989, 0.5870, 0.1140]
 *   gray = (H*W x 3) @ weights → reshape(H, W)
 *
 * Here we avoid any convolution and compute a weighted sum across the channel dimension:
 *   (N,3,H,W) * (1,3,1,1) -> sum over channel dim (1) -> (N,1,H,W)
 *
 * Expected input shape: (N, 3, H, W)
 * Output shape: (N, 1, H, W)
 */
public class Rgb2GrayScaleMatMul(constCtx: ExecutionContext) : Model<FP16, Float> {

    // Use a local execution context only to build constant tensors (weights).
    // Runtime compute ops will come from the input tensor's ops during forward.
    private val weights: Tensor<FP16, Float>

    init {
        val factors = constCtx.fromFloatArray<FP16, Float>(
            shape = Shape(1, 3),
            dtype = FP16::class,
            data = floatArrayOf(0.2989f, 0.5870f, 0.1140f)
        )

        // Reshape like PyTorch view(1,3,1,1)
        weights = constCtx.ops.reshape(factors, Shape(1, 3, 1, 1))
    }


    private val impl = object : Module<FP16, Float>() {
        // Pre-allocate luminance weights in broadcastable shape (1, 3, 1, 1)


        override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> {
            // Validate shape (N,3,H,W)
            val dims = input.shape.dimensions
            require(input.rank == 4) { "Rgb2GrayScaleMultiply expects rank-4 input (B,3,H,W), but was ${input.shape}" }
            require(dims[1] == 3) { "Rgb2GrayScaleMultiply expects 3 channels at dim=1, but was ${dims[1]}" }

            // Elementwise weighted sum across channel dimension using broadcasting
            val weighted = input * weights // (N,3,H,W)
            val grayHW = input.ops.sum(weighted, 1) // (N,H,W)

            // Restore channel dimension to get (N,1,H,W)
            val gray = input.ops.unsqueeze(grayHW, 1)
            return gray
        }

        override val name: String = "Rgb2GrayScaleMultiply"
        override val modules: List<Module<FP16, Float>> = emptyList()
    }

    override fun model(executionContext: ExecutionContext): Module<FP16, Float> = impl

    override fun modelCard(): ModelCard {
        // Minimal model card for a simple color space conversion utility.
        return ModelCard(
            license = "apache-2.0",
            libraryName = "skainet",
            pipelineTag = "image-processing",
            language = listOf("en"),
            modalities = listOf("image"),
            baseModel = "n/a",
            contextLength = 0,
            datasets = emptyList(),
            metrics = emptyList(),
            modelIndex = listOf(
                ModelIndexEntry(
                    name = "Rgb2GrayScaleMultiply",
                    results = emptyList()
                )
            ),
            intendedUse = "RGB→Grayscale via direct weighted sum (no conv) with weights [0.2989, 0.5870, 0.1140]. Input: (N,3,H,W). Output: (N,3,H,W).",
            limitations = "Assumes input tensors are normalized appropriately and channel order is RGB. No color profile handling; purely linear weights."
        )
    }

}
