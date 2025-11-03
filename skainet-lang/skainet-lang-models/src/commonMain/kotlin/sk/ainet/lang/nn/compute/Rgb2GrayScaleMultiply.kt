package sk.ainet.lang.nn.compute

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

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
public class Rgb2GrayScaleMultiply : Model {

    // Use a local execution context only to build constant tensors (weights).
    // Runtime compute ops will come from the input tensor's ops during forward.
    private val constCtx: ExecutionContext = DefaultNeuralNetworkExecutionContext()

    // Pre-allocate luminance weights in broadcastable shape (1, 3, 1, 1)
    private val weights: Tensor<FP32, Float> = constCtx.fromFloatArray(
        shape = Shape(1, 3, 1, 1),
        dtype = FP32::class,
        data = floatArrayOf(0.2989f, 0.5870f, 0.1140f)
    )

    private val impl = object : Module<FP32, Float>() {
        override val name: String = "Rgb2GrayScaleMultiply"
        override val modules: List<Module<FP32, Float>> = emptyList()

        override fun forward(input: Tensor<FP32, Float>): Tensor<FP32, Float> {
            // Validate shape (N,3,H,W)
            val dims = input.shape.dimensions
            require(input.rank == 4) { "Rgb2GrayScaleMultiply expects rank-4 input (N,3,H,W), but was ${input.shape}" }
            require(dims[1] == 3) { "Rgb2GrayScaleMultiply expects 3 channels at dim=1, but was ${dims[1]}" }

            // Elementwise weighted sum across channel dimension using broadcasting
            val weighted = input * weights // (N,3,H,W)
            val grayHW = weighted.sum(dim = 1) // (N,H,W)

            // Restore channel dimension to get (N,1,H,W)
            val gray = input.ops.unsqueeze(grayHW, 1)
            return gray
        }
    }

    override fun <T : DType, V> model(): Module<FP32, Float> = impl

    override fun modelCard(): String =
        "RGB→Grayscale via direct weighted sum (no conv) with weights [0.2989, 0.5870, 0.1140]. " +
        "Input: (N,3,H,W). Output: (N,1,H,W)."
}
