package sk.ainet.lang.tensor.ops

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.context.data
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Demonstrates how to convert a 2D matrix (W, H) to a 4D tensor with shape (1, 1, W, H)
 * using existing unsqueeze operation.
 */
class MatrixToBCHWConversionTest {
    private val ctx = DefaultNeuralNetworkExecutionContext()

    @Test
    fun convertMatrixWH_to_1x1xWH_with_unsqueeze() {
        // Build a 5x5 matrix using the data DSL
        val matrix: Tensor<FP32, Float> = data<FP32, Float>(ctx) {
            matrix(5, 5) {
                ones()
            }
        }
        assertEquals(Shape(5, 5), matrix.shape)

        // Convert (W, H) -> (1, W, H)
        val unsqueezed0 = matrix.ops.unsqueeze(matrix, dim = 0)
        assertEquals(Shape(1, 5, 5), unsqueezed0.shape)

        // Convert (1, W, H) -> (1, 1, W, H)
        val bchw = unsqueezed0.ops.unsqueeze(unsqueezed0, dim = 1)
        assertEquals(Shape(1, 1, 5, 5), bchw.shape)
    }
}
