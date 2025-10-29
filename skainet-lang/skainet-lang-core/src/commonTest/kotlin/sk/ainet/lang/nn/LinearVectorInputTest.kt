package sk.ainet.lang.nn

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

class LinearVectorInputTest {
    @Test
    fun linear_accepts_1D_input_vector_and_produces_1D_output() {
        data {
            // Build a simple Linear layer: in=3, out=2
            val weights = tensor<FP32, Float> {
                shape(2, 3) { init { (it[0] * 3 + it[1] + 1).toFloat() } }
            }
            val bias = tensor<FP32, Float> {
                shape(1, 2) { init { if (it[1] == 0) 10f else 20f } }
            }

            val layer = Linear(inFeatures = 3, outFeatures = 2, initWeights = weights, initBias = bias)

            // 1D input vector of size 3 -> [1,2,3]
            val x = tensor<FP32, Float> { shape(3) { init { (it[0] + 1).toFloat() } } }

            // Expected: y = x @ W^T + b
            // W = [[1,2,3],[4,5,6]] from formula above
            // x @ W^T = [14, 32]; + b = [24, 52]
            val y = layer(x)
            // In core module with VoidTensorOps, numerical values may be zeros. We only assert shape behavior.
            assertEquals(Shape(2), y.shape)
            assertEquals(1, y.rank)
        }
    }
}
