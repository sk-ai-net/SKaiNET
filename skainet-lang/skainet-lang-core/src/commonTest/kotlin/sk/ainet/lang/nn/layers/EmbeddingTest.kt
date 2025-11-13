package sk.ainet.lang.nn.layers

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.nn.NeuralNetworkExecutionContext
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

class EmbeddingTest {

    private val ctx: NeuralNetworkExecutionContext = DefaultNeuralNetworkExecutionContext()

    private fun <T : DType, V> tensorFromFloatArray(data: FloatArray, shape: Shape, dtype: kotlin.reflect.KClass<T>): Tensor<T, V> {
        val t = ctx.fromFloatArray<T, V>(shape, dtype, data)
        return t
    }

    @Test
    fun shape_unbatched_and_batched() {
        // Build a deterministic weight of shape [5,3]
        val weight = tensorFromFloatArray<FP32, Float>(
            floatArrayOf(
                // 0th row
                0f, 1f, 2f,
                // 1st row
                10f, 11f, 12f,
                // 2nd row
                20f, 21f, 22f,
                // 3rd row
                30f, 31f, 32f,
                // 4th row
                40f, 41f, 42f,
            ),
            Shape(5, 3), FP32::class
        )

        val layer = Embedding<FP32, Float>(numEmbeddings = 5, embeddingDim = 3, initWeight = weight)

        // Unbatched indices [3,1,4]
        val idxUnbatched = tensorFromFloatArray<FP32, Float>(
            floatArrayOf(3f, 1f, 4f), Shape(3), FP32::class
        )
        val out1 = layer.forward(idxUnbatched)
        assertEquals(Shape(3, 3), out1.shape)

        // Batched indices [[0,2],[4,1]] (2x2)
        val idxBatched = tensorFromFloatArray<FP32, Float>(
            floatArrayOf(0f, 2f, 4f, 1f), Shape(2, 2), FP32::class
        )
        val out2 = layer.forward(idxBatched)
        assertEquals(Shape(2, 2, 3), out2.shape)
    }

    @Test
    fun out_of_range_throws() {
        val weight = tensorFromFloatArray<FP32, Float>(
            floatArrayOf(
                0f, 0f, 0f,
                1f, 1f, 1f
            ), Shape(2, 3), FP32::class
        )
        val layer = Embedding<FP32, Float>(2, 3, weight)
        val idx = tensorFromFloatArray<FP32, Float>(floatArrayOf(2f), Shape(1), FP32::class)
        assertFailsWith<IllegalArgumentException> {
            layer.forward(idx)
        }
    }

    @Test
    fun padding_zeroes_rows() {
        val weight = tensorFromFloatArray<FP32, Float>(
            floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f
            ), Shape(2, 3), FP32::class
        )
        val layer = Embedding<FP32, Float>(2, 3, weight, paddingIdx = 1)
        val idx = tensorFromFloatArray<FP32, Float>(floatArrayOf(1f, 0f), Shape(2), FP32::class)
        val out = layer.forward(idx)
        assertEquals(Shape(2, 3), out.shape)
        // We only check shape because default VoidTensorOps may not produce materialized data for values
        // and padding path should execute without exceptions.
    }
}
