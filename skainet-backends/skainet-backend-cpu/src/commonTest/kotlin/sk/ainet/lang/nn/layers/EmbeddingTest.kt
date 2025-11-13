package sk.ainet.lang.nn.layers

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.NonIntegralIndexException
import sk.ainet.lang.types.FP32

class EmbeddingTest {

    private fun makeWeights(exec: DirectCpuExecutionContext): Tensor<FP32, Float> {
        // 4 x 3 matrix with simple, distinct values per row
        val data = floatArrayOf(
            // row 0
            0.1f, 0.2f, 0.3f,
            // row 1
            1.0f, 1.1f, 1.2f,
            // row 2
            2.0f, 2.1f, 2.2f,
            // row 3
            3.0f, 3.1f, 3.2f,
        )
        return exec.fromFloatArray(Shape(4, 3), FP32::class, data)
    }

    @Test
    fun embedding_withIntIndices_returnsExpectedRows() {
        val exec = DirectCpuExecutionContext()
        val weight = makeWeights(exec)
        val emb = Embedding(
            numEmbeddings = 4,
            embeddingDim = 3,
            initWeight = weight,
            paddingIdx = null,
            name = "emb"
        )

        val out: Tensor<FP32, Float> = emb.forward(intArrayOf(2, 1, 3), exec) as Tensor<FP32, Float>
        // Expect rows [2,1,3] stacked: shape [3,3]
        assertEquals(3, out.shape[0])
        assertEquals(3, out.shape[1])

        // row 0 from embedding row 2
        assertEquals(2.0f, out.data[0, 0]); assertEquals(2.1f, out.data[0, 1]); assertEquals(2.2f, out.data[0, 2])
        // row 1 from embedding row 1
        assertEquals(1.0f, out.data[1, 0]); assertEquals(1.1f, out.data[1, 1]); assertEquals(1.2f, out.data[1, 2])
        // row 2 from embedding row 3
        assertEquals(3.0f, out.data[2, 0]); assertEquals(3.1f, out.data[2, 1]); assertEquals(3.2f, out.data[2, 2])
    }

    @Test
    fun embedding_forwardAny_rejectsNonIntegralFloats_whenStrict() {
        val exec = DirectCpuExecutionContext()
        val weight = makeWeights(exec)
        val emb = Embedding(
            numEmbeddings = 4,
            embeddingDim = 3,
            initWeight = weight,
            paddingIdx = null,
            name = "emb"
        )
        // Float indices with a non-integral 1.5 should be rejected in strict mode
        val idxFloat: Tensor<FP32, Float> = exec.fromFloatArray(Shape(3), FP32::class, floatArrayOf(0f, 1.5f, 2f))
        assertFailsWith<NonIntegralIndexException> {
            emb.forwardAny(idxFloat, exec, strict = true)
        }
    }

    @Test
    fun embedding_withPaddingIndex_producesZeroRow() {
        val exec = DirectCpuExecutionContext()
        val base = makeWeights(exec)
        val emb = Embedding(
            numEmbeddings = 4,
            embeddingDim = 3,
            initWeight = base,
            paddingIdx = 1, // row 1 will be zeroed when selected
            name = "emb"
        )

        val out: Tensor<FP32, Float> = emb.forward(intArrayOf(1, 2), exec) as Tensor<FP32, Float>
        // First output row corresponds to padding index -> zeros
        assertEquals(0f, out.data[0, 0]); assertEquals(0f, out.data[0, 1]); assertEquals(0f, out.data[0, 2])
        // Second output row corresponds to row 2 in weights (not zero)
        assertEquals(2.0f, out.data[1, 0]); assertEquals(2.1f, out.data[1, 1]); assertEquals(2.2f, out.data[1, 2])
    }
}
