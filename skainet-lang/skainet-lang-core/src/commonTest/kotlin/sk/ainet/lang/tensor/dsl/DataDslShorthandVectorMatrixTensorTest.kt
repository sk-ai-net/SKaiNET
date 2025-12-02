package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DataDslShorthandVectorMatrixTensorTest {

    @Test
    fun `typed data dsl shorthand - vector without explicit dtype`() {
        val t = data<FP32, Float> { _ ->
            vector(length = 3) { from(1f, 2f, 3f) }
        }

        assertEquals(1, t.rank, "Vector should be rank-1")
        assertEquals(3, t.volume, "Vector volume should match length")
        assertEquals(1.0f, t.data[0])
        assertEquals(2.0f, t.data[1])
        assertEquals(3.0f, t.data[2])
    }

    @Test
    fun `typed data dsl shorthand - matrix without explicit dtype`() {
        val t = data<FP32, Float> { _ ->
            matrix(rows = 2, columns = 2) { ones() }
        }

        assertEquals(2, t.rank, "Matrix should be rank-2")
        assertEquals(4, t.volume, "Matrix volume should be rows*columns")
        assertEquals(1.0f, t.data[0, 0])
        assertEquals(1.0f, t.data[0, 1])
        assertEquals(1.0f, t.data[1, 0])
        assertEquals(1.0f, t.data[1, 1])
    }

    @Test
    fun `typed data dsl shorthand - tensor without explicit dtype`() {
        val t = data<FP32, Float> { _ ->
            tensor { shape(2, 2) { full(7f) } }
        }

        assertEquals(2, t.rank, "Tensor should be rank-2 for shape(2,2)")
        assertEquals(Shape(2, 2), t.shape)
        for (i in 0 until 2) {
            for (j in 0 until 2) {
                assertEquals(7.0f, t.data[i, j])
            }
        }
    }
}
