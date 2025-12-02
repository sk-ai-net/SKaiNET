package sk.ainet.lang.tensor.dsl

import sk.ainet.context.createDataMap
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DataDslUntypedReifiedConvenienceTest {

    @Test
    fun `untyped data dsl - reified vector and matrix`() {
        data { _ ->
            val v = vector<FP32, Float>(length = 3) { from(1f, 2f, 3f) }
            assertEquals(1, v.rank)
            assertEquals(3, v.volume)
            assertEquals(2.0f, v.data[1])

            val m = matrix<FP32, Float>(rows = 2, columns = 3) { zeros() }
            assertEquals(2, m.rank)
            assertEquals(6, m.volume)
            assertEquals(0.0f, m.data[1, 2])
        }
    }

    @Test
    fun `createDataMap - reified named tensor`() {
        val map = createDataMap { _ ->
            tensor<FP32, Float>(name = "T") { shape(2, 2) { ones() } }
        }

        assertEquals(1, map.size)
        val t = map["T"]!!
        assertEquals(Shape(2, 2), t.shape)
        assertTrue(t.volume == 4)
    }
}
