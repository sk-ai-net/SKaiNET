package sk.ainet.lang.tensor.dsl

import sk.ainet.context.createDataMap
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class DataDslCreateDataMapTest {

    @Test
    fun `createDataMap returns hashmap of all named tensors`() {
        val map = createDataMap { _ ->
            tensor<FP32, Float>(name = "w1") { shape(2, 2) { ones() } }
            tensor<FP32, Float>(name = "b1") { shape(2) { zeros() } }
            tensor<FP32, Float>(name = "w2") { shape(3, 2) { ones() } }
        }

        assertEquals(3, map.size, "Map should contain all created tensors")
        assertTrue(map.containsKey("w1"))
        assertTrue(map.containsKey("b1"))
        assertTrue(map.containsKey("w2"))
        // basic sanity on values
        val t: Tensor<*, *>? = map["b1"]
        assertNotNull(t)
        assertEquals(1, t.rank)
        assertEquals(2, t.volume)
    }

    @Test
    fun `createDataMap throws if a tensor is unnamed`() {
        val ex = assertFailsWith<IllegalStateException> {
            createDataMap { _ ->
                tensor<FP32, Float>(name = "a") { shape(1) { zeros() } }
                // unnamed tensor
                tensor<FP32, Float> { shape(1) { ones() } }
            }
        }
        assertEquals(
            ex.message?.contains("All tensors in createDataMap block must be named uniquely"),
            true,
            "Should require that all tensors in the block are named"
        )
    }

    @Test
    fun `createDataMap throws on duplicate names`() {
        val ex = assertFailsWith<IllegalStateException> {
            createDataMap { _ ->
                tensor<FP32, Float>(name = "dup") { shape(1) { zeros() } }
                tensor<FP32, Float>(name = "dup") { shape(1) { ones() } }
            }
        }
        assertEquals(ex.message?.contains("must be unique"), true, "Should indicate that tensor names must be unique")
    }

    @Test
    fun `createDataMap throws when no tensor was created`() {
        val ex = assertFailsWith<IllegalStateException> {
            createDataMap { _ ->
                // no tensors
            }
        }
        assertEquals(
            ex.message?.contains("No tensor was created in createData block"),
            true,
            "Exception message should indicate that no tensor was created"
        )
    }
}
