package sk.ainet.lang.tensor.dsl

import sk.ainet.context.createData
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class DataDslReturnLastTensorTest {

    @Test
    fun `createData returns the last tensor created in the block`() {
        var lastCreated: Tensor<*, *>? = null

        val result = createData { _ ->
            // First tensor
            tensor<FP32, Float> {
                shape(2, 2) { ones() }
            }
            // Second tensor (should be returned by createData)
            val t2 = tensor<FP32, Float> {
                shape(3) { zeros() }
            }
            lastCreated = t2
        }

        // The returned tensor should be exactly the last created one
        assertSame(lastCreated, result, "createData should return the last tensor created inside the block")
        // Basic sanity on returned tensor
        assertEquals(1, result.rank)
        assertEquals(3, result.volume)
    }

    @Test
    fun `createData throws when no tensor was created in the block`() {
        val ex = assertFailsWith<IllegalStateException> {
            createData { _ ->
                // No tensor created here
            }
        }
        assertTrue(
            ex.message?.contains("No tensor was created in createData block") == true,
            "Exception message should indicate that no tensor was created"
        )
    }
}
