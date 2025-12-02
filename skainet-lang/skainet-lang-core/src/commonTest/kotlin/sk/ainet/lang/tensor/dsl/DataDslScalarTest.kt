package sk.ainet.lang.tensor.dsl

import sk.ainet.context.createDataMap
import sk.ainet.context.data
import sk.ainet.lang.tensor.isScalar
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DataDslScalarTest {

    @Test
    fun `data dsl can create FP32 scalar`() {
        val t = data<FP32, Float> { _ ->
            // dtype can be omitted in typed data DSL
            scalar(3.14159f)
        }

        assertTrue(t.isScalar(), "Tensor should be rank-0 (scalar)")
        assertEquals(1, t.volume, "Scalar volume must be 1")
        // Access scalar value via zero-index vararg (no indices)
        assertEquals(3.14159f, t.data.get(), 0.0001f)
    }

    @Test
    fun `data dsl can create Int32 scalar`() {
        val t = data<Int32, Int> { _ ->
            scalar(42, Int32::class)
        }

        assertTrue(t.isScalar(), "Tensor should be rank-0 (scalar)")
        assertEquals(1, t.volume, "Scalar volume must be 1")
        assertEquals(42, t.data.get())
    }

    @Test
    fun `createDataMap requires names for scalars too`() {
        val ex = assertFailsWith<IllegalStateException> {
            createDataMap { _ ->
                // Creating an unnamed scalar increments createdTensorsCount,
                // but does not register a name → should fail the map invariant
                scalar<FP32, Float>(1.0f, FP32::class)
            }
        }
        assertTrue(
            ex.message?.contains("All tensors in createDataMap block must be named uniquely") == true,
            "Expected error about naming requirement in createDataMap"
        )
    }
}
