package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertFailsWith
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.createDataMap
import sk.ainet.execute.context.computation
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.operators.withOps
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor

/**
 * Demonstrates creating a map of named tensors with a specific ExecutionContext (CPU)
 * and then using those tensors inside a computation block that uses the same context.
 */
class DataMapExecutionContextInteropTest {

    @Test
    fun createDataMap_withDirectCpuExecutionContext_thenUseInComputation() {
        val exec = DirectCpuExecutionContext()

        val data = createDataMap(exec) { _ ->
            // Create two named FP32 tensors on the same CPU execution context
            tensor<FP32, Float>("a") { shape(2, 2) { ones() } }   // [[1,1],[1,1]]
            tensor<FP32, Float>("b") { shape(2, 2) { ones() } }   // [[1,1],[1,1]]
        }

        // Map contains our named tensors
        assertEquals(setOf("a", "b"), data.keys)
        val a = requireNotNull(data["a"]) { "Tensor 'a' missing from data map" }
        val b = requireNotNull(data["b"]) { "Tensor 'b' missing from data map" }

        // Now operate on them inside a computation using the SAME execution context
        computation(exec) {
            // Bind ops explicitly because these tensors were created in the data DSL (no ops bound there)
            val aT = a as Tensor<FP32, Float>
            val bT = b as Tensor<FP32, Float>
            val aBound: Tensor<FP32, Float> = aT.withOps(exec.ops)
            val bBound: Tensor<FP32, Float> = bT.withOps(exec.ops)

            val sum: Tensor<FP32, Float> = aBound + bBound // elementwise add

            // Expect a 2x2 tensor of 2's
            assertEquals(2f, sum.data[0, 0]); assertEquals(2f, sum.data[0, 1])
            assertEquals(2f, sum.data[1, 0]); assertEquals(2f, sum.data[1, 1])
        }
    }

    @Test
    fun createDataMap_requires_all_tensors_named() {
        val exec = DirectCpuExecutionContext()
        val ex = assertFailsWith<IllegalStateException> {
            createDataMap(exec) { _ ->
                tensor<FP32, Float>("named") { shape(1) { ones() } }
                // Unnamed tensor inside createDataMap block triggers error
                tensor<FP32, Float> { shape(1) { ones() } }
            }
        }
        // The error message from ContextDsl enforces all tensors are named uniquely
        // when using createDataMap
        assertNotNull(ex.message)
    }
}
