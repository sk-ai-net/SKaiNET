package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.execute.context.computation
import sk.ainet.execute.context.dsl.tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.pprint
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Tests for all currently implemented CPU ops using the execute DSL syntax.
 * Implemented ops in DefaultCpuOps: add, subtract, multiply, divide.
 */
class DefaultCpuOpsExecutionDslTest {

    @Test
    fun fp32_sameShape_ops_with_init() {
        val ctx = DirectCpuExecutionContext()
        computation(ctx) {
            val a = tensor<FP32, Float> { shape(3) { init { idx -> (idx[0] + 1).toFloat() } } } // [1,2,3]
            val b = tensor<FP32, Float> { shape(3) { init { idx -> (idx[0] * 2 + 4).toFloat() } } } // [4,6,8]

            val add = a + b
            val sub = b - a
            val mul = a * b
            val div = b / a

            assertEquals(Shape(3), add.shape)
            // add -> [5,8,11]
            assertEquals(5f, add.data[0]); assertEquals(8f, add.data[1]); assertEquals(11f, add.data[2])
            // subtract -> [3,4,5]
            assertEquals(3f, sub.data[0]); assertEquals(4f, sub.data[1]); assertEquals(5f, sub.data[2])
            // multiply -> [4,12,24]
            assertEquals(4f, mul.data[0]); assertEquals(12f, mul.data[1]); assertEquals(24f, mul.data[2])
            // divide -> [4/1, 6/2, 8/3]
            assertEquals(4f, div.data[0]); assertEquals(3f, div.data[1]); assertEquals(8f / 3f, div.data[2])
        }
    }

    @Test
    fun fp32_broadcast_scalar_vector_and_vector_matrix() {
        val ctx = DirectCpuExecutionContext()
        computation(ctx) {
            // scalar (represented as shape(1)) times vector
            val scalar = tensor<FP32, Float> { shape(1) { ones() } } // [1]
            val vec = tensor<FP32, Float> { shape(4) { init { i -> (i[0] + 1).toFloat() } } } // [1,2,3,4]
            val r1 = scalar * vec
            assertEquals(Shape(4), r1.shape)
            assertEquals(1f, r1.data[0]); assertEquals(2f, r1.data[1]); assertEquals(3f, r1.data[2]); assertEquals(
            4f,
            r1.data[3]
        )

            // vector + matrix (broadcast along rows)
            val mat = tensor<FP32, Float> {
                shape(2, 3) { init { (it[0] * 3 + it[1] + 1).toFloat() } }
            } // [[1,2,3],[4,5,6]]
            val addVec = tensor<FP32, Float> { shape(3) { init { i -> (10 - i[0]).toFloat() } } } // [10,9,8]
            println(mat.pprint())
            println("+")
            println(addVec.pprint())
            val r2 = mat + addVec
            println("=")
            println(r2.pprint())
            // Row 0 -> [11,11,11], Row 1 -> [14,14,14]
            assertEquals(11f, r2.data[0, 0]); assertEquals(11f, r2.data[0, 1]); assertEquals(11f, r2.data[0, 2])
            assertEquals(14f, r2.data[1, 0]); assertEquals(14f, r2.data[1, 1]); assertEquals(14f, r2.data[1, 2])
        }
    }

    @Test
    fun fp32_zeros_ones_full_mix() {
        val ctx = DirectCpuExecutionContext()
        computation(ctx) {
            val zeros = tensor<FP32, Float> { shape(2, 2) { zeros() } }
            val ones = tensor<FP32, Float> { shape(2, 2) { ones() } }
            val twos = ones + ones
            val rAdd = zeros + ones
            val rSub = twos - ones
            val rMul = twos * zeros
            // Avoid division by zero for div
            val three = twos + ones
            val rDiv = three / ones

            // Assertions
            // rAdd -> ones
            assertEquals(1f, rAdd.data[0, 0]); assertEquals(1f, rAdd.data[0, 1]); assertEquals(
            1f,
            rAdd.data[1, 0]
        ); assertEquals(1f, rAdd.data[1, 1])
            // rSub -> ones
            assertEquals(1f, rSub.data[0, 0]); assertEquals(1f, rSub.data[0, 1]); assertEquals(
            1f,
            rSub.data[1, 0]
        ); assertEquals(1f, rSub.data[1, 1])
            // rMul -> zeros
            assertEquals(0f, rMul.data[0, 0]); assertEquals(0f, rMul.data[0, 1]); assertEquals(
            0f,
            rMul.data[1, 0]
        ); assertEquals(0f, rMul.data[1, 1])
            // rDiv -> three
            assertEquals(3f, rDiv.data[0, 0]); assertEquals(3f, rDiv.data[0, 1]); assertEquals(
            3f,
            rDiv.data[1, 0]
        ); assertEquals(3f, rDiv.data[1, 1])
        }
    }

    @Test
    fun int32_sameShape_and_division_behavior() {
        val ctx = DirectCpuExecutionContext()
        computation(ctx) {
            val a = tensor<Int32, Int> { shape(4) { init { i -> i[0] + 1 } } } // [1,2,3,4]
            val b = tensor<Int32, Int> { shape(4) { init { i -> (i[0] + 1) * 3 } } } // [3,6,9,12]

            val add = a + b // [4,8,12,16]
            val sub = b - a // [2,4,6,8]
            val mul = a * b // [3,12,27,48]
            val div = b / a // [3,3,3,3]

            assertEquals(Shape(4), add.shape)
            assertEquals(4, add.data[0]); assertEquals(8, add.data[1]); assertEquals(12, add.data[2]); assertEquals(
            16,
            add.data[3]
        )
            assertEquals(2, sub.data[0]); assertEquals(4, sub.data[1]); assertEquals(6, sub.data[2]); assertEquals(
            8,
            sub.data[3]
        )
            assertEquals(3, mul.data[0]); assertEquals(12, mul.data[1]); assertEquals(27, mul.data[2]); assertEquals(
            48,
            mul.data[3]
        )
            assertEquals(3, div.data[0]); assertEquals(3, div.data[1]); assertEquals(3, div.data[2]); assertEquals(
            3,
            div.data[3]
        )
        }
    }

    @Test
    fun int32_division_by_zero_yields_zero() {
        val ctx = DirectCpuExecutionContext()
        computation(ctx) {
            val numerator = tensor<Int32, Int> { shape(3) { init { i -> (i[0] + 1) * 10 } } } // [10,20,30]
            val denom = tensor<Int32, Int> { shape(3) { init { i -> if (i[0] == 1) 0 else 2 } } } // [2,0,2]
            val r = numerator / denom
            assertEquals(Shape(3), r.shape)
            assertEquals(5, r.data[0]) // 10/2
            assertEquals(0, r.data[1]) // div by zero -> 0 (per implementation)
            assertEquals(15, r.data[2]) // 30/2
        }
    }
}
