package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Shape
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.tensor

class DefaultCpuOpsReductionOpsTest {

    private val ctx = DirectCpuExecutionContext()
    private val ops = ctx.ops

    @Test
    fun sum_all_dims_scalar_and_values() {
        data(ctx) { _ ->
            // Build 2x3x4 tensor with values = i*100 + j*10 + k
            val t = tensor<FP32, Float> { shape(2, 3, 4) { init { (it[0] * 100 + it[1] * 10 + it[2]).toFloat() } } }
            val r = ops.sum(t, null)
            // Sum all numbers from pattern
            var expected = 0f
            for (i in 0 until 2) for (j in 0 until 3) for (k in 0 until 4) {
                expected += (i * 100 + j * 10 + k).toFloat()
            }
            // Scalar (rank-0) expected by backend implementation
            assertEquals(0, r.shape.rank)
            assertEquals(expected, r.data.get())
        }
    }

    @Test
    fun sum_specific_dim_and_negative_dim() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 3, 4) { init { (1 + it[0] + it[1] + it[2]).toFloat() } } }
            // Sum along dim 1 -> shape (2,4)
            val r1 = ops.sum(t, 1)
            assertEquals(Shape(2, 4), r1.shape)
            // Manually verify one position [i,k]
            val i = 1; val k = 2
            var expectedR1 = 0f
            for (j in 0 until 3) expectedR1 += (1 + i + j + k)
            assertEquals(expectedR1, r1.data[i, k])

            // Sum along last dim (-1) -> shape (2,3)
            val r2 = ops.sum(t, -1)
            assertEquals(Shape(2, 3), r2.shape)
            val j = 2
            var expectedR2 = 0f
            for (kk in 0 until 4) expectedR2 += (1 + i + j + kk)
            assertEquals(expectedR2, r2.data[i, j])
        }
    }

    @Test
    fun mean_all_dims_and_dim() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 2) { init { (it[0] * 10 + it[1]).toFloat() } } }
            val rAll = ops.mean(t, null)
            // rank-0 scalar
            assertEquals(0, rAll.shape.rank)
            val expectedAll = listOf(0f,1f,10f,11f).average().toFloat()
            assertEquals(expectedAll, rAll.data.get() as Float)

            // Mean along dim 0 -> shape (2)
            val r0 = ops.mean(t, 0)
            assertEquals(Shape(2), r0.shape)
            // column-wise mean
            assertEquals(((0f + 10f) / 2f), r0.data[0])
            assertEquals(((1f + 11f) / 2f), r0.data[1])
        }
    }

    @Test
    fun variance_all_dims_and_dim() {
        data(ctx) { _ ->
            // 1D simple range to make variance easy
            val t = tensor<FP32, Float> { shape(4) { init { it[0].toFloat() } } } // [0,1,2,3]
            val rAll = ops.variance(t, null)
            // rank-0 scalar
            assertEquals(0, rAll.shape.rank)
            // population variance of [0,1,2,3] is 1.25
            assertEquals(1.25f, rAll.data.get() as Float)

            // 2x2; variance along dim 1 (each row)
            val m = tensor<FP32, Float> { shape(2, 2) { init { (it[0] * 10 + it[1]).toFloat() } } } // [[0,1],[10,11]]
            val r1 = ops.variance(m, 1)
            assertEquals(Shape(2), r1.shape)
            // For [0,1] mean=0.5 var=((0^2+1^2)/2 - 0.5^2)=((0+1)/2 - 0.25)=0.5-0.25=0.25
            assertEquals(0.25f, r1.data[0])
            // For [10,11] same variance
            assertEquals(0.25f, r1.data[1])
        }
    }

    @Test
    fun reductions_invalid_dim_throws() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 3) { zeros() } }
            assertFailsWith<IllegalArgumentException> { ops.sum(t, 5) }
            assertFailsWith<IllegalArgumentException> { ops.mean(t, -4) }
            assertFailsWith<IllegalArgumentException> { ops.variance(t, 3) }
        }
    }
}
