package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

class DefaultCpuOpsUnsqueezeExtraTest {
    private val ctx = DirectCpuExecutionContext()
    private val ops get() = ctx.ops

    @Test
    fun unsqueeze_preserves_values_positive_and_negative_dims() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 2) { init { (it[0] * 10 + it[1]).toFloat() } } }

            val uMid = ops.unsqueeze(t, dim = 1)
            assertEquals(Shape(2, 1, 2), uMid.shape)
            // spot-check mapping
            assertEquals(t.data[0, 1], uMid.data[0, 0, 1])
            assertEquals(t.data[1, 0], uMid.data[1, 0, 0])

            val uEnd = ops.unsqueeze(t, dim = -1)
            assertEquals(Shape(2, 2, 1), uEnd.shape)
            assertEquals(t.data[1, 1], uEnd.data[1, 1, 0])
        }
    }

    @Test
    fun unsqueeze_scalar_creates_rank2_shape_and_keeps_value() {
        data(ctx) { _ ->
            val s = tensor<FP32, Float> { shape(1) { init { 42f } } }
            val u = ops.unsqueeze(s, 0)
            assertEquals(Shape(1, 1), u.shape)
            assertEquals(42f, u.data[0, 0])
        }
    }
}
