package sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.relu

/**
 * Micrograd-style scalar arithmetic chain executed on Direct CPU backend.
 * Verifies the forward pass evaluates to 24.7041.
 */
class MicrogradArithmeticDirectCpuTest {

    private fun scalar(ctx: DirectCpuExecutionContext, v: Float): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(1), FP32::class, floatArrayOf(v))

    @Test
    fun micrograd_like_scalar_chain_produces_expected_value() {
        val ctx = DirectCpuExecutionContext()


        // Inputs as scalars (Shape(1))
        val a = scalar(ctx, -4f)
        val b = scalar(ctx, 2f)

        val one = scalar(ctx, 1f)
        val two = scalar(ctx, 2f)
        val three = scalar(ctx, 3f)
        val ten = scalar(ctx, 10f)
        val zero = scalar(ctx, 0f)

        var c = a + b
        // d = a * b + b^3  (no pow op; b^3 = b * b * b)
        var d = a * b + (b * b * b)
        // c += c + 1  -> c = c + c + 1
        c = c + c + one
        // c += 1 + c + (-a)  -> (-a) as zero - a
        val negA = zero - a
        c = c + one + c + negA
        // d += d * 2 + relu(b + a)
        d = d + (d * two) + (b + a).relu()
        // d += d * 3 + relu(b - a)
        d = d + (d * three) + (b - a).relu()
        // e = c - d
        val e = c - d
        // f = e^2 (no pow; e * e)
        val f = e * e
        // g = f / 2
        var g = f / two
        // g += 10 / f
        g += (ten / f)
        // Extract scalar and assert
        val out = g.data[0]
        assertEquals(24.7041f, out, 1e-4f)
    }
}
