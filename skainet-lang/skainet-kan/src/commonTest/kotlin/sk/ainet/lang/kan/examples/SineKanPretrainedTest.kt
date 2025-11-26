package sk.ainet.lang.kan.examples

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class SineKanPretrainedTest {

    @Test
    fun `pretrained sine matches math sin within tolerance`() {
        val ctx = DirectCpuExecutionContext()
        val model = SineKanPretrained.create(ctx)

        val angles = listOf(0.0f, (PI / 6).toFloat(), (PI / 4).toFloat(), (PI / 3).toFloat(), (PI / 2).toFloat())
        val tol = 0.001

        angles.forEach { angle ->
            val input = ctx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(angle))
            val output = model.forward(input, ctx)
            val predicted = output.data[0]
            val expected = sin(angle.toDouble()).toFloat()
            val delta = abs(predicted - expected)
            assertTrue(delta <= tol, "angle=$angle expected=$expected got=$predicted delta=$delta")
        }
    }
}
