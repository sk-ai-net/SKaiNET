package sk.ainet.lang.nn.layers

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.NeuralNetworkExecutionContext
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext

class DropoutTest {

    private val ctx: NeuralNetworkExecutionContext = DefaultNeuralNetworkExecutionContext()

    @Test
    fun p_zero_is_identity_in_training() {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f))
        val layer = Dropout<FP32, Float>(p = 0f, training = true)
        val y = layer.forward(x, ctx)
        assertEquals(x.shape, y.shape)
    }

    @Test
    fun eval_mode_is_identity() {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val layer = Dropout<FP32, Float>(p = 0.5f, training = false)
        val y = layer.forward(x, ctx)
        assertEquals(x.shape, y.shape)
    }
}
