package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SoftmaxTest {
    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun make(shape: Shape, fill: Float = 0f): Tensor<FP32, Float> =
        tensor(ctx, FP32::class) { tensor { shape(shape) { full(fill) } } }

    private fun approx(a: Float, b: Float, eps: Float = 1e-5f) = kotlin.math.abs(a - b) <= eps

    @Test
    fun testSoftmaxModule_PreservesShape() {
        val mod = Softmax<FP32, Float>(dimension = -1)
        val input = make(Shape(2, 3, 4), fill = 0.5f)
        val out = mod.forward(input, ctx)
        assertEquals(input.shape, out.shape)
        assertEquals("Softmax", mod.name)
        assertTrue(mod.modules.isEmpty())
    }

    // Note: Numeric normalization tests are executed in the CPU backend suite.
    // Here we only verify DSL wiring and shapes using the default (void) context.

    @Test
    fun testSoftmax_DSLIntegration() {
        val model = definition<FP32, Float> {
            network {
                input(6)
                dense(3) {
                    // defaults are zeros; fine for building test
                }
                softmax(dim = -1)
            }
        }
        assertNotNull(model)
        val x = make(Shape(2, 6), fill = 0.0f)
        val y = model.forward(x, ctx)
        assertEquals(Shape(2, 3), y.shape)
    }
}
