package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.activations.Softmax
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SoftmaxCpuTest {
    private val ctx = DirectCpuExecutionContext()

    private fun make(shape: Shape, fill: Float = 0f): Tensor<FP32, Float> =
        tensor(ctx, FP32::class) { tensor { shape(shape) { full(fill) } } }

    private fun approx(a: Float, b: Float, eps: Float = 1e-5f) = kotlin.math.abs(a - b) <= eps

    @Test
    fun testTensorSoftmax_Normalization() {
        val input = make(Shape(2, 4), fill = 1.0f)
        val mod = Softmax<FP32, Float>(dimension = -1)
        val out = mod.forward(input, ctx)
        assertEquals(input.shape, out.shape)
        val sums = out.sum(dim = -1)
        assertEquals(Shape(2), sums.shape)
        val s0 = sums.data.get(0) as Float
        val s1 = sums.data.get(1) as Float
        assertTrue(approx(1.0f, s0), "Row 0 sum=$s0 not ≈ 1.0")
        assertTrue(approx(1.0f, s1), "Row 1 sum=$s1 not ≈ 1.0")
        // probabilities in [0,1]
        for (i in 0 until out.shape[0]) {
            for (j in 0 until out.shape[1]) {
                val v = out.data.get(i, j) as Float
                assertTrue(v >= 0f && v <= 1f, "prob out[$i,$j]=$v not in [0,1]")
            }
        }
    }

    @Test
    fun testDslSoftmax_WithCpuContextForward() {
        val model = definition<FP32, Float> {
            network {
                input(6)
                dense(3) { }
                softmax(dim = -1)
            }
        }
        assertNotNull(model)
        val x = make(Shape(2, 6), fill = 0.0f)
        val y = model.forward(x, ctx)
        assertEquals(Shape(2, 3), y.shape)
        val sums = y.sum(dim = -1)
        val s0 = sums.data.get(0) as Float
        val s1 = sums.data.get(1) as Float
        assertTrue(approx(1.0f, s0))
        assertTrue(approx(1.0f, s1))
    }
}
