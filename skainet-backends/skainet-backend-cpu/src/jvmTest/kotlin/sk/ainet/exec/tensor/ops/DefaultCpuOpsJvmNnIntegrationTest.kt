package sk.ainet.exec.tensor.ops

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

class DefaultCpuOpsJvmNnIntegrationTest {

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun linear_forward_matches_manual_vector_on_and_off() {
        // Configure once: this test validates both vector=true and vector=false produce same numerics
        fun runOnce(vectorFlag: Boolean) {
            System.setProperty("skainet.cpu.vector.enabled", vectorFlag.toString())

            data(DirectCpuExecutionContext())  { ctx ->
                // in=3, out=2
                val weights = tensor<FP32, Float> {
                    // Rows = outFeatures, Cols = inFeatures
                    shape(2, 3) { init { (it[0] * 3 + it[1] + 1).toFloat() } } // [[1,2,3],[4,5,6]]
                }
                val bias = tensor<FP32, Float> {
                    shape(1, 2) { init { if (it[1] == 0) 10f else 20f } } // [10,20]
                }
                val layer = Linear(inFeatures = 3, outFeatures = 2, initWeights = weights, initBias = bias)

                // Case 1: 1D input
                val x = tensor<FP32, Float> { shape(3) { init { (it[0] + 1).toFloat() } } } // [1,2,3]
                val y = layer.forward(x, ctx)
                // x @ W^T = [14,32]; + b = [24,52]
                assertEquals(Shape(2), y.shape)
                assertEquals(24f, y.data[0], 1e-5f)
                assertEquals(52f, y.data[1], 1e-5f)

                // Case 2: 2D batch input [B=2, in=3]
                val xb = tensor<FP32, Float> {
                    shape(2, 3) { init { (1 + it[0] * 3 + it[1]).toFloat() } } // [[1,2,3],[4,5,6]]
                }
                val yb = layer.forward(xb, ctx)
                // Manual:
                // Row0: [1,2,3] -> [24,52]
                // Row1: [4,5,6] -> [(4+10+18)+10=42, (16+25+36)+20=97]
                val expected = floatArrayOf(24f, 52f, 42f, 97f)
                assertEquals(Shape(2, 2), yb.shape)
                val out = floatArrayOf(yb.data[0, 0], yb.data[0, 1], yb.data[1, 0], yb.data[1, 1])
                for (i in expected.indices) assertEquals(expected[i], out[i], 1e-5f)
            }
        }

        runOnce(vectorFlag = true)
        runOnce(vectorFlag = false)
    }
}
