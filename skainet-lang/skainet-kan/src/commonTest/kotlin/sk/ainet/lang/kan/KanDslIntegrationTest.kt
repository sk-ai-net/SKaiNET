package sk.ainet.lang.kan

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class KanDslIntegrationTest {

    private val ctx = DefaultNeuralNetworkExecutionContext()

    @Test
    fun `kanLayer builds and runs with DSL`() {
        val model = definition<FP32, Float> {
            sequential {
                input(4)
                kanLayer(outputDim = 3, gridSize = 2) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }

        // MLP wrapper should contain a KanLayer child
        val asMlp = model as MLP<FP32, Float>
        assertTrue(asMlp.modules.any { it is KanLayer<*, *> }, "KAN layer should be present in MLP modules")

        val input = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val output = model.forward(input, ctx)

        assertEquals(Shape(3), output.shape, "KAN output should match requested outputDim")
    }
}
