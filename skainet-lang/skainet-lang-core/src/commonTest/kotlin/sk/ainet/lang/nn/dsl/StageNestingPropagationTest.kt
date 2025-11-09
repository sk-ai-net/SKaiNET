package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.reflection.Summary
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Verifies that nested stages correctly receive the parent's lastDimension as input
 * and propagate their own lastDimension back to the parent.
 */
class StageNestingPropagationTest {

    @Test
    fun testNestedStagesPropagateLastDimension() {
        val model = definition<FP32, Float> {
            network {
                input(10)
                stage("A") {
                    dense(outputDimension = 20, id = "A_dense") {
                        // deterministic init for shape-only checks
                        weights { ones() }
                        bias { zeros() }
                    }
                    stage("B") {
                        dense(outputDimension = 30, id = "B_dense") {
                            weights { ones() }
                            bias { zeros() }
                        }
                    }
                }
                // If stage propagation works, this dense must see input=30
                dense(outputDimension = 5, id = "out_dense") {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }

        val summary = Summary<FP32, Float>()
        val nodes = summary.summary(model, Shape(1, 10), FP32::class)

        // Expect three linear layers with shapes: 10->20, 20->30, 30->5
        assertEquals(3, nodes.size, "Expected three linear layers produced across nested stages")

        // Ensure layers are correctly named so we can refer to them in asserts
        assertEquals("A_dense", nodes[0].name)
        assertEquals("B_dense", nodes[1].name)
        assertEquals("out_dense", nodes[2].name)

        // Check parameter counts per layer to validate in/out features wiring
        // layer1: 20*10 weights + 20 bias = 220
        assertEquals(220L, nodes[0].params, "First layer should be 10->20 (params=220)")
        // layer2: 30*20 weights + 30 bias = 630
        assertEquals(630L, nodes[1].params, "Second layer should be 20->30 (params=630)")
        // layer3: 5*30 weights + 5 bias = 155
        assertEquals(155L, nodes[2].params, "Third layer should be 30->5 (params=155)")

        // Additionally verify the final output shape is [1,5]
        assertEquals(Shape(intArrayOf(1, 5)), nodes.last().output)

        // Sanity: ensure inputs progress as expected
        assertEquals(Shape(intArrayOf(1, 10)), nodes[0].input)
        assertEquals(Shape(intArrayOf(1, 20)), nodes[1].input)
        assertEquals(Shape(intArrayOf(1, 30)), nodes[2].input)

        // And outputs progress accordingly
        assertEquals(Shape(intArrayOf(1, 20)), nodes[0].output)
        assertEquals(Shape(intArrayOf(1, 30)), nodes[1].output)
        assertEquals(Shape(intArrayOf(1, 5)), nodes[2].output)

        // If all assertions pass, stage nesting and lastDimension propagation are working.
        assertTrue(true)
    }
}
