package sk.ainet.lang.kan.examples

import kotlin.test.Test
import kotlin.test.assertTrue

class SineKanPretrainedGraphvizTest {

    @Test
    fun toGraphvizDot_producesReasonableDot() {
        val dot = SineKanPretrained.toGraphvizDot()

        print(dot)

        // Basic DOT structure
        assertTrue(dot.contains("digraph"), "DOT should start with digraph")
        // Nodes
        assertTrue(dot.contains("input"), "Should contain input node")
        assertTrue(dot.contains("sine_kan_pretrained"), "Should contain KAN node id")
        // Edge
        assertTrue(dot.contains("input -> sine_kan_pretrained"), "Should contain edge from input to KAN")
        // Some metadata
        assertTrue(dot.contains("gridSize: 16"), "Should mention grid size")
        assertTrue(dot.contains("degree: 3"), "Should mention degree")
    }
}
