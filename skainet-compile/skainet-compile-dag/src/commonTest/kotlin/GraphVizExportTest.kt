package sk.ainet

import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.utils.drawDot
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.BaseOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for GraphViz DOT export utilities.
 * These tests construct small graphs manually and verify the generated DOT output.
 */
class GraphVizExportTest {

    /**
     * Simple test operation used to populate GraphNode.operation in tests.
     */
    private class TestOperation(
        override val name: String,
        override val type: String,
        override val parameters: Map<String, Any> = emptyMap()
    ) : BaseOperation(name, type, parameters) {

        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> = inputs

        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid

        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs

        override fun clone(newParameters: Map<String, Any>): Operation = TestOperation(name, type, newParameters)
    }

    @Test
    fun testBasicGraphVizExport() {
        // Create a simple compute graph
        val graph = DefaultComputeGraph()

        // Create operations
        val inputOp = TestOperation("input", "input")
        val processOp = TestOperation("process", "compute", mapOf("kernel_size" to 3, "stride" to 1))
        val outputOp = TestOperation("output", "output")

        // Create nodes
        val inputNode = GraphNode(
            id = "input_node",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(1, 10), "FP32"))
        )

        val processNode = GraphNode(
            id = "process_node",
            operation = processOp,
            inputs = listOf(TensorSpec("process_in", listOf(1, 10), "FP32")),
            outputs = listOf(TensorSpec("process_out", listOf(1, 5), "FP32"))
        )

        val outputNode = GraphNode(
            id = "output_node",
            operation = outputOp,
            inputs = listOf(TensorSpec("output_in", listOf(1, 5), "FP32")),
            outputs = listOf(TensorSpec("output_out", listOf(1, 5), "FP32"))
        )

        // Add nodes and edges
        graph.addNode(inputNode)
        graph.addNode(processNode)
        graph.addNode(outputNode)

        graph.addEdge(GraphEdge("edge1", inputNode, processNode, 0, 0, inputNode.outputs.first()))
        graph.addEdge(GraphEdge("edge2", processNode, outputNode, 0, 0, processNode.outputs.first()))

        // Render DOT
        val dotGraph = drawDot(graph)
        assertNotNull(dotGraph)
        assertTrue(dotGraph.content.isNotEmpty(), "DOT content should not be empty")

        // Verify structure
        assertTrue(dotGraph.content.contains("digraph {"))
        assertTrue(dotGraph.content.contains("rankdir=LR"))
        assertTrue(dotGraph.content.contains("input_node"))
        assertTrue(dotGraph.content.contains("process_node"))
        assertTrue(dotGraph.content.contains("output_node"))
        assertTrue(dotGraph.content.contains("->"))
        assertTrue(dotGraph.content.contains("}"))

        // Verify parameters are present for process node
        assertTrue(dotGraph.content.contains("kernel_size"))
        assertTrue(dotGraph.content.contains("stride"))
    }

    @Test
    fun testSubsetGraphVizExport() {
        val graph = DefaultComputeGraph()

        val node1 = GraphNode(
            "node1",
            TestOperation("op1", "type1"),
            emptyList(),
            listOf(TensorSpec("out1", listOf(1), "FP32"))
        )
        val node2 = GraphNode(
            "node2",
            TestOperation("op2", "type2"),
            listOf(TensorSpec("in2", listOf(1), "FP32")),
            listOf(TensorSpec("out2", listOf(1), "FP32"))
        )
        val node3 = GraphNode(
            "node3",
            TestOperation("op3", "type3"),
            listOf(TensorSpec("in3", listOf(1), "FP32")),
            listOf(TensorSpec("out3", listOf(1), "FP32"))
        )

        graph.addNode(node1)
        graph.addNode(node2)
        graph.addNode(node3)

        graph.addEdge(GraphEdge("edge1", node1, node2, 0, 0, node1.outputs.first()))
        graph.addEdge(GraphEdge("edge2", node2, node3, 0, 0, node2.outputs.first()))

        val dotSubset = drawDot(graph, listOf(node3))
        assertNotNull(dotSubset)
        assertTrue(dotSubset.content.isNotEmpty())

        // All three nodes should appear when tracing backward from node3
        assertTrue(dotSubset.content.contains("node1"))
        assertTrue(dotSubset.content.contains("node2"))
        assertTrue(dotSubset.content.contains("node3"))
    }

    @Test
    fun testDifferentRankDirections() {
        val graph = DefaultComputeGraph()
        val node = GraphNode(
            id = "test",
            operation = TestOperation("test", "test"),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("out", listOf(1), "FP32"))
        )
        graph.addNode(node)

        val dotLR = drawDot(graph, "LR")
        assertTrue(dotLR.content.contains("rankdir=LR"))

        val dotTB = drawDot(graph, "TB")
        assertTrue(dotTB.content.contains("rankdir=TB"))
    }
}
