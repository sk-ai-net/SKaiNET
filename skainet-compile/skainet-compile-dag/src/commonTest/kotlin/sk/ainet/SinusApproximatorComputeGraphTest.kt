package sk.ainet.lang.nn.mlp

import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Unit test for converting SinusApproximator model into a compute graph
 */
class SinusApproximatorComputeGraphTest {

    @Test
    fun testSinusApproximatorToComputeGraph() {
        println("[DEBUG_LOG] Starting SinusApproximator to ComputeGraph conversion test")
        
        // Create the SinusApproximator model
        val sinusModel = SinusApproximator()

        print(sinusModel.model<FP32,Float>().describe(Shape(1, 1), FP32::class))

        
        // Create a compute graph to represent the model
        val graph = DefaultComputeGraph()
        
        // Create operations for neural network layers
        val inputOp = createInputOperation()
        val dense1Op = createDenseOperation("dense1", 1, 16)
        val relu1Op = createActivationOperation("relu1")
        val dense2Op = createDenseOperation("dense2", 16, 16) 
        val relu2Op = createActivationOperation("relu2")
        val outputOp = createDenseOperation("output", 16, 1)
        
        // Create nodes for each layer
        val inputNode = GraphNode(
            id = "input",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(1, 1), "FP32"))
        )
        
        val dense1Node = GraphNode(
            id = "dense1",
            operation = dense1Op,
            inputs = listOf(TensorSpec("dense1_in", listOf(1, 1), "FP32")),
            outputs = listOf(TensorSpec("dense1_out", listOf(1, 16), "FP32"))
        )
        
        val relu1Node = GraphNode(
            id = "relu1", 
            operation = relu1Op,
            inputs = listOf(TensorSpec("relu1_in", listOf(1, 16), "FP32")),
            outputs = listOf(TensorSpec("relu1_out", listOf(1, 16), "FP32"))
        )
        
        val dense2Node = GraphNode(
            id = "dense2",
            operation = dense2Op,
            inputs = listOf(TensorSpec("dense2_in", listOf(1, 16), "FP32")),
            outputs = listOf(TensorSpec("dense2_out", listOf(1, 16), "FP32"))
        )
        
        val relu2Node = GraphNode(
            id = "relu2",
            operation = relu2Op,
            inputs = listOf(TensorSpec("relu2_in", listOf(1, 16), "FP32")),
            outputs = listOf(TensorSpec("relu2_out", listOf(1, 16), "FP32"))
        )
        
        val outputNode = GraphNode(
            id = "output",
            operation = outputOp,
            inputs = listOf(TensorSpec("output_in", listOf(1, 16), "FP32")),
            outputs = listOf(TensorSpec("output_out", listOf(1, 1), "FP32"))
        )
        
        // Add all nodes to graph
        graph.addNode(inputNode)
        graph.addNode(dense1Node)
        graph.addNode(relu1Node)
        graph.addNode(dense2Node)
        graph.addNode(relu2Node)
        graph.addNode(outputNode)
        
        println("[DEBUG_LOG] Added ${graph.nodes.size} nodes to compute graph")
        
        // Create edges to connect the layers
        val edges = listOf(
            GraphEdge("edge1", inputNode, dense1Node, 0, 0, inputNode.outputs.first()),
            GraphEdge("edge2", dense1Node, relu1Node, 0, 0, dense1Node.outputs.first()),
            GraphEdge("edge3", relu1Node, dense2Node, 0, 0, relu1Node.outputs.first()),
            GraphEdge("edge4", dense2Node, relu2Node, 0, 0, dense2Node.outputs.first()),
            GraphEdge("edge5", relu2Node, outputNode, 0, 0, relu2Node.outputs.first())
        )
        
        // Add all edges to graph
        edges.forEach { graph.addEdge(it) }
        
        println("[DEBUG_LOG] Added ${graph.edges.size} edges to compute graph")
        
        // Validate the graph structure
        val validation = graph.validate()
        assertTrue(validation is ValidationResult.Valid, "Compute graph should be valid")
        println("[DEBUG_LOG] Graph validation: ${validation}")
        
        // Test graph properties
        assertEquals(6, graph.nodes.size, "Should have 6 nodes (input + 3 dense + 2 relu)")
        assertEquals(5, graph.edges.size, "Should have 5 edges connecting the layers")
        
        // Test input/output nodes
        val inputNodes = graph.getInputNodes()
        val outputNodes = graph.getOutputNodes()
        
        assertEquals(1, inputNodes.size, "Should have exactly one input node")
        assertEquals(1, outputNodes.size, "Should have exactly one output node")
        assertEquals("input", inputNodes.first().id, "Input node should be the input layer")
        assertEquals("output", outputNodes.first().id, "Output node should be the output layer")
        
        println("[DEBUG_LOG] Input nodes: ${inputNodes.map { it.id }}")
        println("[DEBUG_LOG] Output nodes: ${outputNodes.map { it.id }}")
        
        // Test topological ordering
        val topologicalOrder = graph.getTopologicalOrder()
        assertEquals(6, topologicalOrder.size, "Topological order should contain all nodes")
        
        // Verify the order makes sense (input first, output last)
        assertEquals("input", topologicalOrder.first().id, "Input should be first in topological order")
        assertEquals("output", topologicalOrder.last().id, "Output should be last in topological order")
        
        println("[DEBUG_LOG] Topological order: ${topologicalOrder.map { it.id }}")
        
        // Test model description
        val modelCard = sinusModel.modelCard()
        assertNotNull(modelCard, "Model card should not be null")
        assertTrue(modelCard.isNotEmpty(), "Model card should not be empty")
        
        println("[DEBUG_LOG] Model card length: ${modelCard.length}")
        println("[DEBUG_LOG] SinusApproximator to ComputeGraph conversion test completed successfully")
    }
    
    // Helper functions to create operations
    private fun createInputOperation(): Operation {
        return TestNeuralNetworkOperation("input", "input", mapOf("shape" to listOf(1, 1)))
    }
    
    private fun createDenseOperation(name: String, inputSize: Int, outputSize: Int): Operation {
        return TestNeuralNetworkOperation(
            name, 
            "dense", 
            mapOf(
                "input_size" to inputSize,
                "output_size" to outputSize,
                "has_bias" to true
            )
        )
    }
    
    private fun createActivationOperation(name: String): Operation {
        return TestNeuralNetworkOperation(name, "activation", mapOf("type" to "relu"))
    }
}

/**
 * Test implementation of Operation for neural network layers
 */
class TestNeuralNetworkOperation(
    override val name: String,
    override val type: String,
    override val parameters: Map<String, Any> = emptyMap()
) : Operation {
    
    override fun <T : sk.ainet.lang.types.DType, V> execute(
        inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>
    ): List<sk.ainet.lang.tensor.Tensor<T, V>> {
        // Mock implementation - just return the inputs
        return inputs
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        // Simple inference based on operation type
        return when (type) {
            "input" -> inputs
            "dense" -> {
                val outputSize = parameters["output_size"] as? Int ?: 1
                inputs.map { it.copy(shape = listOf(1, outputSize)) }
            }
            "activation" -> inputs
            else -> inputs
        }
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation {
        return TestNeuralNetworkOperation(name, type, newParameters)
    }
    
    override fun serialize(): Map<String, Any> {
        return mapOf(
            "name" to name,
            "type" to type,
            "parameters" to parameters
        )
    }
}