package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Unit tests for GraphTensorOps that verify graph creation during tensor operations.
 * 
 * This test suite covers the requirement from the issue description to create 
 * a simple unit test that checks graph creation by performing tensor operations.
 */
class GraphTensorOpsTest {
    
    private val dataFactory = DenseTensorDataFactory()
    
    /**
     * Helper function to create test tensors filled with ones
     */
    private fun createOnesTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.ones<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }
    
    /**
     * Test graph creation by performing simple tensor operations.
     * This test simulates the pattern described in the issue:
     * - Create tensors with ones
     * - Perform addition operation within execution context
     * - Verify that graph nodes are created correctly
     */
    @Test
    fun testGraphCreationWithSimpleTensorOps() {
        // Create execution context and graph
        val executionContext = DefaultExecutionContext()
        val graph = DefaultComputeGraph()
        
        // Create base tensor ops (VoidTensorOps for actual computation)
        val baseOps = VoidTensorOps<Float>()
        
        // Create graph-aware tensor ops that will record operations
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
        // Switch to graph mode to enable recording
        executionContext.switchToGraph()
        
        // Simulate the tensor creation pattern from the issue description
        // val a = tensor<FP32,Float> { Shape1(1) { ones() } }
        val a = createOnesTensor(Shape(1))
        
        // val b = tensor<FP32,Float> { Shape1(1) { ones() } }  
        val b = createOnesTensor(Shape(1))
        
        // Verify initial graph state
        assertEquals(0, graph.nodes.size, "Graph should be empty initially")
        
        // Start recording operations
        executionContext.startRecording()
        assertTrue(executionContext.isRecording, "Execution context should be recording")
        
        // Perform the operation: val graph = exec<FP32,Float> { a + b }
        val result = graphOps.add(a, b)
        
        // Verify the result tensor properties
        assertNotNull(result, "Result tensor should not be null")
        assertEquals(Shape(1), result.shape, "Result should have shape (1)")
        assertEquals(FP32::class, result.dtype, "Result should have FP32 dtype")
        
        // Verify that graph node was created
        assertEquals(1, graph.nodes.size, "Graph should contain one operation node")
        
        val addNode = graph.nodes.first()
        assertNotNull(addNode, "Add operation node should exist")
        assertTrue(addNode.id.startsWith("add_"), "Node ID should start with 'add_'")
        assertEquals(2, addNode.inputs.size, "Add operation should have 2 inputs")
        assertEquals(1, addNode.outputs.size, "Add operation should have 1 output")
        
        // Verify input tensor specs
        val input0 = addNode.inputs[0]
        val input1 = addNode.inputs[1]
        assertEquals("input_0", input0.name)
        assertEquals("input_1", input1.name)
        assertEquals(listOf(1), input0.shape)
        assertEquals(listOf(1), input1.shape)
        assertEquals("FP32", input0.dtype)
        assertEquals("FP32", input1.dtype)
        
        // Verify output tensor spec
        val output = addNode.outputs[0]
        assertEquals("output_0", output.name)
        assertEquals(listOf(1), output.shape)
        assertEquals("FP32", output.dtype)
        
        // Stop recording
        val tape = executionContext.stopRecording()
        assertNotNull(tape, "Recording should produce a tape")
    }
    
    /**
     * Test multiple operations to verify graph building with multiple nodes
     */
    @Test
    fun testGraphCreationWithMultipleOps() {
        val executionContext = DefaultExecutionContext()
        val graph = DefaultComputeGraph()
        val baseOps = VoidTensorOps<Float>()
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
        executionContext.switchToGraph()
        executionContext.startRecording()
        
        // Create tensors
        val a = createOnesTensor(Shape(1))
        val b = createOnesTensor(Shape(1))
        val c = createOnesTensor(Shape(1))
        
        // Perform multiple operations: (a + b) - c
        val intermediate = graphOps.add(a, b)
        val result = graphOps.subtract(intermediate, c)
        
        // Verify result
        assertNotNull(result)
        assertEquals(Shape(1), result.shape)
        assertEquals(FP32::class, result.dtype)
        
        // Verify graph contains two operations
        assertEquals(2, graph.nodes.size, "Graph should contain two operation nodes")
        
        val nodes = graph.nodes.toList()
        val addNode = nodes.find { it.id.startsWith("add_") }
        val subtractNode = nodes.find { it.id.startsWith("subtract_") }
        
        assertNotNull(addNode, "Add node should exist")
        assertNotNull(subtractNode, "Subtract node should exist")
        
        executionContext.stopRecording()
    }
    
    /**
     * Test that operations in eager mode don't create graph nodes
     */
    @Test
    fun testEagerModeDoesNotCreateGraphNodes() {
        val executionContext = DefaultExecutionContext()
        val graph = DefaultComputeGraph()
        val baseOps = VoidTensorOps<Float>()
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
        // Stay in eager mode (default)
        assertEquals(ExecutionMode.EAGER, executionContext.executionMode)
        
        val a = createOnesTensor(Shape(1))
        val b = createOnesTensor(Shape(1))
        
        // Perform operation in eager mode
        val result = graphOps.add(a, b)
        
        // Verify result is computed correctly
        assertNotNull(result)
        assertEquals(Shape(1), result.shape)
        assertEquals(FP32::class, result.dtype)
        
        // Verify no graph nodes were created
        assertEquals(0, graph.nodes.size, "Graph should remain empty in eager mode")
    }
}