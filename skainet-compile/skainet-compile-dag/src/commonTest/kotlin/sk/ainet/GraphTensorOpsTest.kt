package sk.ainet

import sk.ainet.context.ExecutionStats
import sk.ainet.context.MemoryInfo
import sk.ainet.context.Phase
import sk.ainet.context.GraphExecutionContext
import sk.ainet.lang.graph.GraphTensorOps
import sk.ainet.lang.graph.ExecutionTape
import sk.ainet.lang.graph.TapeStack
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.hooks.ForwardHooks
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
    private val baseOps: TensorOps = VoidTensorOps()

    /**
     * Helper function to create test tensors filled with ones
     */
    private fun createOnesTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.ones<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    /**
     * Minimal GraphExecutionContext implementation for tests.
     * Provides tape stack and recording controls without any backend.
     */
    private class TestGraphExecutionContext(
        override val ops: TensorOps = VoidTensorOps(),
        override val phase: Phase = Phase.EVAL,
        override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
        override val hooks: ForwardHooks? = null,
        override val memoryInfo: MemoryInfo = MemoryInfo.getEmptyInfo(),
        override val executionStats: ExecutionStats = ExecutionStats(),
    ) : GraphExecutionContext {
        private val _tapes = DefaultTapeStack()
        override val tapeStack: TapeStack get() = _tapes
        override val currentTape: ExecutionTape? get() = _tapes.currentTape

        fun startRecording() {
            val tape = DefaultExecutionTape()
            tape.startRecording()
            _tapes.pushTape(tape)
        }

        fun stopRecording(): ExecutionTape? {
            val tape = _tapes.popTape()
            tape?.stopRecording()
            return tape
        }

        override fun collectGarbage() { /* no-op */ }
        override fun resetExecutionStats() { /* no-op */ }
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
        val executionContext = TestGraphExecutionContext()
        val graph = DefaultComputeGraph()
        
        // Create graph-aware tensor ops that will record operations
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
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
        
        // Verify that graph nodes were created: 2 input nodes + 1 add op node
        assertEquals(3, graph.nodes.size, "Graph should contain two input nodes and one add operation node")
        
        val addNode = graph.nodes.first { it.id.startsWith("add_") }
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
        val executionContext = TestGraphExecutionContext()
        val graph = DefaultComputeGraph()
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
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
        
        // Verify graph contains two operation nodes (add and subtract)
        val opNodes = graph.nodes.filter { !it.operation.type.equals("input") }
        assertTrue(opNodes.size >= 2, "Graph should contain at least two operation nodes")
        
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
        val executionContext = TestGraphExecutionContext()
        val graph = DefaultComputeGraph()
        val graphOps = GraphTensorOps(baseOps, graph, executionContext)
        
        // Stay in eager mode (not recording) by default
        
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