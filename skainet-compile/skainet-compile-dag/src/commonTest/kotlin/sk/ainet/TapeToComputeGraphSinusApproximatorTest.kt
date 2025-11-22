package sk.ainet

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests the toComputeGraph() implementation on ExecutionTape using a SinusApproximator-like sequence.
 * We simulate the model execution by manually pushing RecordedOperation entries with proper TensorSpec.
 */
class TapeToComputeGraphSinusApproximatorTest {

    @Test
    fun testTapeToComputeGraphWithSinusApproximator() {
        println("[DEBUG_LOG] Starting tape->graph test (SinusApproximator)")

        val ctx = DirectCpuExecutionContext()
        val model = SinusApproximator()
        // Ensure model reflection still works (sanity)
        println(model.model(ctx).describe(Shape(1, 1), ctx, FP32::class))

        // Use a subclass to add operations directly by specs (no real tensors required)
        val tape = object : DefaultExecutionTape() {
            fun addBySpec(op: Operation, inputs: List<TensorSpec>, outputs: List<TensorSpec>) {
                _operations.add(
                    sk.ainet.lang.graph.RecordedOperation(
                        operation = op,
                        inputs = inputs,
                        outputs = outputs,
                        timestamp = operationCounter++
                    )
                )
            }
        }

        // Build a simple sequence: input -> dense1 -> relu1 -> dense2 -> relu2 -> output
        val inputOp = TestNeuralNetworkOperation("input", "input", mapOf("shape" to listOf(1, 1)))
        val dense1Op = TestNeuralNetworkOperation("dense1", "dense", mapOf("input_size" to 1,  "output_size" to 16))
        val relu1Op  = TestNeuralNetworkOperation("relu1",  "activation", mapOf("type" to "relu"))
        val dense2Op = TestNeuralNetworkOperation("dense2", "dense", mapOf("input_size" to 16, "output_size" to 16))
        val relu2Op  = TestNeuralNetworkOperation("relu2",  "activation", mapOf("type" to "relu"))
        val outOp    = TestNeuralNetworkOperation("output", "dense", mapOf("input_size" to 16, "output_size" to 1))

        // Helper to make specs
        fun spec(name: String, shape: List<Int>) = TensorSpec(name, shape, "FP32")

        tape.addBySpec(inputOp, inputs = emptyList(), outputs = listOf(spec("input_out", listOf(1, 1))))
        tape.addBySpec(dense1Op, inputs = listOf(spec("dense1_in", listOf(1, 1))), outputs = listOf(spec("dense1_out", listOf(1, 16))))
        tape.addBySpec(relu1Op,  inputs = listOf(spec("relu1_in",  listOf(1, 16))), outputs = listOf(spec("relu1_out",  listOf(1, 16))))
        tape.addBySpec(dense2Op, inputs = listOf(spec("dense2_in", listOf(1, 16))), outputs = listOf(spec("dense2_out", listOf(1, 16))))
        tape.addBySpec(relu2Op,  inputs = listOf(spec("relu2_in",  listOf(1, 16))), outputs = listOf(spec("relu2_out",  listOf(1, 16))))
        tape.addBySpec(outOp,    inputs = listOf(spec("output_in", listOf(1, 16))), outputs = listOf(spec("output_out", listOf(1, 1))))

        val graph = tape.toComputeGraph() as DefaultComputeGraph

        // Basic structure
        assertEquals(6, graph.nodes.size, "Graph should contain 6 nodes recorded on the tape")
        assertEquals(5, graph.edges.size, "Graph should contain 5 edges connecting sequential operations")

        // Validate graph
        val validation = graph.validate()
        assertTrue(validation is ValidationResult.Valid, "Generated compute graph must be valid: $validation")
        println("[DEBUG_LOG] Graph validation OK: $validation")

        // Check inputs/outputs
        val inputNodes = graph.getInputNodes()
        val outputNodes = graph.getOutputNodes()
        assertEquals(1, inputNodes.size, "Exactly one input node expected")
        assertEquals(1, outputNodes.size, "Exactly one output node expected")
        println("[DEBUG_LOG] Input nodes: ${inputNodes.map { it.id }} | Output nodes: ${outputNodes.map { it.id }}")

        // Topological order must include all nodes in sequence
        val topo = graph.getTopologicalOrder()
        assertEquals(6, topo.size, "Topological order should include all nodes")
        println("[DEBUG_LOG] Topological order: ${topo.map { it.id }}")
    }
}
