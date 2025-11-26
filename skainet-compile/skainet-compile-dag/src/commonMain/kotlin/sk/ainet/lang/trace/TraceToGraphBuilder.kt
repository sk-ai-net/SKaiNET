package sk.ainet.lang.trace

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Shared builder to convert OpTrace streams into a ComputeGraph.
 * Used by both GraphSink (online) and DefaultExecutionTape.toComputeGraph() (offline).
 *
 * Deterministic ID policy (FR7):
 * - Node IDs: sequential per builder instance using insertion order, formatted as: "n<seq>_<opType>"
 *   Example: n0_Add, n1_Relu. This ensures stability for a given trace ordering.
 * - Edge IDs: derived from endpoints and port indices, formatted as:
 *   "e_<srcNodeId>_<srcOut>__<dstNodeId>_<dstIn>"
 *   Example: e_n0_Add_0__n1_Relu_0. This is deterministic given the node IDs and wiring.
 *
 * As a result, building the graph online (GraphSink) or offline (tape->graph) produces stable,
 * reproducible identifiers for nodes and edges provided the trace sequence is the same.
 */
public class TraceToGraphBuilder(private val graph: ComputeGraph) {

    private var nextNodeId = 0L

    private data class Producer(val node: GraphNode, val outIndex: Int, val spec: TensorSpec)
    private val producersByTensorId = mutableMapOf<String, Producer>()

    /**
     * Add a single OpTrace into the graph, wiring known producers to inputs
     * and registering the outputs as new producers.
     */
    public fun addTrace(trace: OpTrace) {
        val op = TraceBackedOperation(trace.opType, parameters = trace.attributes.filterValues { it != null } as Map<String, Any>)

        val inputSpecs = buildInputSpecs(trace)
        val outputSpecs = buildOutputSpecs(trace)

        val nodeId = "n${nextNodeId++}_${trace.opType}"
        val node = GraphNode(
            id = nodeId,
            operation = op,
            inputs = inputSpecs,
            outputs = outputSpecs
        )
        graph.addNode(node)

        // Wire edges from producers
        trace.inputs.forEachIndexed { idx, tRef ->
            val prod = producersByTensorId[tRef.id]
            if (prod != null) {
                val edgeId = "e_${prod.node.id}_${prod.outIndex}__${node.id}_$idx"
                val tensorSpec = inputSpecs.getOrNull(idx) ?: prod.spec
                graph.addEdge(
                    GraphEdge(
                        id = edgeId,
                        source = prod.node,
                        destination = node,
                        sourceOutputIndex = prod.outIndex,
                        destinationInputIndex = idx,
                        tensorSpec = tensorSpec
                    )
                )
            }
        }

        // Register output producers
        trace.outputs.forEachIndexed { outIdx, tRef ->
            val spec = outputSpecs.getOrNull(outIdx) ?: TensorSpec(
                name = tRef.id,
                shape = null,
                dtype = "unknown",
            )
            producersByTensorId[tRef.id] = Producer(node, outIdx, spec)
        }
    }

    public fun addAll(traces: Iterable<OpTrace>) {
        traces.forEach { addTrace(it) }
    }

    private fun buildInputSpecs(trace: OpTrace): List<TensorSpec> {
        val shapes = (trace.attributes["inputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["inputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = trace.inputs.size
        return List(count) { i ->
            val name = trace.inputs[i].id
            val shape = shapes?.getOrNull(i)
            val dtype = dtypes?.getOrNull(i) ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
        }
    }

    private fun buildOutputSpecs(trace: OpTrace): List<TensorSpec> {
        val shapes = (trace.attributes["outputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["outputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = trace.outputs.size
        return List(count) { i ->
            val name = trace.outputs[i].id
            val shape = shapes?.getOrNull(i)
            val dtype = dtypes?.getOrNull(i) ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
        }
    }

    /** Minimal Operation to host trace metadata for GraphNode. */
    private class TraceBackedOperation(
        override val name: String,
        override val type: String = "trace",
        override val parameters: Map<String, Any>
    ) : Operation {
        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = TraceBackedOperation(name, type, newParameters)
        @Suppress("UNCHECKED_CAST")
        override fun serialize(): Map<String, Any> = mapOf(
            "name" to name,
            "type" to type,
            "parameters" to parameters
        )
        override fun getDescription(): String = "$name($parameters)"
    }
}
