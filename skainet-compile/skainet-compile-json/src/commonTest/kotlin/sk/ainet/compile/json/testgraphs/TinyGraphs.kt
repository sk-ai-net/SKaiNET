package sk.ainet.compile.json.testgraphs

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Tiny synthetic graph builders for JSON export tests.
 * These utilities construct minimal ComputeGraph instances without relying on runtime tracing.
 */
public object TinyGraphs {

    /**
     * Build a tiny graph of the form:
     *   input(1x4 FP32) -> add(+, bias) -> relu -> output(1x4 FP32)
     * where `bias` is provided as a second input node for simplicity.
     */
    public fun tinyAddReluGraph(): ComputeGraph {
        val graph = SimpleTestComputeGraph()

        // Input specs
        val inSpec = TensorSpec(name = "input_out", shape = listOf(1, 4), dtype = "FP32")
        val biasSpec = TensorSpec(name = "bias_out", shape = listOf(1, 4), dtype = "FP32")
        val addOutSpec = TensorSpec(name = "add_out", shape = listOf(1, 4), dtype = "FP32")
        val reluOutSpec = TensorSpec(name = "relu_out", shape = listOf(1, 4), dtype = "FP32")

        // Nodes
        val inputNode = GraphNode(
            id = "input",
            operation = InputOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(inSpec)
        )
        val biasNode = GraphNode(
            id = "bias",
            operation = InputOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(biasSpec)
        )
        val addNode = GraphNode(
            id = "add",
            operation = AddOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = listOf(inSpec, biasSpec),
            outputs = listOf(addOutSpec)
        )
        val reluNode = GraphNode(
            id = "relu",
            operation = ReluOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = listOf(addOutSpec),
            outputs = listOf(reluOutSpec)
        )

        // Register nodes
        graph.addNode(inputNode)
        graph.addNode(biasNode)
        graph.addNode(addNode)
        graph.addNode(reluNode)

        // Edges
        graph.addEdge(
            GraphEdge(
                id = "e_input_add",
                source = inputNode,
                destination = addNode,
                sourceOutputIndex = 0,
                destinationInputIndex = 0,
                tensorSpec = inSpec
            )
        )
        graph.addEdge(
            GraphEdge(
                id = "e_bias_add",
                source = biasNode,
                destination = addNode,
                sourceOutputIndex = 0,
                destinationInputIndex = 1,
                tensorSpec = biasSpec
            )
        )
        graph.addEdge(
            GraphEdge(
                id = "e_add_relu",
                source = addNode,
                destination = reluNode,
                sourceOutputIndex = 0,
                destinationInputIndex = 0,
                tensorSpec = addOutSpec
            )
        )

        return graph
    }
}

/**
 * Minimal in-test implementation of ComputeGraph sufficient for JSON export mapping.
 */
private class SimpleTestComputeGraph : ComputeGraph {
    private val _nodes = mutableListOf<GraphNode>()
    private val _edges = mutableListOf<GraphEdge>()

    override val nodes: List<GraphNode> get() = _nodes
    override val edges: List<GraphEdge> get() = _edges

    override fun addNode(node: GraphNode): GraphNode {
        require(_nodes.none { it.id == node.id }) { "Duplicate node id: ${node.id}" }
        _nodes.add(node)
        return node
    }

    override fun addEdge(edge: GraphEdge): GraphEdge {
        _edges.add(edge)
        return edge
    }

    override fun removeNode(node: GraphNode): Boolean {
        val removed = _nodes.remove(node)
        if (removed) {
            _edges.removeAll { it.source == node || it.destination == node }
        }
        return removed
    }

    override fun removeEdge(edge: GraphEdge): Boolean = _edges.remove(edge)

    override fun getInputNodes(): List<GraphNode> = _nodes.filter { n -> _edges.none { it.destination == n } }

    override fun getOutputNodes(): List<GraphNode> = _nodes.filter { n -> _edges.none { it.source == n } }

    override fun getInputNodes(node: GraphNode): List<GraphNode> = _edges.filter { it.destination == node }.map { it.source }

    override fun getOutputNodes(node: GraphNode): List<GraphNode> = _edges.filter { it.source == node }.map { it.destination }

    override fun getTopologicalOrder(): List<GraphNode> {
        // Simple Kahn's algorithm for small graphs
        val inDegree = _nodes.associateWith { n -> _edges.count { it.destination == n } }.toMutableMap()
        val queue = ArrayDeque(inDegree.filterValues { it == 0 }.keys)
        val order = mutableListOf<GraphNode>()
        while (queue.isNotEmpty()) {
            val n = queue.removeFirst()
            order.add(n)
            _edges.filter { it.source == n }.forEach { e ->
                val m = e.destination
                inDegree[m] = (inDegree[m] ?: 0) - 1
                if (inDegree[m] == 0) queue.add(m)
            }
        }
        return order
    }

    override fun validate(): ValidationResult {
        // For tests, perform a minimal cycle check using topological order length
        val order = getTopologicalOrder()
        return if (order.size == _nodes.size) ValidationResult.Valid
        else ValidationResult.Invalid(listOf("Graph contains cycles or disconnected issues"))
    }

    override fun copy(): ComputeGraph {
        val copy = SimpleTestComputeGraph()
        val nodeMap = mutableMapOf<GraphNode, GraphNode>()
        _nodes.forEach { n ->
            val nn = GraphNode(n.id, n.operation, n.inputs.toList(), n.outputs.toList(), n.metadata.toMap())
            nodeMap[n] = nn
            copy.addNode(nn)
        }
        _edges.forEach { e ->
            copy.addEdge(
                GraphEdge(
                    id = e.id,
                    source = nodeMap[e.source]!!,
                    destination = nodeMap[e.destination]!!,
                    sourceOutputIndex = e.sourceOutputIndex,
                    destinationInputIndex = e.destinationInputIndex,
                    tensorSpec = e.tensorSpec,
                    metadata = e.metadata
                )
            )
        }
        return copy
    }

    override fun clear() {
        _nodes.clear()
        _edges.clear()
    }
}
