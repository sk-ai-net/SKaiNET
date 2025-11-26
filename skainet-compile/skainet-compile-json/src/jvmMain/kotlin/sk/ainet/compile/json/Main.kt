package sk.ainet.compile.json

import kotlinx.coroutines.runBlocking
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Minimal CLI entry point for exporting a tiny synthetic graph to JSON.
 * This is a proof of concept until dynamic model loading is implemented.
 *
 * Supported CLI options:
 *  --output=path/to/file.json (optional; if omitted, prints to stdout)
 *  --label=custom_label (optional; default: "tiny_graph")
 */
public fun main(args: Array<String>): Unit = runBlocking {
    val params = args.associate { arg ->
        val idx = arg.indexOf('=')
        if (idx > 0) arg.substring(0, idx) to arg.substring(idx + 1) else arg to ""
    }

    val outputPath = params["--output"]
    val label = params["--label"] ?: "tiny_graph"

    val graph = buildTinyGraph()
    val export = exportGraphToJson(graph, label)

    if (outputPath.isNullOrBlank()) {
        println(export.toJsonString(pretty = true))
    } else {
        writeExportToFile(export, outputPath, pretty = true)
        println("Exported JSON to $outputPath")
    }
}

// --- Tiny synthetic graph builder (self-contained for CLI PoC) ---
private fun buildTinyGraph(): ComputeGraph {
    val g = SimpleCliComputeGraph()

    val inSpec = TensorSpec(name = "input_out", shape = listOf(1, 4), dtype = "FP32")
    val biasSpec = TensorSpec(name = "bias_out", shape = listOf(1, 4), dtype = "FP32")
    val addOutSpec = TensorSpec(name = "add_out", shape = listOf(1, 4), dtype = "FP32")
    val reluOutSpec = TensorSpec(name = "relu_out", shape = listOf(1, 4), dtype = "FP32")

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

    g.addNode(inputNode)
    g.addNode(biasNode)
    g.addNode(addNode)
    g.addNode(reluNode)

    g.addEdge(
        GraphEdge(
            id = "e_input_add",
            source = inputNode,
            destination = addNode,
            sourceOutputIndex = 0,
            destinationInputIndex = 0,
            tensorSpec = inSpec
        )
    )
    g.addEdge(
        GraphEdge(
            id = "e_bias_add",
            source = biasNode,
            destination = addNode,
            sourceOutputIndex = 0,
            destinationInputIndex = 1,
            tensorSpec = biasSpec
        )
    )
    g.addEdge(
        GraphEdge(
            id = "e_add_relu",
            source = addNode,
            destination = reluNode,
            sourceOutputIndex = 0,
            destinationInputIndex = 0,
            tensorSpec = addOutSpec
        )
    )

    return g
}

// Minimal in-CLI implementation of ComputeGraph sufficient for JSON export mapping.
private class SimpleCliComputeGraph : ComputeGraph {
    private val _nodes = mutableListOf<GraphNode>()
    private val _edges = mutableListOf<GraphEdge>()

    override val nodes: List<GraphNode> get() = _nodes
    override val edges: List<GraphEdge> get() = _edges

    override fun addNode(node: GraphNode): GraphNode {
        require(_nodes.none { it.id == node.id }) { "Duplicate node id: ${'$'}{node.id}" }
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

    override fun validate(): sk.ainet.lang.tensor.ops.ValidationResult {
        val order = getTopologicalOrder()
        return if (order.size == _nodes.size) sk.ainet.lang.tensor.ops.ValidationResult.Valid
        else sk.ainet.lang.tensor.ops.ValidationResult.Invalid(listOf("Graph contains cycles or disconnected issues"))
    }

    override fun copy(): ComputeGraph = SimpleCliComputeGraph().also { /* not needed for CLI */ }

    override fun clear() {
        _nodes.clear(); _edges.clear()
    }
}
