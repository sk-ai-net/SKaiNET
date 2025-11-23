package sk.ainet.compile.json

import sk.ainet.compile.json.model.*
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode

/**
 * Basic mapper from a simple internal ComputeGraph to the JSON export model.
 * This covers nodes, incoming edges, and minimal tensor metadata for ports.
 */
public fun exportGraphToJson(graph: ComputeGraph, label: String = "graph"): SkJsonExport {
    val nodesById = graph.nodes.associateBy { it.id }

    // Pre-group edges by destination for quick incoming lookup
    val edgesByDest: Map<String, List<GraphEdge>> = graph.edges.groupBy { it.destination.id }

    val jsonNodes = graph.nodes.map { node ->
        val incoming = edgesByDest[node.id].orEmpty().map { e ->
            SkJsonEdgeRef(
                sourceNodeId = e.source.id,
                sourceNodeOutputId = e.sourceOutputIndex.toString(),
                targetNodeInputId = e.destinationInputIndex.toString()
            )
        }

        SkJsonNode(
            id = node.jsonNodeId(),
            label = node.operationName,
            namespace = "", // no namespaces yet for simple graphs
            subgraphIds = emptyList(),
            attrs = emptyList(),
            incomingEdges = incoming,
            outputsMetadata = node.outputs.mapIndexed { idx, spec ->
                SkJsonPortMeta(
                    id = idx.toString(),
                    attrs = listOfNotNull(
                        SkJsonAttr("__tensor_tag", spec.name),
                        SkJsonAttr("tensor_shape", spec.jsonShape())
                    )
                )
            },
            inputsMetadata = node.inputs.mapIndexed { idx, spec ->
                SkJsonPortMeta(
                    id = idx.toString(),
                    attrs = listOfNotNull(
                        SkJsonAttr("__tensor_tag", spec.name),
                        SkJsonAttr("tensor_shape", spec.jsonShape())
                    )
                )
            },
            style = null,
            config = null
        )
    }

    val jsonGraph = SkJsonGraph(
        id = "main_graph",
        nodes = jsonNodes,
        attrs = emptyList()
    )

    return SkJsonExport(
        label = label,
        graphs = listOf(jsonGraph)
    )
}

private fun GraphNode.jsonNodeId(): String = this.id

private fun sk.ainet.lang.tensor.ops.TensorSpec.jsonShape(): String {
    val shapeStr = if (this.shape == null) "[]" else this.shape.joinToString(prefix = "[", postfix = "]")
    return "${this.dtype}${shapeStr}"
}
