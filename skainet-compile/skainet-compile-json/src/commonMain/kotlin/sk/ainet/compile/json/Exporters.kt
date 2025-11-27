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

        val attrs = node.toJsonAttrs()

        SkJsonNode(
            id = node.jsonNodeId(),
            label = node.operationName,
            namespace = "", // no namespaces yet for simple graphs
            subgraphIds = emptyList(),
            attrs = attrs,
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
    val s = this.shape
    val shapeStr = if (s == null) "[]" else s.joinToString(prefix = "[", postfix = "]")
    return "${this.dtype}${shapeStr}"
}

// --- Attrs mapping helpers ---

private fun GraphNode.toJsonAttrs(): List<SkJsonAttr> {
    // Deterministic ordered attribute assembly based on op name
    val p = operation.parameters
    val opName = operationName.lowercase()
    val (knownAttrs, knownKeys) = when (opName) {
        "conv2d" -> {
            val attrs = mutableListOf<SkJsonAttr>()
            // kernel_shape from weight spec if available (last two dims)
            kernelShapeFromWeight()?.let { ks ->
                attrs += SkJsonAttr("kernel_shape", ks)
            }
            p["stride"]?.let { attrs += SkJsonAttr("strides", toAttrString(it)) }
            p["padding"]?.let { attrs += SkJsonAttr("pads", toAttrString(it)) }
            p["dilation"]?.let { attrs += SkJsonAttr("dilations", toAttrString(it)) }
            p["groups"]?.let { attrs += SkJsonAttr("group", it.toString()) }
            attrs to setOf("stride", "padding", "dilation", "groups")
        }
        "maxpool2d", "avgpool2d" -> {
            val attrs = mutableListOf<SkJsonAttr>()
            p["kernelSize"]?.let { attrs += SkJsonAttr("kernel_shape", toAttrString(it)) }
            p["stride"]?.let { attrs += SkJsonAttr("strides", toAttrString(it)) }
            p["padding"]?.let { attrs += SkJsonAttr("pads", toAttrString(it)) }
            attrs to setOf("kernelSize", "stride", "padding")
        }
        "reshape" -> {
            val attrs = listOfNotNull(p["newShape"]?.let { SkJsonAttr("shape", toAttrString(it)) })
            attrs to setOf("newShape")
        }
        "flatten" -> {
            val attrs = mutableListOf<SkJsonAttr>()
            p["startDim"]?.let { attrs += SkJsonAttr("start_dim", it.toString()) }
            p["endDim"]?.let { attrs += SkJsonAttr("end_dim", it.toString()) }
            attrs to setOf("startDim", "endDim")
        }
        "softmax" -> {
            val attrs = listOfNotNull(p["dim"]?.let { SkJsonAttr("axis", it.toString()) })
            attrs to setOf("dim")
        }
        "squeeze" -> {
            // Map dim -> axes (ONNX style); -1 means "all singleton dims"
            val attrs = listOfNotNull(p["dim"]?.let { SkJsonAttr("axes", it.toString()) })
            attrs to setOf("dim")
        }
        "unsqueeze" -> {
            val attrs = listOfNotNull(p["dim"]?.let { SkJsonAttr("axes", it.toString()) })
            attrs to setOf("dim")
        }
        else -> emptyList<SkJsonAttr>() to emptySet()
    }

    // Append any unmapped parameters as __unmapped_* to preserve information
    val extras = p
        .filterKeys { it !in knownKeys }
        .entries
        .sortedBy { it.key }
        .map { (k, v) -> SkJsonAttr("__unmapped_" + k, toAttrString(v)) }

    return knownAttrs + extras
}

private fun GraphNode.kernelShapeFromWeight(): String? {
    // Prefer semantic name 'weight' when available
    val byName = inputs.firstOrNull { it.name == "weight" }?.shape
    val shape = byName ?: run {
        // Fallback: conv2d signature is (input, weight, [bias]). If semantic names are missing
        // (e.g., when built from OpTrace without port names), infer from position 1.
        inputs.getOrNull(1)?.shape
    }
    if (shape == null || shape.size < 4) return null
    val kH = shape[shape.size - 2]
    val kW = shape[shape.size - 1]
    return "($kH, $kW)"
}

private fun toAttrString(value: Any): String = when (value) {
    is Pair<*, *> -> "(${value.first}, ${value.second})"
    is IntArray -> value.joinToString(prefix = "[", postfix = "]")
    is LongArray -> value.joinToString(prefix = "[", postfix = "]")
    is Array<*> -> value.joinToString(prefix = "[", postfix = "]") { it.toString() }
    is List<*> -> value.joinToString(prefix = "[", postfix = "]") { it.toString() }
    else -> value.toString()
}
