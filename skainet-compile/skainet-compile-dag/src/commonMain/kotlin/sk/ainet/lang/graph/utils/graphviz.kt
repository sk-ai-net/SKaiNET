package sk.ainet.lang.graph.utils

import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.ComputeGraph

/**
 * Represents a Graphviz DOT graph output
 */
public data class DotGraph(val content: String)

/**
 * Traces the compute graph starting from output nodes to collect all nodes and dependencies
 */
public fun trace(graph: ComputeGraph): Pair<Set<GraphNode>, Set<Pair<GraphNode, GraphNode>>> {
    val nodes = graph.nodes.toSet()
    val edges = mutableSetOf<Pair<GraphNode, GraphNode>>()
    
    // Build edges based on graph dependencies
    for (node in nodes) {
        val inputNodes = graph.getInputNodes(node)
        for (inputNode in inputNodes) {
            edges.add(inputNode to node)
        }
    }
    
    return nodes to edges
}

/**
 * Generates a Graphviz DOT representation of the compute graph
 */
public fun drawDot(graph: ComputeGraph, rankdir: String = "LR"): DotGraph {
    require(rankdir in listOf("LR", "TB"))

    val (nodes, edges) = trace(graph)
    val dotContent = StringBuilder()

    dotContent.appendLine("digraph {")
    dotContent.appendLine("    rankdir=$rankdir;")

    // Add nodes
    for (node in nodes) {
        val nodeId = node.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        
        val labelContent = "${node.operationName} | ${node.id}"
        
        // Assign shape based on operation type
        val shapeAttributes = when (node.operation.type) {
            "input" -> "shape=record, style=filled, fillcolor=lightblue"
            "math" -> "shape=circle"
            else -> "shape=record" // values and other operations use rectangle
        }
        
        dotContent.appendLine("    $nodeId [label=\"$labelContent\", $shapeAttributes];")
        
        // Add operation node if operation has parameters
        if (node.operation.parameters.isNotEmpty()) {
            val opId = "${nodeId}_op"
            val paramStr = node.operation.parameters.entries.joinToString("\\n") { "${it.key}: ${it.value}" }
            dotContent.appendLine("    $opId [label=\"$paramStr\", shape=box, style=dashed];")
            dotContent.appendLine("    $opId -> $nodeId [style=dotted];")
        }
    }

    // Add edges
    for ((from, to) in edges) {
        val fromId = from.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        val toId = to.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        dotContent.appendLine("    $fromId -> $toId;")
    }

    dotContent.appendLine("}")

    return DotGraph(dotContent.toString())
}

/**
 * Generates a Graphviz DOT representation for a subset of nodes in the compute graph
 */
public fun drawDot(graph: ComputeGraph, outputNodes: List<GraphNode>, rankdir: String = "LR"): DotGraph {
    require(rankdir in listOf("LR", "TB"))

    // Trace from output nodes backward to collect relevant nodes
    val relevantNodes = mutableSetOf<GraphNode>()
    val edges = mutableSetOf<Pair<GraphNode, GraphNode>>()
    
    fun collectDependencies(node: GraphNode) {
        if (node !in relevantNodes) {
            relevantNodes.add(node)
            val inputNodes = graph.getInputNodes(node)
            for (inputNode in inputNodes) {
                edges.add(inputNode to node)
                collectDependencies(inputNode)
            }
        }
    }
    
    for (outputNode in outputNodes) {
        collectDependencies(outputNode)
    }

    val dotContent = StringBuilder()

    dotContent.appendLine("digraph {")
    dotContent.appendLine("    rankdir=$rankdir;")

    // Add nodes
    for (node in relevantNodes) {
        val nodeId = node.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        
        val labelContent = "${node.operationName} | ${node.id}"
        
        // Assign shape based on operation type
        val shapeAttributes = when (node.operation.type) {
            "input" -> "shape=record, style=filled, fillcolor=lightblue"
            "math" -> "shape=circle"
            else -> "shape=record" // values and other operations use rectangle
        }
        
        dotContent.appendLine("    $nodeId [label=\"$labelContent\", $shapeAttributes];")
        
        // Add operation node if operation has parameters
        if (node.operation.parameters.isNotEmpty()) {
            val opId = "${nodeId}_op"
            val paramStr = node.operation.parameters.entries.joinToString("\\n") { "${it.key}: ${it.value}" }
            dotContent.appendLine("    $opId [label=\"$paramStr\", shape=box, style=dashed];")
            dotContent.appendLine("    $opId -> $nodeId [style=dotted];")
        }
    }

    // Add edges
    for ((from, to) in edges) {
        val fromId = from.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        val toId = to.id.replace("[^a-zA-Z0-9_]".toRegex(), "_")
        dotContent.appendLine("    $fromId -> $toId;")
    }

    dotContent.appendLine("}")

    return DotGraph(dotContent.toString())
}