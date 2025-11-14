package sk.ainet.lang.nn.topology

/**
 * Traversal utilities for ModuleNode hierarchies.
 */

// Depth-first traversal applying [visit] to each node, including root.
public fun ModuleNode.walkDepthFirst(visit: (ModuleNode) -> Unit) {
    fun dfs(n: ModuleNode) {
        visit(n)
        n.children.forEach { dfs(it) }
    }
    dfs(this)
}

// Collect all parameters from the subtree rooted at this node.
public fun ModuleNode.collectParams(): List<ModuleParameter<*, *>> {
    val acc = mutableListOf<ModuleParameter<*, *>>()
    walkDepthFirst { node -> acc += node.params }
    return acc
}

// Find first node by id using DFS.
public fun ModuleNode.findById(id: String): ModuleNode? {
    var found: ModuleNode? = null
    walkDepthFirst { node -> if (found == null && node.id == id) found = node }
    return found
}

// Find first node by exact path match.
public fun ModuleNode.findByPath(path: String): ModuleNode? {
    var found: ModuleNode? = null
    walkDepthFirst { node -> if (found == null && node.path == path) found = node }
    return found
}

// Annotate nodes with hierarchical paths for nicer logs.
// Example: bindPaths(root, "model") -> root.path = "model",
// children paths become "model/childName".
public fun bindPaths(root: ModuleNode, base: String = root.name, separator: String = "/") {
    fun bind(node: ModuleNode, current: String) {
        node.path = current
        node.children.forEach { child ->
            val childSegment = child.name.ifEmpty { child.id }
            bind(child, if (current.isEmpty()) childSegment else "$current$separator$childSegment")
        }
    }
    bind(root, base)
}
