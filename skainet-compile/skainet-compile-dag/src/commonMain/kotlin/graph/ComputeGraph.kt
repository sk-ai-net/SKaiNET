package sk.ainet.lang.graph

/**
 * Interface for managing a computational graph with nodes and edges.
 * This represents the core data structure for graph-based model execution.
 */
public interface ComputeGraph {
    
    /**
     * All nodes in the graph
     */
    public val nodes: List<GraphNode>
    
    /**
     * All edges in the graph
     */
    public val edges: List<GraphEdge>
    
    /**
     * Adds a new node to the graph
     */
    public fun addNode(node: GraphNode): GraphNode
    
    /**
     * Adds a new edge connecting two nodes
     */
    public fun addEdge(edge: GraphEdge): GraphEdge
    
    /**
     * Removes a node and all connected edges from the graph
     */
    public fun removeNode(node: GraphNode): Boolean
    
    /**
     * Removes an edge from the graph
     */
    public fun removeEdge(edge: GraphEdge): Boolean
    
    /**
     * Gets all input nodes (nodes with no incoming edges)
     */
    public fun getInputNodes(): List<GraphNode>
    
    /**
     * Gets all output nodes (nodes with no outgoing edges)
     */
    public fun getOutputNodes(): List<GraphNode>
    
    /**
     * Gets nodes that are inputs to the given node
     */
    public fun getInputNodes(node: GraphNode): List<GraphNode>
    
    /**
     * Gets nodes that receive output from the given node
     */
    public fun getOutputNodes(node: GraphNode): List<GraphNode>
    
    /**
     * Returns nodes in topological order for execution
     */
    public fun getTopologicalOrder(): List<GraphNode>
    
    /**
     * Validates the graph structure (checks for cycles, etc.)
     */
    public fun validate(): ValidationResult
    
    /**
     * Creates a copy of this graph
     */
    public fun copy(): ComputeGraph
    
    /**
     * Clears all nodes and edges
     */
    public fun clear()
}

