package sk.ainet.lang.graph

/**
 * Represents an edge in the computational graph, connecting two nodes
 * and representing tensor flow between operations.
 */
public data class GraphEdge(
    /**
     * Unique identifier for this edge
     */
    public val id: String,
    
    /**
     * The source node (producer of the tensor)
     */
    public val source: GraphNode,
    
    /**
     * The destination node (consumer of the tensor)
     */
    public val destination: GraphNode,
    
    /**
     * Index of the output from the source node (for multi-output operations)
     */
    public val sourceOutputIndex: Int = 0,
    
    /**
     * Index of the input to the destination node (for multi-input operations)
     */
    public val destinationInputIndex: Int = 0,
    
    /**
     * Specification of the tensor flowing through this edge
     */
    public val tensorSpec: TensorSpec,
    
    /**
     * Additional metadata for this edge
     */
    public val metadata: Map<String, Any> = emptyMap()
) {
    
    /**
     * Gets the source node ID
     */
    public val sourceId: String get() = source.id
    
    /**
     * Gets the destination node ID
     */
    public val destinationId: String get() = destination.id
    
    /**
     * Gets the tensor name
     */
    public val tensorName: String get() = tensorSpec.name
    
    /**
     * Gets the tensor shape
     */
    public val tensorShape: List<Int>? get() = tensorSpec.shape
    
    /**
     * Gets the tensor data type
     */
    public val tensorDtype: String get() = tensorSpec.dtype
    
    override fun toString(): String {
        return "GraphEdge(id=$id, ${source.id}[$sourceOutputIndex] -> ${destination.id}[$destinationInputIndex], tensor=${tensorSpec.name})"
    }
}

/**
 * Lightweight version of GraphEdge that only stores node IDs
 * Useful for serialization and when nodes are stored separately
 */
public data class GraphEdgeRef(
    public val id: String,
    public val sourceId: String,
    public val destinationId: String,
    public val sourceOutputIndex: Int = 0,
    public val destinationInputIndex: Int = 0,
    public val tensorSpec: TensorSpec,
    public val metadata: Map<String, Any> = emptyMap()
)