package sk.ainet.lang.graph

/**
 * Represents a node in the computational graph containing operation metadata.
 * Each node represents a single operation with its inputs, outputs, and parameters.
 */
public data class GraphNode(
    /**
     * Unique identifier for this node
     */
    public val id: String,
    
    /**
     * The operation this node represents
     */
    public val operation: Operation,
    
    /**
     * Metadata about inputs to this operation
     */
    public val inputs: List<TensorSpec>,
    
    /**
     * Metadata about outputs from this operation
     */
    public val outputs: List<TensorSpec>,
    
    /**
     * Additional metadata for this node
     */
    public val metadata: Map<String, Any> = emptyMap()
) {
    
    /**
     * Gets the operation name
     */
    public val operationName: String get() = operation.name
    
    /**
     * Gets the operation type
     */
    public val operationType: String get() = operation.type
    
    override fun toString(): String {
        return "GraphNode(id=$id, operation=${operation.name}, inputs=${inputs.size}, outputs=${outputs.size})"
    }
}