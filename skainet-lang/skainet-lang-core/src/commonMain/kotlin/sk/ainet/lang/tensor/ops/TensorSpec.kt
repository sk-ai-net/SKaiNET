package sk.ainet.lang.tensor.ops

/**
 * Specification for a tensor in the graph, containing shape and type information
 */
public data class TensorSpec(
    /**
     * Name/identifier of this tensor
     */
    public val name: String,

    /**
     * Shape of the tensor (null for dynamic shapes)
     */
    public val shape: List<Int>?,

    /**
     * Data type of the tensor
     */
    public val dtype: String,

    /**
     * Whether this tensor requires gradients
     */
    public val requiresGrad: Boolean = false,

    /**
     * Additional metadata for this tensor
     */
    public val metadata: Map<String, Any> = emptyMap()
)