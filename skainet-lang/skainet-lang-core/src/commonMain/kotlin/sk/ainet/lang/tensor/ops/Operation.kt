package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Interface for operations that can be represented as graph nodes.
 * Each operation defines how to execute a computation and its metadata.
 */
public interface Operation {
    
    /**
     * Unique name of this operation type
     */
    public val name: String
    
    /**
     * Type/category of this operation (e.g., "math", "nn", "shape")
     */
    public val type: String
    
    /**
     * Parameters for this operation
     */
    public val parameters: Map<String, Any>
    
    /**
     * Execute this operation with the given inputs
     */
    public fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>>
    
    /**
     * Validate that the given inputs are compatible with this operation
     */
    public fun validateInputs(inputs: List<TensorSpec>): ValidationResult

    /**
     * Infer the output tensor specifications from input specifications
     */
    public fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec>
    
    /**
     * Clone this operation with potentially different parameters
     */
    public fun clone(newParameters: Map<String, Any> = parameters): Operation
    
    /**
     * Serialize this operation to a map
     */
    public fun serialize(): Map<String, Any>
    
    /**
     * Get a human-readable description of this operation
     */
    public fun getDescription(): String = "$name($parameters)"
}

/**
 * Result of graph validation
 */
public sealed class ValidationResult {
    public object Valid : ValidationResult()
    public data class Invalid(val errors: List<String>) : ValidationResult()
}

/**
 * Abstract base class for operations that provides common functionality
 */
public abstract class BaseOperation(
    override val name: String,
    override val type: String,
    override val parameters: Map<String, Any> = emptyMap()
) : Operation {
    
    override fun serialize(): Map<String, Any> = mapOf(
        "name" to name,
        "type" to type,
        "parameters" to parameters
    )
    
    override fun toString(): String = getDescription()
    
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Operation) return false
        return name == other.name && 
               type == other.type && 
               parameters == other.parameters
    }
    
    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + type.hashCode()
        result = 31 * result + parameters.hashCode()
        return result
    }
}