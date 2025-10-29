package sk.ainet.compile.nn

import sk.ainet.lang.tensor.ops.Operation

/**
 * Registry for managing operation types and their implementations.
 * Provides centralized management of available operations and their metadata.
 */
public interface OperationRegistry {
    
    /**
     * All registered operation names
     */
    public val registeredOperations: Set<String>
    
    /**
     * Register an operation implementation
     */
    public fun registerOperation(name: String, factory: OperationFactory)
    
    /**
     * Unregister an operation
     */
    public fun unregisterOperation(name: String): Boolean
    
    /**
     * Check if an operation is registered
     */
    public fun isRegistered(name: String): Boolean
    
    /**
     * Create an operation instance by name
     */
    public fun createOperation(name: String, parameters: Map<String, Any> = emptyMap()): Operation
    
    /**
     * Get operation metadata
     */
    public fun getOperationMetadata(name: String): OperationMetadata?
    
    /**
     * Get all operation metadata
     */
    public fun getAllOperationMetadata(): Map<String, OperationMetadata>
    
    /**
     * Deserialize an operation from a map
     */
    public fun deserializeOperation(data: Map<String, Any>): Operation
}

/**
 * Factory interface for creating operations
 */
public interface OperationFactory {
    /**
     * Create an operation with the given parameters
     */
    public fun create(parameters: Map<String, Any>): Operation
    
    /**
     * Get metadata about this operation
     */
    public fun getMetadata(): OperationMetadata
}

/**
 * Metadata about an operation type
 */
public data class OperationMetadata(
    /**
     * Name of the operation
     */
    public val name: String,
    
    /**
     * Type/category of the operation
     */
    public val type: String,
    
    /**
     * Description of what this operation does
     */
    public val description: String,
    
    /**
     * Expected input specifications
     */
    public val inputSpecs: List<ParameterSpec>,
    
    /**
     * Expected output specifications
     */
    public val outputSpecs: List<ParameterSpec>,
    
    /**
     * Parameter specifications
     */
    public val parameterSpecs: List<ParameterSpec>,
    
    /**
     * Whether this operation supports gradients
     */
    public val supportsGradients: Boolean = false,
    
    /**
     * Additional metadata
     */
    public val additionalMetadata: Map<String, Any> = emptyMap()
)

/**
 * Specification for a parameter (input, output, or configuration parameter)
 */
public data class ParameterSpec(
    /**
     * Name of the parameter
     */
    public val name: String,
    
    /**
     * Type of the parameter
     */
    public val type: String,
    
    /**
     * Whether this parameter is required
     */
    public val required: Boolean = true,
    
    /**
     * Default value (if any)
     */
    public val defaultValue: Any? = null,
    
    /**
     * Description of the parameter
     */
    public val description: String = "",
    
    /**
     * Additional constraints or metadata
     */
    public val constraints: Map<String, Any> = emptyMap()
)

/**
 * Default implementation of OperationRegistry
 */
public class DefaultOperationRegistry : OperationRegistry {
    
    private val factories = mutableMapOf<String, OperationFactory>()
    
    override val registeredOperations: Set<String> get() = factories.keys.toSet()
    
    override fun registerOperation(name: String, factory: OperationFactory) {
        factories[name] = factory
    }
    
    override fun unregisterOperation(name: String): Boolean {
        return factories.remove(name) != null
    }
    
    override fun isRegistered(name: String): Boolean {
        return factories.containsKey(name)
    }
    
    override fun createOperation(name: String, parameters: Map<String, Any>): Operation {
        val factory = factories[name] 
            ?: throw IllegalArgumentException("Operation '$name' is not registered")
        return factory.create(parameters)
    }
    
    override fun getOperationMetadata(name: String): OperationMetadata? {
        return factories[name]?.getMetadata()
    }
    
    override fun getAllOperationMetadata(): Map<String, OperationMetadata> {
        return factories.mapValues { it.value.getMetadata() }
    }
    
    override fun deserializeOperation(data: Map<String, Any>): Operation {
        val name = data["name"] as? String 
            ?: throw IllegalArgumentException("Missing 'name' field in operation data")
        val parameters = data["parameters"] as? Map<String, Any> ?: emptyMap()
        
        return createOperation(name, parameters)
    }
}

/**
 * Global operation registry
 */
public object GlobalOperationRegistry {
    private var _current: OperationRegistry = DefaultOperationRegistry()
    
    /**
     * Get the current global operation registry
     */
    public fun current(): OperationRegistry = _current
    
    /**
     * Set the global operation registry
     */
    public fun setCurrent(registry: OperationRegistry) {
        _current = registry
    }
    
    /**
     * Register an operation in the global registry
     */
    public fun registerOperation(name: String, factory: OperationFactory) {
        _current.registerOperation(name, factory)
    }
    
    /**
     * Create an operation using the global registry
     */
    public fun createOperation(name: String, parameters: Map<String, Any> = emptyMap()): Operation {
        return _current.createOperation(name, parameters)
    }
}