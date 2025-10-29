package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Interface for recording operations during execution for deferred/graph-based execution.
 * The tape captures operations as they are performed, allowing for later replay or optimization.
 */
public interface ExecutionTape {
    
    /**
     * Whether this tape is currently recording operations
     */
    public val isRecording: Boolean
    
    /**
     * All recorded operations on this tape
     */
    public val operations: List<RecordedOperation>
    
    /**
     * Start recording operations on this tape
     */
    public fun startRecording()
    
    /**
     * Stop recording operations on this tape
     */
    public fun stopRecording()
    
    /**
     * Record an operation on this tape
     */
    public fun <T : DType, V> recordOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    )
    
    /**
     * Replay all recorded operations
     */
    public fun <T : DType, V> replay(): List<Tensor<T, V>>
    
    /**
     * Clear all recorded operations
     */
    public fun clear()
    
    /**
     * Create a copy of this tape
     */
    public fun copy(): ExecutionTape
    
    /**
     * Optimize the recorded operations (e.g., operation fusion, dead code elimination)
     */
    public fun optimize(): ExecutionTape
    
    /**
     * Prune unnecessary operations from the tape
     */
    public fun prune(keepOutputs: Set<String> = emptySet()): ExecutionTape
    
    /**
     * Convert the tape to a compute graph
     */
    public fun toComputeGraph(): ComputeGraph
}

/**
 * Represents a recorded operation with its inputs and outputs
 */
public data class RecordedOperation(
    /**
     * The operation that was performed
     */
    public val operation: Operation,
    
    /**
     * Input tensor specifications
     */
    public val inputs: List<TensorSpec>,
    
    /**
     * Output tensor specifications
     */
    public val outputs: List<TensorSpec>,
    
    /**
     * Timestamp when the operation was recorded (sequential counter)
     */
    public val timestamp: Long = 0L,
    
    /**
     * Additional metadata for this recorded operation
     */
    public val metadata: Map<String, Any> = emptyMap()
)

/**
 * Stack for managing nested execution tapes
 */
public interface TapeStack {
    
    /**
     * The currently active tape (top of stack)
     */
    public val currentTape: ExecutionTape?
    
    /**
     * All tapes in the stack
     */
    public val tapes: List<ExecutionTape>
    
    /**
     * Push a new tape onto the stack
     */
    public fun pushTape(tape: ExecutionTape)
    
    /**
     * Pop the top tape from the stack
     */
    public fun popTape(): ExecutionTape?
    
    /**
     * Clear all tapes from the stack
     */
    public fun clear()
    
    /**
     * Whether any tape in the stack is recording
     */
    public fun isRecording(): Boolean
}

/**
 * Gradient tape for automatic differentiation support
 */
public interface GradientTape : ExecutionTape {
    
    /**
     * Whether to compute gradients for recorded operations
     */
    public val computeGradients: Boolean
    
    /**
     * Compute gradients for the recorded operations
     */
    public fun <T : DType, V> computeGradients(
        targets: List<Tensor<T, V>>,
        sources: List<Tensor<T, V>>
    ): Map<Tensor<T, V>, Tensor<T, V>>
    
    /**
     * Watch tensors for gradient computation
     */
    public fun <T : DType, V> watch(tensors: List<Tensor<T, V>>)
    
    /**
     * Stop watching tensors
     */
    public fun <T : DType, V> stopWatching(tensors: List<Tensor<T, V>>)
}