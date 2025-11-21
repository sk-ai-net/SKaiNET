package sk.ainet.context

import sk.ainet.lang.graph.ExecutionTape
import sk.ainet.lang.graph.TapeStack

/**
 * Context for managing execution state, including mode switching,
 * device management, and memory management.
 */
public interface GraphExecutionContext : ExecutionContext {


    /**
     * Current execution tape (null if not recording)
     */
    public val currentTape: ExecutionTape?

    /**
     * Tape stack for nested execution contexts
     */
    public val tapeStack: TapeStack

    /**
     * Whether operations should be recorded
     */
    public val isRecording: Boolean get() = currentTape?.isRecording == true


    /**
     * Force garbage collection
     */
    public fun collectGarbage()

    /**
     * Reset execution statistics
     */
    public fun resetExecutionStats()
}