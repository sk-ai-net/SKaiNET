package sk.ainet.context

/**
 * Memory usage information
 */
public data class MemoryInfo(
    /**
     * Total memory available on the device
     */
    public val totalMemory: Long,

    /**
     * Memory currently in use
     */
    public val usedMemory: Long,

    /**
     * Free memory available
     */
    public val freeMemory: Long = totalMemory - usedMemory,

    /**
     * Memory usage as percentage
     */
    public val usagePercentage: Double = (usedMemory.toDouble() / totalMemory) * 100.0
) {
    public companion object Companion {
        public fun getEmptyInfo(): MemoryInfo = MemoryInfo(
            totalMemory = 0,
            usedMemory = 0,
            freeMemory = 0,
            usagePercentage = 0.0
        )

    }
}


/**
 * Execution statistics
 */
public data class ExecutionStats(
    /**
     * Total number of operations executed
     */
    public val operationsExecuted: Long = 0,

    /**
     * Total execution time in milliseconds
     */
    public val totalExecutionTime: Long = 0,

    /**
     * Average execution time per operation
     */
    public val averageExecutionTime: Double =
        if (operationsExecuted > 0) totalExecutionTime.toDouble() / operationsExecuted else 0.0,

    /**
     * Number of tensors created
     */
    public val tensorsCreated: Long = 0,

    /**
     * Peak memory usage
     */
    public val peakMemoryUsage: Long = 0
)


