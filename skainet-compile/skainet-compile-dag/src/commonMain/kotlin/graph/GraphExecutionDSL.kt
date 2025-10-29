package sk.ainet.lang.graph

import sk.ainet.context.ExecutionStats
import sk.ainet.context.GraphExecutionContext
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType

/**
 * Result of graph execution containing the computed result and execution metadata
 */
public data class GraphExecutionResult<V>(
    val result: V,
    val graph: ComputeGraph,
    val tape: ExecutionTape?,
) {
    /**
     * Execute the recorded graph with the current execution context
     */
    public fun execute(): V {
        // TODO: Implement graph execution
        return result
    }

}


/**
 * Creates a graph execution context and executes the given block within it.
 * Similar to the network context pattern but for graph execution.
 *
 * Usage:
 * val result = exec<FP32, Float> {
 *     a + b
 * }
 */
public fun <V, R> exec(
    dataFactory: TensorDataFactory = DenseTensorDataFactory(),
    block: GraphExecutionContext<V>.() -> R
) {//}: GraphExecutionResult<V> {
    /*

    return try {
    //    setCurrentContext(context)
        context.executionContext.startRecording(context.executionContext.createTape())
        
        val result = context.block()
        val tape = context.executionContext.stopRecording()
        
        GraphExecutionResult(
            result = result,
            graph = context.graph,
            tape = tape,
            executionContext = context.executionContext
        )
    } finally {
        clearCurrentContext()
    }

     */
}