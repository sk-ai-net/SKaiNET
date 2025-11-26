package sk.ainet.lang.graph.dsl

import sk.ainet.context.ContextDsl
import sk.ainet.context.ContextDslItem
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.tape.ExecutionTape
import sk.ainet.lang.graph.exec.GraphExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultComputeGraph

/**
 * Result of graph execution containing the computed result and execution metadata
 */
public data class GraphExecutionResult<V>(
    val result: V,
    val graph: ComputeGraph,
    val tape: ExecutionTape,
)

@ContextDsl
public interface GraphDslItem


/**
 * Creates a graph execution context and executes the given block within it.
 * Similar to the network context pattern but for graph execution.
 *
 * Usage:
 * val result:GraphExecutionResult<Float> = graphExec<FP32, Float> {
 *     a + b
 * }
 */
public fun <V, R> compileGraphExec(
    graphExecutionContext: GraphExecutionContext,
    block: GraphExecContextDsl.(executionContext: GraphExecutionContext) -> R
): GraphExecutionResult<R> {
    // Manually manage tape via factory to avoid platform reflection
    val tape = graphExecutionContext.createTapeFactory(graphExecutionContext)
    tape.startRecording()
    graphExecutionContext.tapeStack.pushTape(tape)

    val dsl = GraphExecContextDslImpl(graphExecutionContext)
    val result = try {
        dsl.block(graphExecutionContext)
    } finally {
        // Pop and stop regardless of success
        graphExecutionContext.tapeStack.popTape()
        tape.stopRecording()
    }

    val nonNullTape: ExecutionTape = tape
    val graph = (nonNullTape as? DefaultExecutionTape)?.toComputeGraph()
        ?: DefaultComputeGraph()

    return GraphExecutionResult(result, graph, nonNullTape)
}

@ContextDsl
// Has to remain public so new keyword/block builder can be attached from other libraries
public interface GraphExecContextDsl : ContextDslItem {

}

public class GraphExecContextDslImpl(private val graphExecutionContext: GraphExecutionContext) : GraphExecContextDsl {
}
