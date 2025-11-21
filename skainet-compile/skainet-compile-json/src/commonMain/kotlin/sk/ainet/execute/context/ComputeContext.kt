package sk.ainet.execute.context

import sk.ainet.context.ContextDsl
import sk.ainet.context.ExecutionContext
import sk.ainet.execute.context.dsl.ComputationContextDsl
import sk.ainet.execute.context.dsl.ComputationContextDslImpl



@ContextDsl
public fun <V> computation(
    executionContext: ExecutionContext,
    content: ComputationContextDsl.(ExecutionContext) -> V
): V {
    val dsl = ComputationContextDslImpl(executionContext)
    return dsl.content(executionContext)
}
