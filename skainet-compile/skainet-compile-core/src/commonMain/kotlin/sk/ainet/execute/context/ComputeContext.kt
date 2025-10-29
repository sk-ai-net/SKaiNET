package sk.ainet.execute.context

import sk.ainet.context.ContextDsl
import sk.ainet.context.ExecutionContext
import sk.ainet.execute.context.dsl.ComputationContextDsl
import sk.ainet.execute.context.dsl.ComputationContextDslImpl


@ContextDsl
public fun computation(
    executionContext: ExecutionContext,
    content: ComputationContextDsl.() -> Unit
) {
    val dsl = ComputationContextDslImpl(
        executionContext
    )
    dsl.apply(content)
}

public fun interface ComputationBlockWithContext<V> {
    public fun ComputationContextDsl.invoke(computation: ExecutionContext)
}

@ContextDsl
public fun <V> computation(
    executionContext: ExecutionContext,
    content: ComputationBlockWithContext<V>
) {
    val dsl = ComputationContextDslImpl(executionContext)
    content.run { dsl.invoke(executionContext) }
}
