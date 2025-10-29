package sk.ainet.context

import sk.ainet.lang.tensor.dsl.DataContextDsl
import sk.ainet.lang.tensor.dsl.DataDefinitionContextDslImpl

@DslMarker
public annotation class ContextDsl

@ContextDsl
public interface ContextDslItem


// Overload that exposes the executionContext as a lambda parameter inside the DSL block
@ContextDsl
public fun  data(
    executionContext: ExecutionContext = DefaultDataExecutionContext(),
    content: DataContextDsl.(executionContext: ExecutionContext) -> Unit
) {
    val dsl = DataDefinitionContextDslImpl(executionContext)
    dsl.content(executionContext)
}
