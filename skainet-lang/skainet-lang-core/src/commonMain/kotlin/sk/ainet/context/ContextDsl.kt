package sk.ainet.context

import sk.ainet.lang.tensor.Tensor
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

// Variant that returns the last created tensor from the context block
@ContextDsl
public fun createData(
    executionContext: ExecutionContext = DefaultDataExecutionContext(),
    content: DataContextDsl.(executionContext: ExecutionContext) -> Unit
): Tensor<*, *> {
    val dsl = DataDefinitionContextDslImpl(executionContext)
    dsl.content(executionContext)
    return dsl.lastTensor ?: error("No tensor was created in createData block")
}
