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


// Variant that returns a map of all tensors created in the block, keyed by their unique names
// All tensors in the block must be named using the named `tensor(...)` overload; names must be unique
@ContextDsl
public fun createDataMap(
    executionContext: ExecutionContext = DefaultDataExecutionContext(),
    content: DataContextDsl.(executionContext: ExecutionContext) -> Unit
): Map<String, Tensor<*, *>> {
    val dsl = DataDefinitionContextDslImpl(executionContext)
    dsl.content(executionContext)
    if (dsl.createdTensorsCount == 0) error("No tensor was created in createData block")
    if (dsl.tensorsByName.size != dsl.createdTensorsCount) {
        error("All tensors in createDataMap block must be named uniquely")
    }
    // return an immutable copy
    return dsl.tensorsByName.toMap()
}
