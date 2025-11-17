package sk.ainet.context

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.DataContextDsl
import sk.ainet.lang.tensor.dsl.DataDefinitionContextDslImpl
import sk.ainet.lang.tensor.dsl.TypedDataContextDsl
import sk.ainet.lang.tensor.dsl.TypedDataContextDslImpl
import sk.ainet.lang.types.DType

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

// Typed overload that returns a Tensor<T, V> and provides a default dtype for the block
@ContextDsl
public inline fun <reified T : DType, V> data(
    executionContext: ExecutionContext = DefaultDataExecutionContext(),
    noinline content: TypedDataContextDsl<T, V>.(executionContext: ExecutionContext) -> Tensor<T, V>
): Tensor<T, V> {
    val dsl: TypedDataContextDsl<T, V> = TypedDataContextDslImpl(executionContext, T::class)
    return dsl.content(executionContext)
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
