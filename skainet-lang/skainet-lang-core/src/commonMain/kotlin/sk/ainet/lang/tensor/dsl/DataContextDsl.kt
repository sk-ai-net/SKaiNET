package sk.ainet.lang.tensor.dsl

import sk.ainet.context.ContextDsl
import sk.ainet.context.ContextDslItem
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@ContextDsl
// Has to remain public so new keyword/block builder can be attached from other libraries
public interface DataContextDsl : ContextDslItem {

    // make data factory available in context block
    public val executionContext: ExecutionContext

    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>

    // Overload that allows assigning an optional unique name to the tensor within the block
    public fun <T : DType, V> tensor(
        dtype: KClass<T>,
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>
}

public inline fun <reified T : DType, V> DataContextDsl.tensor(
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, content)

public inline fun <reified T : DType, V> DataContextDsl.tensor(
    name: String,
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, name, content)

internal class DataDefinitionContextDslImpl(
    override val executionContext: ExecutionContext,
) : DataContextDsl {
    // Tracks the last tensor created within this DSL instance
    internal var lastTensor: Tensor<*, *>? = null

    // Keeps all named tensors created within the block; names must be unique in this context
    internal val tensorsByName: LinkedHashMap<String, Tensor<*, *>> = LinkedHashMap()

    // Tracks total number of tensors created in the block (named or unnamed)
    internal var createdTensorsCount: Int = 0

    override fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val ctx = TensorFactoryContext<T, V>(executionContext, dtype)
        val t = ctx.content()
        createdTensorsCount++
        lastTensor = t
        return t
    }

    override fun <T : DType, V> tensor(
        dtype: KClass<T>,
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val ctx = TensorFactoryContext<T, V>(executionContext, dtype)
        val t = ctx.content()
        if (tensorsByName.containsKey(name)) {
            error("Tensor name '$name' must be unique within createData block")
        }
        tensorsByName[name] = t
        createdTensorsCount++
        lastTensor = t
        return t
    }
}
