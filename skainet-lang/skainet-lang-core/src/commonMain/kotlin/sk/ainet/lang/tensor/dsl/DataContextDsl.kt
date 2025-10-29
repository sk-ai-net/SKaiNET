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
}

public inline fun <reified T : DType, V> DataContextDsl.tensor(
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, content)

internal class DataDefinitionContextDslImpl(
    override val executionContext: ExecutionContext,
) : DataContextDsl {
    override fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val ctx = TensorFactoryContext<T, V>(executionContext, dtype)
        return ctx.content()
    }
}
