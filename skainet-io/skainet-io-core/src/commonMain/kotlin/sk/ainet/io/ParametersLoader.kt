package sk.ainet.io

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

interface ParametersLoader {
    suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    )
}

suspend inline fun <reified T : DType, V> ParametersLoader.load(
    ctx: ExecutionContext,
    noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
) = load(ctx, T::class, onTensorLoaded)