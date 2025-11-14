package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Convenience operator to call a Module with an explicit ExecutionContext.
 * This does not re-introduce the legacy context-less invoke; callers must pass ctx.
 */
public operator fun <T : DType, V> Module<T, V>.invoke(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
    this.forward(input, ctx)
