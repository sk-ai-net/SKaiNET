package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.DType

public class Softmax<T : DType, V>(private val dimension: Int, override val name: String = "Softmax") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            with(input) { softmax(dimension) }
        }
}

