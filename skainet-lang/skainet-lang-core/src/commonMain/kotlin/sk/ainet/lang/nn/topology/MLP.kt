package sk.ainet.lang.nn.topology

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public class MLP<T : DType, V>(vararg modules: Module<T, V>, override val name: String = "FeedForwardNetwork") :
    Module<T, V>(), ModuleParameters<T, V> {
    private val modulesList = modules.toList()
    override val modules: List<Module<T, V>>
        get() = modulesList

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            var tmp = input
            modulesList.forEach { module ->
                tmp = module.forward(tmp, ctx)
            }
            tmp
        }

    override val params: List<ModuleParameter<T, V>>
        get() = emptyList()
}