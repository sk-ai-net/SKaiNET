package sk.ainet.compile.nn

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.sequential


import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.NeuralNetworkExecutionContext
import sk.ainet.lang.types.DType


/**
 * Context for the DSL to define the data type and operations.
 *
 * This class holds the information about the data type and operations
 * that should be used in the DSL. It's used to make the DSL generic
 * and to avoid hardcoding the data type.
 *
 * @param T The default data type.
 */
public interface NeuralNetworkContext<T : DType, V> {

    public val executionContext: NeuralNetworkExecutionContext

}

/**
 * Creates a context for the DSL with the given configuration.
 *
 * @param T The type of data processed by the modules.
 * @param init The configuration function.
 * @return The configured context.
 */
public fun <T : DType, V> definition(init: NeuralNetworkContext<T, V>.(NeuralNetworkContext<T, V>) -> Module<T, V>): Module<T, V> {
    val instance = DefaultNetworkContext<T, V>()
    return instance.init(instance)
}

/**
 * Extension function to create a network within a NetworkContext.
 * This bridges the context wrapper with the network DSL using the context's tensor factory.
 */
public inline fun <reified T : DType, V> NeuralNetworkContext<T, V>.network(
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> = sequential(executionContext, content)

public class DefaultNetworkContext<T : DType, V> : NeuralNetworkContext<T, V> {
    override val executionContext: NeuralNetworkExecutionContext
        get() = DefaultNeuralNetworkExecutionContext()
}