package de.jugda.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.topology.ModuleParameter as NamedParameter
import sk.ainet.lang.tensor.ext.cat
import sk.ainet.apps.knanogpt.transformer.Head
import sk.ainet.apps.knanogpt.transformer.TransformerConfig


/**
 *     """ multiple heads of self-attention in parallel """
 *
 */
class MultiHeadAttention(
    config: TransformerConfig,
    override val name: String = "MultiHeadAttention"
) : Module() {

    private val _modules = mutableListOf<Module>()
    private val _heads = mutableListOf<Module>()

    init {
        with(config) {
            _heads += List(num_heads) {
                // Head now requires an ExecutionContext; we will pass it at call time
                Head(config, dropout, it)
            }
            _modules += listOf(
                Linear(head_size * num_heads, n_embd),
                Dropout(dropout)
            )
        }
    }

    override val params: List<NamedParameter>
        get() = modules.map { it.params }.flatten()
    override val modules: List<Module>
        get() = _modules

    override fun forward(input: Tensor, ctx: ExecutionContext): Tensor =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val conectcated = cat(
                _heads.map { head ->
                    head.forward(input, ctx)
                },
                dim = -1
            )

            var data = conectcated
            _modules.forEach { module ->
                data = module.forward(data, ctx)
            }
            data
        }
}


