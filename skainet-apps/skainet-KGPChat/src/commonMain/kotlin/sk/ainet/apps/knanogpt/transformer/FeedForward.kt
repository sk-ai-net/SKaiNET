package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module


class FeedForward(
    config: TransformerConfig,
    override val name: String = "FeedForward"
) : Module() {

    private val sequential: FeedForwardNetwork

    init {
        with(config) {
            sequential = network {
                input(n_embd)
                dense(4 * n_embd) {
                    activation = relu

                }
                dense(n_embd)
                dropout(dropout)
            }
        }
    }

    override val params: List<NamedParameter>
        get() = sequential.params
    override val modules: List<Module>
        get() = sequential.modules

    override fun forward(input: Tensor, ctx: ExecutionContext): Tensor =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            sequential.forward(input, ctx)
        }
}
