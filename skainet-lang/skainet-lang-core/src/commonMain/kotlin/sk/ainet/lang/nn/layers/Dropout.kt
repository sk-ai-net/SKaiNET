package sk.ainet.lang.nn.layers

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Dropout layer that is aware of ExecutionContext phases.
 *
 * Current implementation keeps identity semantics on both TRAIN and EVAL phases,
 * but exposes a context-aware forward that can later be extended to apply
 * stochastic masking when ctx.inTraining is true.
 */
public class Dropout<T : DType, V>(
    public val p: Float = 0.5f,
    public var training: Boolean = true,
    override val name: String = "Dropout"
) : Module<T, V>() {

    init {
        require(p >= 0f) { "Dropout($name): p must be >= 0, was $p" }
        require(p < 1f) { "Dropout($name): p must be < 1 to avoid division by zero, was $p" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    /**
     * Context-aware forward that can use ExecutionContext.phase. Hooks are dispatched if available.
     */
    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            // Placeholder behavior: identity in both phases. When RNG and elementwise ops are available,
            // implement: if (ctx.inTraining && p > 0f) then output = input * mask / (1 - p)
            input
        }
}
