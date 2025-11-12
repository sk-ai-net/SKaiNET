package sk.ainet.lang.nn.layers

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Dropout layer (API skeleton). Currently acts as identity in forward pass while
 * providing parameter validation and training flag. This keeps behavior consistent across
 * backends without introducing RNG dependencies here.
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

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        // TODO(skainet #module-1.5): Respect ExecutionContext.inTraining when context-aware forward is available.
        // For now, identity semantics. Future: when training, apply mask and scale by 1/(1-p)
        return input
    }
}
