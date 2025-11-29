package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Minimal multi-input functional wrapper to express a computation as a single callable model.
 *
 * This is intentionally lightweight: it stores input metadata and a build lambda that
 * performs the full forward pass using existing Modules/DSL and returns the final Tensor.
 */
public class Functional<T : DType, V> private constructor(
    public val inputs: List<FuncInput>,
    private val compute: (args: Args<T, V>, ctx: ExecutionContext) -> Tensor<T, V>
) {
    /**
     * Arguments accessor passed to the compute lambda. Retrieves tensors by input name.
     */
    public class Args<T : DType, V>(private val feed: Map<String, Tensor<T, V>>) {
        public operator fun get(name: String): Tensor<T, V> =
            feed[name] ?: error("Functional: missing input tensor for name='$name'")
    }

    /** Executes the functional model with a map of named input tensors. */
    public fun forward(feed: Map<String, Tensor<T, V>>, ctx: ExecutionContext): Tensor<T, V> =
        compute(Args(feed), ctx)

    public companion object {
        /** Factory to create a Functional from input specs and a compute lambda. */
        public fun <T : DType, V> of(
            inputs: List<FuncInput>,
            build: (args: Args<T, V>, ctx: ExecutionContext) -> Tensor<T, V>
        ): Functional<T, V> = Functional(inputs, build)
    }
}

/** Lightweight description of an expected input. Dimensions are optional metadata. */
public data class FuncInput(
    val name: String,
    val dimensions: IntArray = intArrayOf()
)
