package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Dual-typed module for dtype-transforming or index-consuming ops.
 * InT is the input dtype, OutT is the output dtype.
 * V is the value buffer backend type.
 */
public abstract class DualModule<InT : DType, OutT : DType, V> : ModuleNode {

    /** Human-readable module name */
    public abstract override val name: String

    /** Child modules/nodes for traversal. Keep dtype-agnostic. */
    public abstract val modules: List<ModuleNode>

    /** Forward pass that may optionally use an ExecutionContext. */
    public abstract fun forward(input: Tensor<InT, V>, ctx: ExecutionContext? = null): Tensor<OutT, V>

    // ModuleNode implementation
    override val id: String get() = name
    override var path: String? = null
    override val children: List<ModuleNode> get() = modules

    @Suppress("UNCHECKED_CAST")
    override val params: List<ModuleParameter<*, *>>
        get() = when (this) {
            is ModuleParameters<*, *> -> (this as ModuleParameters<Any?, Any?>).params as List<ModuleParameter<*, *>>
            else -> emptyList()
        }

    public operator fun invoke(input: Tensor<InT, V>, ctx: ExecutionContext? = null): Tensor<OutT, V> = forward(input, ctx)
}

/**
 * Composition helpers to chain unary and dual modules.
 */
public fun <T : DType, U : DType, V> compose(u: Module<T, V>, d: DualModule<T, U, V>): DualModule<T, U, V> =
    object : DualModule<T, U, V>() {
        override val name: String = "Compose(${u.name}→${d.name})"
        override val modules: List<ModuleNode> = listOf(u, d)
        override fun forward(input: Tensor<T, V>, ctx: ExecutionContext?): Tensor<U, V> = d.forward(u.forward(input), ctx)
    }

public fun <T : DType, U : DType, V> compose(d: DualModule<T, U, V>, u: Module<U, V>): DualModule<T, U, V> =
    object : DualModule<T, U, V>() {
        override val name: String = "Compose(${d.name}→${u.name})"
        override val modules: List<ModuleNode> = listOf(d, u)
        override fun forward(input: Tensor<T, V>, ctx: ExecutionContext?): Tensor<U, V> = u.forward(d.forward(input, ctx))
    }
