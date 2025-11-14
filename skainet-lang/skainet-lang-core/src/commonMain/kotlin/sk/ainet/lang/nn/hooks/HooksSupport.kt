package sk.ainet.lang.nn.hooks

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.topology.ModuleNode

/**
 * Utility to safely dispatch forward hooks if a context with hooks is provided.
 */
public inline fun <I, O> withForwardHooks(
    ctx: ExecutionContext?,
    module: ModuleNode,
    input: I,
    forward: () -> O
): O {
    val hooks = ctx?.hooks ?: return forward()
    hooks.onForwardBegin(module, input as Any)
    val out = forward()
    hooks.onForwardEnd(module, input as Any, out as Any)
    return out
}
