package sk.ainet.lang.nn.hooks

import sk.ainet.lang.nn.topology.ModuleNode

/**
 * Forward pass hook interface. Implementations can record diagnostics, timings, or build a tape.
 */
public interface ForwardHooks {
    public fun onForwardBegin(module: ModuleNode, input: Any)
    public fun onForwardEnd(module: ModuleNode, input: Any, output: Any)
}
