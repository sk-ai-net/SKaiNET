package sk.ainet.lang.nn.topology

import sk.ainet.lang.nn.Module

/**
 * A generic, dtype-agnostic node interface for traversing module hierarchies.
 * This allows programmatic traversal, parameter collection, and introspection
 * without depending on the input/output dtype generics of Module.
 */
public interface ModuleNode {
    /** A stable identifier for the node. Not required to be globally unique. */
    public val id: String

    /** Human-readable name of the node, typically the module's name. */
    public val name: String

    /** Optional path annotation for nicer logs, e.g., "model.encoder.layer1". */
    public var path: String?

    /** Child nodes in the hierarchy. */
    public val children: List<ModuleNode>

    /** Parameters owned by this node (weights, biases, etc.). */
    public val params: List<ModuleParameter<*, *>>
}

/**
 * Default ModuleNode implementation adapter for legacy Module<T,V> types.
 * This provides a lightweight bridge to the traversal API without changing
 * the existing Module APIs beyond implementing ModuleNode.
 */
public abstract class ModuleNodeAdapter : ModuleNode {
    override var path: String? = null
}
