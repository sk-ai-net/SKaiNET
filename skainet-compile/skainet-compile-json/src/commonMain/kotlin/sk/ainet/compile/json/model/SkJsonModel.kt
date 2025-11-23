package sk.ainet.compile.json.model

import kotlinx.serialization.Serializable

/**
 * Kotlinx-serializable data model for JSON export as specified in json-tasks.md.
 */

@Serializable
public data class SkJsonExport(
    val label: String,
    val graphs: List<SkJsonGraph>
)

@Serializable
public data class SkJsonGraph(
    val id: String,
    val nodes: List<SkJsonNode>,
    val attrs: List<SkJsonAttr> = emptyList()
)

@Serializable
public data class SkJsonNode(
    val id: String,
    val label: String,
    val namespace: String = "",
    val subgraphIds: List<String> = emptyList(),
    val attrs: List<SkJsonAttr> = emptyList(),
    val incomingEdges: List<SkJsonEdgeRef> = emptyList(),
    val outputsMetadata: List<SkJsonPortMeta> = emptyList(),
    val inputsMetadata: List<SkJsonPortMeta> = emptyList(),
    val style: SkJsonStyle? = null,
    val config: SkJsonConfig? = null,
)

@Serializable
public data class SkJsonAttr(
    val key: String,
    val value: String
)

@Serializable
public data class SkJsonEdgeRef(
    val sourceNodeId: String,
    val sourceNodeOutputId: String,
    val targetNodeInputId: String
)

@Serializable
public data class SkJsonPortMeta(
    val id: String,
    val attrs: List<SkJsonAttr>
)

// Empty placeholder types for future styling and configuration.
@Serializable
public class SkJsonStyle

@Serializable
public class SkJsonConfig
