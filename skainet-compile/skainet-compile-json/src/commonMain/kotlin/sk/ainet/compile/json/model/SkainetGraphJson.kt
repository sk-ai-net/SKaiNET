package sk.ainet.compile.json.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * Data classes describing the JSON export format used by run14.onnx.json.
 * These models are intentionally flexible (strings for IDs and values, nullable/JsonElement for free-form fields)
 * so we can export various graphs without coupling to a specific backend.
 */

@Serializable
public data class SkainetModelJson(
    val label: String,
    val graphs: List<SkainetGraphJson>
)

@Serializable
public data class SkainetGraphJson(
    val id: String,
    val nodes: List<SkainetNodeJson>
)

@Serializable
public data class SkainetNodeJson(
    val id: String,
    val label: String,
    val namespace: String = "",
    val subgraphIds: List<String> = emptyList(),
    val attrs: List<SkainetAttrJson> = emptyList(),
    val incomingEdges: List<SkainetEdgeJson> = emptyList(),
    val outputsMetadata: List<SkainetPortMetadataJson> = emptyList(),
    val inputsMetadata: List<SkainetPortMetadataJson> = emptyList(),
    val style: JsonElement? = null,
    val config: JsonElement? = null
)

@Serializable
public data class SkainetAttrJson(
    val key: String,
    val value: String
)

@Serializable
public data class SkainetEdgeJson(
    val sourceNodeId: String,
    val sourceNodeOutputId: String,
    val targetNodeInputId: String
)

@Serializable
public data class SkainetPortMetadataJson(
    val id: String,
    val attrs: List<SkainetAttrJson> = emptyList()
)
