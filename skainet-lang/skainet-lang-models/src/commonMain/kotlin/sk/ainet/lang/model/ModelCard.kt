package sk.ainet.lang.model

/**
 * Kotlin representation of a model card as described by the provided YAML schema.
 *
 * Notes on field naming:
 * - YAML keys that contain hyphens or underscores are mapped to idiomatic camelCase properties.
 * - This file intentionally avoids adding serialization dependencies; it is a plain data model.
 */
public data class ModelCard(

    // license: apache-2.0
    val license: String,

    // library_name: transformers
    val libraryName: String,

    // pipeline_tag: text-generation
    val pipelineTag: String,

    // language: [en]
    val language: List<String>,

    // modalities: [text]
    val modalities: List<String>,

    // base_model: meta-llama/Llama-3-8B
    val baseModel: String,

    // context_length: 8192
    val contextLength: Int,

    // datasets: [openorca, truthful_qa]
    val datasets: List<String>,

    // metrics: [arc, hellaswag]
    val metrics: List<String>,

    // model-index: [...]
    val modelIndex: List<ModelIndexEntry>,

    // intended_use: | ...
    val intendedUse: String,

    // limitations: | ...
    val limitations: String,
)

public data class ModelIndexEntry(
    val name: String,
    val results: List<ModelResult>,
)

public data class ModelResult(
    val task: Task,
    val dataset: Dataset,
    val metrics: List<Metric>,
)

public data class Task(
    val type: String,
)

public data class Dataset(
    val name: String,
    val type: String,
)

public data class Metric(
    val name: String,
    val value: Double,
)