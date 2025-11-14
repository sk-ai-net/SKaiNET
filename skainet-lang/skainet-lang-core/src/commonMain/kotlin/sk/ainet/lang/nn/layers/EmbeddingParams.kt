package sk.ainet.lang.nn.layers

/**
 * Parameters for configuring an Embedding layer.
 *
 * This is a lightweight holder used to describe the table size and optional
 * behaviors. Some fields are placeholders for future features (no-ops for now)
 * to keep parity with common NN APIs.
 */
public data class EmbeddingParams(
    val numEmbeddings: Int,
    val embeddingDim: Int,
    val paddingIdx: Int? = null,
    val maxNorm: Float? = null,           // reserved, not applied yet
    val scaleGradByFreq: Boolean = false  // reserved, not applied yet
)