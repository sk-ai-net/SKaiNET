package sk.ainet.io.core.spi

import sk.ainet.io.core.CloseableTensorArchive
import sk.ainet.io.core.TensorReader
import sk.ainet.io.core.TensorSource

/**
 * Provider SPI and probe results used by [TensorReader] to select a concrete format implementation.
 */
interface FormatReaderProvider {
    /** Stable identifier of the format, e.g. "safetensors", "gguf". */
    fun formatId(): String

    /**
     * Probe the given [source] to determine whether this provider can read it.
     * Providers should avoid throwing; return an unsupported result with [reason] instead.
     */
    fun probe(source: TensorSource): ProbeResult

    /** Open the archive. Called only for providers that won the probe selection. */
    fun open(source: TensorSource): CloseableTensorArchive
}

/** Confidence-based probe outcome used to select a provider. */
data class ProbeResult(
    val supported: Boolean,
    /** Confidence in [0, 100]. Higher wins. */
    val confidence: Int,
    val version: String? = null,
    val reason: String? = null,
    /** Optional provider id for diagnostics. */
    val formatId: String? = null
) {
    init {
        require(confidence in 0..100) { "confidence must be in 0..100" }
    }

    fun isBetterThan(other: ProbeResult?): Boolean {
        if (other == null) return supported
        if (this.supported != other.supported) return this.supported
        return this.confidence > other.confidence
    }

    companion object {
        fun unsupported(reason: String? = null, formatId: String? = null): ProbeResult =
            ProbeResult(false, 0, null, reason, formatId)

        fun supported(confidence: Int, version: String? = null, formatId: String? = null, reason: String? = null): ProbeResult =
            ProbeResult(true, confidence, version, reason, formatId)

        /** Prefer header-based high confidence over mere extension guesses. */
        val HEADER_STRONG = 90
        val HEADER_WEAK = 70
        val EXTENSION_HINT = 40
    }
}

/**
 * Cross-platform registry for providers. Useful for Native/JS where ServiceLoader is not available,
 * and also as an override/explicit registration mechanism on all platforms (including tests).
 */
object ProviderRegistry {
    private val providers = mutableListOf<FormatReaderProvider>()

    fun register(provider: FormatReaderProvider) {
        // de-duplicate by formatId
        val id = provider.formatId()
        providers.removeAll { it.formatId() == id }
        providers.add(provider)
    }

    fun unregister(formatId: String) {
        providers.removeAll { it.formatId() == formatId }
    }

    fun clear() { providers.clear() }

    fun all(): List<FormatReaderProvider> = providers.toList()
}

/** Expect/actual discovery that may use platform mechanisms (ServiceLoader on JVM). */
expect object ProviderDiscovery {
    fun discoverProviders(): List<FormatReaderProvider>
}
