package sk.ainet.io.core

import sk.ainet.io.core.spi.FormatReaderProvider
import sk.ainet.io.core.spi.ProviderDiscovery
import sk.ainet.io.core.spi.ProviderRegistry
import sk.ainet.io.core.spi.ProbeResult

/**
 * Default implementation that discovers providers, probes the source, and opens
 * the archive using the highest-confidence supported provider.
 */
class DefaultTensorReader : TensorReader {
    override fun open(source: TensorSource): CloseableTensorArchive {
        val discovered: List<FormatReaderProvider> = ProviderDiscovery.discoverProviders()
        // Merge with runtime-registered ones while de-duplicating by formatId
        val byId = LinkedHashMap<String, FormatReaderProvider>()
        for (p in discovered) byId[p.formatId()] = p
        for (p in ProviderRegistry.all()) byId[p.formatId()] = p
        val providers = byId.values.toList()
        if (providers.isEmpty()) {
            throw IllegalStateException("No FormatReaderProvider implementations discovered. Register via ProviderRegistry or ServiceLoader on JVM.")
        }

        var bestProvider: FormatReaderProvider? = null
        var bestProbe: ProbeResult? = null
        val reasons = mutableListOf<String>()

        for (p in providers) {
            val probe = try {
                val res = p.probe(source)
                if (res.formatId == null) res.copy(formatId = p.formatId()) else res
            } catch (t: Throwable) {
                reasons += "${p.formatId()}: probe threw ${t::class.simpleName}: ${t.message}"
                continue
            }
            if (!probe.supported) {
                val why = probe.reason?.let { ": $it" } ?: ""
                reasons += "${p.formatId()}: unsupported$why"
            }
            if (probe.isBetterThan(bestProbe)) {
                bestProvider = p
                bestProbe = probe
            }
        }

        if (bestProvider == null || bestProbe == null || !bestProbe.supported) {
            val details = if (reasons.isEmpty()) "no providers reported reasons" else reasons.joinToString("; ")
            throw IllegalArgumentException("No provider supports source ${source.id}. Details: $details")
        }

        return bestProvider!!.open(source)
    }
}
