package sk.ainet.io.core.spi

actual object ProviderDiscovery {
    actual fun discoverProviders(): List<FormatReaderProvider> {
        // Prefer runtime registry on Android to avoid reflection costs.
        return ProviderRegistry.all()
    }
}
