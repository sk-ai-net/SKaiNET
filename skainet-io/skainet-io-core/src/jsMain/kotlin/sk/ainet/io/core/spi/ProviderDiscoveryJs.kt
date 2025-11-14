package sk.ainet.io.core.spi

actual object ProviderDiscovery {
    actual fun discoverProviders(): List<FormatReaderProvider> {
        // No ServiceLoader on JS; rely on runtime registry
        return ProviderRegistry.all()
    }
}
