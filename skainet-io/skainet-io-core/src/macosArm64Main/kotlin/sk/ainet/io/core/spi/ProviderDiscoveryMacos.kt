package sk.ainet.io.core.spi

actual object ProviderDiscovery {
    actual fun discoverProviders(): List<FormatReaderProvider> {
        return ProviderRegistry.all()
    }
}
