package sk.ainet.io.core.spi

import java.util.ServiceLoader

actual object ProviderDiscovery {
    actual fun discoverProviders(): List<FormatReaderProvider> {
        return try {
            ServiceLoader.load(FormatReaderProvider::class.java).toList()
        } catch (t: Throwable) {
            // Fallback to registry if ServiceLoader fails for any reason
            ProviderRegistry.all()
        }
    }
}
