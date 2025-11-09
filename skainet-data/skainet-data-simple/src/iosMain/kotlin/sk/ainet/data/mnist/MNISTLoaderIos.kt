package sk.ainet.data.mnist

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.darwin.Darwin
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * iOS implementation of the MNIST loader.
 *
 * @property config The configuration for the MNIST loader.
 */
public class MNISTLoaderIos(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {

    /**
     * Downloads and caches a file.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            // In this simplified iOS implementation, we don't cache files
            // We'll just download the file every time
            println("Downloading file: $url")
            val data = downloadFile(url)

            // Note: In a real implementation, we would use a native gzip library to decompress the data
            // For this example, we're assuming the server provides uncompressed data for iOS clients
            println("iOS implementation does not support gzip decompression in this example. Assuming data is already decompressed.")

            return@withContext data
        }

    /**
     * Downloads a file from a URL.
     *
     * @param url The URL to download from.
     * @return The bytes of the file.
     */
    private suspend fun downloadFile(url: String): ByteArray {
        val client = HttpClient(Darwin) {
            // No plugins needed for basic functionality
        }

        try {
            val httpResponse: HttpResponse = client.get(url)
            return httpResponse.body()
        } finally {
            client.close()
        }
    }

    public companion object {
        /**
         * Creates a new instance of MNISTLoaderIos with the default configuration.
         *
         * @return A new instance of MNISTLoaderIos.
         */
        public fun create(): MNISTLoaderIos {
            return MNISTLoaderIos(MNISTLoaderConfig())
        }

        /**
         * Creates a new instance of MNISTLoaderIos with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of MNISTLoaderIos.
         */
        public fun create(cacheDir: String): MNISTLoaderIos {
            return MNISTLoaderIos(MNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of MNISTLoaderIos with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of MNISTLoaderIos.
         */
        public fun create(config: MNISTLoaderConfig): MNISTLoaderIos {
            return MNISTLoaderIos(config)
        }
    }
}