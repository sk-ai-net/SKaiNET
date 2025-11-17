package sk.ainet.data.mnist

/**
 * JVM implementation of the MNIST loader factory.
 */
public actual object MNISTLoaderFactory {
    /**
     * Creates a new MNIST loader with the default configuration.
     *
     * @return A new MNIST loader.
     */
    public actual fun create(): MNISTLoader {
        return MNISTLoaderJvm.create()
    }

    /**
     * Creates a new MNIST loader with a custom cache directory.
     *
     * @param cacheDir The directory to use for caching.
     * @return A new MNIST loader.
     */
    public actual fun create(cacheDir: String): MNISTLoader {
        return MNISTLoaderJvm.create(cacheDir)
    }

    /**
     * Creates a new MNIST loader with a custom configuration.
     *
     * @param config The configuration to use.
     * @return A new MNIST loader.
     */
    public actual fun create(config: MNISTLoaderConfig): MNISTLoader {
        return MNISTLoaderJvm.create(config)
    }
}