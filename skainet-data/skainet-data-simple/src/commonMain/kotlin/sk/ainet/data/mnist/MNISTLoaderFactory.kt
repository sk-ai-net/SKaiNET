package sk.ainet.data.mnist

/**
 * Factory for creating MNIST loaders.
 */
public expect object MNISTLoaderFactory {
    /**
     * Creates a new MNIST loader with the default configuration.
     *
     * @return A new MNIST loader.
     */
    public fun create(): MNISTLoader

    /**
     * Creates a new MNIST loader with a custom cache directory.
     *
     * @param cacheDir The directory to use for caching.
     * @return A new MNIST loader.
     */
    public fun create(cacheDir: String): MNISTLoader

    /**
     * Creates a new MNIST loader with a custom configuration.
     *
     * @param config The configuration to use.
     * @return A new MNIST loader.
     */
    public fun create(config: MNISTLoaderConfig): MNISTLoader
}