package sk.ainet.data.mnist

/**
 * JS implementation of the MNIST loader factory.
 */
public actual object MNISTLoaderFactory {
    public actual fun create(): MNISTLoader = MNISTLoaderJs.create()
    public actual fun create(cacheDir: String): MNISTLoader = MNISTLoaderJs.create(cacheDir)
    public actual fun create(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderJs.create(config)
}
