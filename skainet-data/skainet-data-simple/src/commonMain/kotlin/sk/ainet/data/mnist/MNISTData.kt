package sk.ainet.data.mnist

import kotlinx.serialization.Serializable

/**
 * Represents a single MNIST image with its label.
 *
 * @property image The pixel data of the image as a ByteArray (28x28 pixels).
 * @property label The label of the image (0-9).
 */
@Serializable
public data class MNISTImage(
    val image: ByteArray,
    val label: Byte
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as MNISTImage

        if (!image.contentEquals(other.image)) return false
        if (label != other.label) return false

        return true
    }

    override fun hashCode(): Int {
        var result = image.contentHashCode()
        result = 31 * result + label.toInt()
        return result
    }
}

/**
 * Represents a dataset of MNIST images.
 *
 * @property images The list of MNIST images.
 */
@Serializable
public data class MNISTDataset(
    val images: List<MNISTImage>
) {
    /**
     * Returns the number of images in the dataset.
     */
    val size: Int
        get() = images.size

    /**
     * Returns a subset of the dataset.
     *
     * @param fromIndex The starting index (inclusive).
     * @param toIndex The ending index (exclusive).
     * @return A new MNISTDataset containing the specified range of images.
     */
    public fun subset(fromIndex: Int, toIndex: Int): MNISTDataset {
        return MNISTDataset(images.subList(fromIndex, toIndex))
    }
}

/**
 * Configuration for the MNIST loader.
 *
 * @property cacheDir The directory where downloaded files will be cached.
 * @property useCache Whether to use cached files if available.
 */
public data class MNISTLoaderConfig(
    val cacheDir: String = "mnist-data",
    val useCache: Boolean = true
)

/**
 * Constants for the MNIST dataset.
 */
public object MNISTConstants {
    public const val IMAGE_SIZE: Int = 28
    public const val IMAGE_PIXELS: Int = IMAGE_SIZE * IMAGE_SIZE

    public const val TRAIN_IMAGES_FILENAME: String = "train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_FILENAME: String = "train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_FILENAME: String = "t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_FILENAME: String = "t10k-labels-idx1-ubyte.gz"

    public const val TRAIN_IMAGES_URL: String =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_URL: String =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_URL: String = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_URL: String = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}

/**
 * Interface for the MNIST loader.
 */
public interface MNISTLoader {
    /**
     * Loads the MNIST training dataset.
     *
     * @return The MNIST training dataset.
     */
    public suspend fun loadTrainingData(): MNISTDataset

    /**
     * Loads the MNIST test dataset.
     *
     * @return The MNIST test dataset.
     */
    public suspend fun loadTestData(): MNISTDataset
}