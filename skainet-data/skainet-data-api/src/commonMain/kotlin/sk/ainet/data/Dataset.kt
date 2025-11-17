package sk.ainet.data

import sk.ainet.lang.types.DType
import kotlin.math.min


/** Just abstract Dataset. */
public abstract class Dataset<T, Y> {
    /** Splits datasets on two sub-datasets according [splitRatio].*/
    public abstract fun split(splitRatio: Double): Pair<Dataset<T, Y>, Dataset<T, Y>>

    /** Returns amount of data rows. */
    public abstract val xSize: Int

    /** Returns row by index [idx]. */
    public abstract fun getX(idx: Int): T

    /** Returns label as [Int] by index [idx]. */
    public abstract fun getY(idx: Int): Y

    /** Shuffles the dataset. */
    public abstract fun shuffle(): Dataset<T, Y>

    /**
     * An iterator over a [Dataset].
     */
    public inner class BatchIterator<T : DType, V> internal constructor(
        private val batchSize: Int
    ) : Iterator<DataBatch<T, V>> {

        private var batchStart = 0

        override fun hasNext(): Boolean = batchStart < xSize

        override fun next(): DataBatch<T, V> {
            val batchLength = min(batchSize, xSize - batchStart)
            val batch = createDataBatch<T, V>(batchStart, batchLength)
            batchStart += batchSize
            return batch
        }
    }

    /** Creates data batch that starts from [batchStart] with length [batchLength]. */
    protected abstract fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V>


    /** Returns [BatchIterator] with fixed [batchSize]. */
    public fun <T : DType, V> batchIterator(batchSize: Int): BatchIterator<T, V> {
        return BatchIterator(batchSize)
    }
}