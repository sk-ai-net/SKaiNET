package sk.ainet.data

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.test.assertNotEquals

/**
 * A lightweight fake dataset for tests that stores simple feature vectors and labels
 * and can build DataBatch tensors using the tensor DSL.
 */
private class FakeDataset(
    private val features: List<FloatArray>,
    private val labels: List<Float>,
    private val ctx: DefaultDataExecutionContext = DefaultDataExecutionContext()
) : Dataset<FloatArray, Float>() {

    override fun split(splitRatio: Double): Pair<Dataset<FloatArray, Float>, Dataset<FloatArray, Float>> {
        val splitIndex = (features.size * splitRatio).toInt().coerceIn(0, features.size)
        val left = FakeDataset(features.subList(0, splitIndex), labels.subList(0, splitIndex), ctx)
        val right =
            FakeDataset(features.subList(splitIndex, features.size), labels.subList(splitIndex, labels.size), ctx)
        return left to right
    }

    override val xSize: Int get() = features.size

    override fun getX(idx: Int): FloatArray = features[idx]
    override fun getY(idx: Int): Float = labels[idx]

    override fun shuffle(): Dataset<FloatArray, Float> {
        val indices = features.indices.shuffled()
        val newFeatures = indices.map { features[it] }
        val newLabels = indices.map { labels[it] }
        return FakeDataset(newFeatures, newLabels, ctx)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val end = (batchStart + batchLength).coerceAtMost(features.size)
        val sliceF = features.subList(batchStart, end)
        val sliceY = labels.subList(batchStart, end)
        val featureSize = if (sliceF.isNotEmpty()) sliceF[0].size else 0

        // Build a single input tensor [batch, feature]
        val xTensor: Tensor<FP32, Float> = tensor(ctx, FP32::class) {
            tensor {
                shape(end - batchStart, featureSize) {
                    val flat = FloatArray((end - batchStart) * featureSize)
                    var k = 0
                    for (i in sliceF.indices) {
                        val row = sliceF[i]
                        for (j in 0 until featureSize) {
                            flat[k++] = row[j]
                        }
                    }
                    fromArray(flat)
                }
            }
        }

        // Build label tensor [batch]
        val yTensor: Tensor<FP32, Float> = tensor(ctx, FP32::class) {
            tensor {
                shape(end - batchStart) {
                    val flat = FloatArray(end - batchStart) { idx -> sliceY[idx] }
                    fromArray(flat)
                }
            }
        }

        val batch = DataBatch(arrayOf(xTensor), yTensor)
        return batch as DataBatch<T, V>
    }
}

class DatasetAndDataBatchTest {

    @Test
    fun dataBatchEqualityAndHashCode() {
        val ctx = DefaultDataExecutionContext()
        val x1: Tensor<FP32, Float> = data<FP32, Float>(ctx) { tensor { shape(2, 2) { from(1f, 2f, 3f, 4f) } } }
        val y1: Tensor<FP32, Float> = tensor(ctx, FP32::class) { tensor { shape(2) { from(0f, 1f) } } }

        val batchA = DataBatch(arrayOf(x1), y1)
        val batchB = DataBatch(arrayOf(x1), y1) // same tensor instances -> should be equal

        assertEquals(batchA, batchB, "Batches with same tensor instances must be equal")
        assertEquals(batchA.hashCode(), batchB.hashCode(), "Equal batches must have same hashCode")

        // Different instances with same values should NOT be equal, since Tensor equality is referential
        val x2: Tensor<FP32, Float> = tensor(ctx, FP32::class) { tensor { shape(2, 2) { from(1f, 2f, 3f, 4f) } } }
        val y2: Tensor<FP32, Float> = tensor(ctx, FP32::class) { tensor { shape(2) { from(0f, 1f) } } }
        val batchC = DataBatch(arrayOf(x2), y2)
        assertNotEquals(batchA, batchC, "Batches with same values but different tensor instances should not be equal")
    }

    @Test
    fun batchIteratorProducesCorrectSlices() {
        val feats = listOf(
            floatArrayOf(1f, 10f, 100f),
            floatArrayOf(2f, 20f, 200f),
            floatArrayOf(3f, 30f, 300f),
            floatArrayOf(4f, 40f, 400f),
            floatArrayOf(5f, 50f, 500f)
        )
        val labels = listOf(0f, 1f, 0f, 1f, 0f)
        val ds = FakeDataset(feats, labels)

        val it = ds.batchIterator<FP32, Float>(2)
        val batches = it.asSequence().toList()
        // Expect 3 batches: 2,2,1
        assertEquals(3, batches.size)
        assertEquals(2, batches[0].y.shape[0])
        assertEquals(2, batches[1].y.shape[0])
        assertEquals(1, batches[2].y.shape[0])

        // Check first batch contents
        val x0 = batches[0].x[0]
        // shape [2,3]
        assertEquals(2, x0.shape[0])
        assertEquals(3, x0.shape[1])
        // verify values
        assertEquals(1f, x0.data[0, 0])
        assertEquals(10f, x0.data[0, 1])
        assertEquals(100f, x0.data[0, 2])
        assertEquals(2f, x0.data[1, 0])
        assertEquals(20f, x0.data[1, 1])
        assertEquals(200f, x0.data[1, 2])

        // Check labels of second batch
        val y1 = batches[1].y
        assertEquals(0f + 1f, y1.data[0] + y1.data[1]) // simple aggregate check 1 + 0 in some order -> 1f
    }

    @Test
    fun splitProducesCorrectSizes() {
        val feats = (1..10).map { i -> floatArrayOf(i.toFloat(), (i * 10).toFloat()) }
        val labels = (1..10).map { i -> (i % 2).toFloat() }
        val ds = FakeDataset(feats, labels)

        val (train, test) = ds.split(0.6)
        assertEquals(6, train.xSize)
        assertEquals(4, test.xSize)

        // Check that concatenating preserves original order
        val recon = (0 until train.xSize).map { (train as FakeDataset).getX(it).toList() } +
                (0 until test.xSize).map { (test as FakeDataset).getX(it).toList() }
        val orig = feats.map { it.toList() }
        assertEquals(orig, recon)
    }

    @Test
    fun shufflePreservesElements() {
        val feats = (1..8).map { i -> floatArrayOf(i.toFloat(), (i * 10).toFloat()) }
        val labels = (1..8).map { i -> (i % 3).toFloat() }
        val ds = FakeDataset(feats, labels)
        val shuffled = ds.shuffle() as FakeDataset

        assertEquals(ds.xSize, shuffled.xSize)
        // Compare multisets of features by stringifying rows for simplicity
        val a = (0 until ds.xSize).map { ds.getX(it).joinToString(",") }.sorted()
        val b = (0 until shuffled.xSize).map { shuffled.getX(it).joinToString(",") }.sorted()
        assertEquals(a, b)
    }
}
