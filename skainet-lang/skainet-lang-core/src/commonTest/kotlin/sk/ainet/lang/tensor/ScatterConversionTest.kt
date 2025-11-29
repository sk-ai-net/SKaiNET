package sk.ainet.lang.tensor

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * Unit tests demonstrating how to convert an image tensor into x, y, color lists for scatter plots.
 *
 * Python (NumPy) reference:
 *  img = np.random.rand(28, 28)
 *  y, x = np.mgrid[0:28, 0:28]
 *  x = x.ravel(); y = y.ravel(); pixels = img.ravel()
 *
 * In SKaiNET we typically have an image tensor shaped [1, H, W] (channel-first with single channel).
 * This test shows how to create such a tensor and how to convert it to x[], y[] and color[] arrays
 * in row-major order suitable for scatter plots.
 */
class ScatterConversionTest {

    @Test
    fun small_1x3x3_image_to_scatter_lists() {
        val H = 3
        val W = 3
        // Prepare deterministic 3x3 bytes [0,1,2,3,4,5,6,7,8]
        val pixelsHW = ByteArray(H * W) { i -> i.toByte() }

        val ctx = DefaultDataExecutionContext()
        val shape = Shape(1, H, W)
        val t = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, pixelsHW)

        val (xs, ys, colors) = toScatterXYColor(t)

        // Expect row-major ravel: (y,x) pairs: (0,0),(0,1),(0,2),(1,0)...
        assertContentEquals(intArrayOf(0,1,2, 0,1,2, 0,1,2), xs)
        assertContentEquals(intArrayOf(0,0,0, 1,1,1, 2,2,2), ys)

        // Colors normalized to 0..1 from bytes (unsigned)
        val expectedColors = FloatArray(H*W) { i -> (i and 0xFF) / 255f }
        assertEquals(H*W, colors.size)
        // Spot check all entries
        for (i in 0 until H*W) {
            assertEquals(expectedColors[i], colors[i])
        }
    }

    @Test
    fun mnist_like_1x28x28_image_to_scatter_lists() {
        val H = 28
        val W = 28
        // Deterministic fill: value = y*W + x (mod 256)
        val pixelsHW = ByteArray(H * W)
        var k = 0
        for (y in 0 until H) {
            for (x in 0 until W) {
                pixelsHW[k++] = ((y * W + x) % 256).toByte()
            }
        }

        val ctx = DefaultDataExecutionContext()
        val shape = Shape(1, H, W)
        val t = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, pixelsHW)

        val (xs, ys, colors) = toScatterXYColor(t)

        // Length checks
        assertEquals(H * W, xs.size)
        assertEquals(H * W, ys.size)
        assertEquals(H * W, colors.size)

        // First and last coordinates should match row-major traversal
        assertEquals(0, xs.first()); assertEquals(0, ys.first())
        assertEquals(W - 1, xs[W - 1]); assertEquals(0, ys[W - 1]) // end of first row
        assertEquals(0, xs[W]); assertEquals(1, ys[W]) // start of second row
        assertEquals(W - 1, xs.last()); assertEquals(H - 1, ys.last())

        // Spot-check a few colors: value was y*W + x scaled to 0..1
        fun expectedColor(y: Int, x: Int): Float = ((y * W + x) and 0xFF) / 255f
        assertEquals(expectedColor(0, 0), colors[0])
        assertEquals(expectedColor(0, W - 1), colors[W - 1])
        assertEquals(expectedColor(1, 0), colors[W])
        assertEquals(expectedColor(H - 1, W - 1), colors.last())
    }

    /**
     * Converts a [1, H, W] single-channel image tensor to scatter-plot arrays (x, y, color).
     * - x: IntArray of x-coordinates in [0, W)
     * - y: IntArray of y-coordinates in [0, H)
     * - color: FloatArray normalized to 0..1 derived from unsigned byte value
     */
    private fun toScatterXYColor(t: Tensor<Int8, Byte>): Triple<IntArray, IntArray, FloatArray> {
        require(t.shape.rank == 3 && t.shape.dimensions[0] == 1) { "Expected shape [1, H, W], got ${t.shape}" }
        val h = t.shape.dimensions[1]
        val w = t.shape.dimensions[2]
        val n = h * w
        val xs = IntArray(n)
        val ys = IntArray(n)
        val colors = FloatArray(n)

        var i = 0
        for (yy in 0 until h) {
            for (xx in 0 until w) {
                xs[i] = xx
                ys[i] = yy
                val b: Byte = t.data[0, yy, xx]
                val u = b.toInt() and 0xFF
                colors[i] = u / 255f
                i++
            }
        }
        return Triple(xs, ys, colors)
    }
}
