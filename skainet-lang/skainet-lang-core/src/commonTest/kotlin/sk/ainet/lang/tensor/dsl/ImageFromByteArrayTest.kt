package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Verifies creating CHW RGB tensor from a byte array that mimics Android Bitmap ARGB_8888 -> RGB extraction.
 * We fake toRgbByteArray() to return channel-first RGB planes: [R...(HxW), G...(HxW), B...(HxW)].
 */
class ImageFromByteArrayTest {

    private fun makeRgbBytesCHW(width: Int, height: Int, rgbPerPixel: Array<IntArray>): ByteArray {
        // rgbPerPixel holds rows of pixels; each pixel encoded as R,G,B in 0..255
        // We will produce bytes in CHW order: R-plane (row-major HxW), then G-plane, then B-plane
        require(rgbPerPixel.size == height)
        rgbPerPixel.forEach { row -> require(row.size == width * 3) }

        val planeSize = width * height
        val out = ByteArray(planeSize * 3)

        // Fill R plane
        var idx = 0
        for (y in 0 until height) {
            val row = rgbPerPixel[y]
            for (x in 0 until width) {
                val r = row[x * 3]
                out[idx++] = r.toByte()
            }
        }
        // Fill G plane
        for (y in 0 until height) {
            val row = rgbPerPixel[y]
            for (x in 0 until width) {
                val g = row[x * 3 + 1]
                out[idx++] = g.toByte()
            }
        }
        // Fill B plane
        for (y in 0 until height) {
            val row = rgbPerPixel[y]
            for (x in 0 until width) {
                val b = row[x * 3 + 2]
                out[idx++] = b.toByte()
            }
        }
        return out
    }

    @Test
    fun `tensor from CHW rgb bytes has expected shape and values`() {
        val w = 2
        val h = 2
        // Pixels row-major: [(10,20,30),(40,50,60)] ; [(70,80,90),(100,110,120)]
        val rgb = arrayOf(
            intArrayOf(10, 20, 30,   40, 50, 60),
            intArrayOf(70, 80, 90,   100, 110, 120)
        )
        val bytes = makeRgbBytesCHW(w, h, rgb)

        val tensor = data<FP32, Float> {
            tensor {
                // CHW shape: (C,H,W)
                shape(3, h, w) {
                    fromArray(bytes.map { (it.toInt() and 0xFF).toFloat() }.toFloatArray())
                }
            }
        }

        // Shape and volume
        assertEquals(Shape(3, h, w), tensor.shape)
        assertEquals(3 * w * h, tensor.volume)

        // Channel 0 (R)
        assertEquals(10f, tensor.data[0, 0, 0])
        assertEquals(40f, tensor.data[0, 0, 1])
        assertEquals(70f, tensor.data[0, 1, 0])
        assertEquals(100f, tensor.data[0, 1, 1])

        // Channel 1 (G)
        assertEquals(20f, tensor.data[1, 0, 0])
        assertEquals(50f, tensor.data[1, 0, 1])
        assertEquals(80f, tensor.data[1, 1, 0])
        assertEquals(110f, tensor.data[1, 1, 1])

        // Channel 2 (B)
        assertEquals(30f, tensor.data[2, 0, 0])
        assertEquals(60f, tensor.data[2, 0, 1])
        assertEquals(90f, tensor.data[2, 1, 0])
        assertEquals(120f, tensor.data[2, 1, 1])
    }
}
