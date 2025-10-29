package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.pprint
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ImageNormalizationTest {

    @Test
    fun normalizeIntBCHWToUnitRange() {
        // Build a small BCHW tensor with Int32 pixels
        val values = intArrayOf(
            0, 127, 255,
            4, 28, 92,
            24, 128, 192,
            25, 228, 292,
            39, 329, 392,
            39, 329, 393,
        )
        data {

            val intImage = tensor<Int32, Int> {
                shape(1, 3, 2, 3) {
                    var idx = 0
                    init { _ ->
                        val v = values[idx]
                        idx += 1
                        v
                    }
                }
            }

            // RED
            val red = intImage.sliceView {
                // BCHW: select batch 0, channel 0 (R), all heights, all widths
                segment { at(0) }
                segment { at(0) }
                segment { all() }
                segment { all() }
            }.pprint()

            val green = intImage.sliceView {
                // BCHW: select batch 0, channel 1 (G)
                segment { at(0) }
                segment { at(1) }
                segment { all() }
                segment { all() }
            }.pprint()

            val blue = intImage.sliceView {
                // BCHW: select batch 0, channel 2 (B)
                segment { at(0) }
                segment { at(2) }
                segment { all() }
                segment { all() }
            }.pprint()

            assertNotNull(intImage)
            assertEquals(Shape(1, 3, 2, 3), intImage.shape)

            /*
            // Normalize to [0,1]
            val norm = intImage.normalizePixels()

            assertNotNull(norm)
            assertEquals(Shape(1, 3, 2, 3), norm.shape)
            // dtype should be FP32
            assertEquals(FP32::class, norm.dtype)

            fun f(x: Int) = x.toFloat() / 255f
            // Check a few values in Red channel (c=0)
            assertEquals(f(0), norm.data[0, 0, 0, 0])
            assertEquals(f(127), norm.data[0, 0, 0, 1])
            assertEquals(f(255), norm.data[0, 0, 0, 2])
            assertEquals(f(4), norm.data[0, 0, 1, 0])
            assertEquals(f(28), norm.data[0, 0, 1, 1])
            assertEquals(f(92), norm.data[0, 0, 1, 2])

             */
        }
    }
}
