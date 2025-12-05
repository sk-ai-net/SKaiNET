package sk.ainet.io.image

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP16
import java.awt.Color
import java.awt.image.BufferedImage

public actual typealias PlatformBitmapImage = BufferedImage

public actual fun platformImageToArgb(
    image: PlatformBitmapImage,
    ctx: ExecutionContext
): Tensor<FP16, Float> {
    val width = image.width
    val height = image.height

    val argb = IntArray(width * height)
    image.getRGB(0, 0, width, height, argb, 0, width)

    val rgbChw = FloatArray(width * height * 3)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val px = argb[y * width + x]
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            val hw = y * width + x
            rgbChw[hw] = r.toFloat()
            rgbChw[1 * height * width + hw] = g.toFloat()
            rgbChw[2 * height * width + hw] = b.toFloat()
        }
    }

    return data<FP16, Float>(ctx) {
        tensor<FP16, Float> {
            shape(1, 3, height, width) {
                fromArray(rgbChw)
            }
        }
    }
}

public actual fun argbToPlatformImage(
    image: Tensor<FP16, Float>,
    ctx: ExecutionContext
): PlatformBitmapImage {
    val shape = image.data.shape
    val channels = shape[1]
    val height = shape[2]
    val width = shape[3]

    val pixels = IntArray(width * height)
    var i = 0
    for (y in 0 until height) {
        for (x in 0 until width) {
            val argb = when (channels) {
                1 -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255)
                    (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
                3 -> {
                    val r = image.data[0, 0, y, x].toInt().coerceIn(0, 255)
                    val g = image.data[0, 1, y, x].toInt().coerceIn(0, 255)
                    val b = image.data[0, 2, y, x].toInt().coerceIn(0, 255)
                    (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                else -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255)
                    (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
            }
            pixels[i++] = argb
        }
    }

    val out = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    out.setRGB(0, 0, width, height, pixels, 0, width)
    return out
}

public actual fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray {
    val w = image.width
    val h = image.height
    val argb = IntArray(w * h)
    image.getRGB(0, 0, w, h, argb, 0, w)
    val out = ByteArray(w * h * 3)
    var o = 0
    for (p in argb) {
        out[o++] = ((p shr 16) and 0xFF).toByte()
        out[o++] = ((p shr 8) and 0xFF).toByte()
        out[o++] = (p and 0xFF).toByte()
    }
    return out
}

public actual fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int> =
    image.width to image.height
