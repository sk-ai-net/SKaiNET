package sk.ainet.io.image

import android.graphics.Bitmap
import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.*
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16
import android.graphics.Bitmap.Config

public actual typealias PlatformBitmapImage = Bitmap

/**
 * Converts this Bitmap to a packed RGB (R,G,B) ByteArray.
 * Output order: row-major (top→bottom, left→right), 3 bytes per pixel.
 * Works for any input config; internally uses ARGB_8888 for correctness.
 */
private fun PlatformBitmapImage.toRgbByteArray(): ByteArray {
    // Ensure ARGB_8888 so channel extraction is predictable
    val src = if (config != Bitmap.Config.ARGB_8888) {
        copy(Bitmap.Config.ARGB_8888, /* isMutable = */false)
    } else this

    val w = src.width
    val h = src.height

    val pixels = IntArray(w * h)
    src.getPixels(pixels, 0, w, 0, 0, w, h)

    val rgb = ByteArray(w * h * 3)
    var o = 0
    // Each pixel is Android ARGB: 0xAARRGGBB
    for (p in pixels) {
        rgb[o++] = ((p shr 16) and 0xFF).toByte()  // R
        rgb[o++] = ((p shr 8) and 0xFF).toByte()  // G
        rgb[o++] = (p and 0xFF).toByte()  // B
    }
    // If we created a copy, we can let GC reclaim it; 'src' will be collected
    return rgb
}

public actual fun platformImageToArgb(
    image: PlatformBitmapImage,
    ctx: ExecutionContext
): Tensor<FP16, Float> {
    val src = if (image.config != Bitmap.Config.ARGB_8888) {
        image.copy(Bitmap.Config.ARGB_8888, false)
    } else image

    val width = src.width
    val height = src.height

    // 1) Pull pixels as ARGB_8888
    val argb = IntArray(width * height)
    src.getPixels(argb, 0, width, /*x=*/0, /*y=*/0, width, height)

    // 2) Convert to FloatArray in RGB order, CHW layout to match (1,3,H,W)
    val rgbChw = FloatArray(width * height * 3)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val px = argb[y * width + x]
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            val hwIndex = y * width + x
            rgbChw[hwIndex] = r.toFloat()
            rgbChw[1 * height * width + hwIndex] = g.toFloat()
            rgbChw[2 * height * width + hwIndex] = b.toFloat()
        }
    }

    // 3) Create a tensor with shape (1, C, H, W)
    return data<FP16, Float>(ctx) {
        tensor<FP16, Float> {
            shape(1, 3, height, width) {
                fromArray(rgbChw)
            }
        }
    }
}

public actual fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray {
    val src = if (image.config != Bitmap.Config.ARGB_8888) {
        image.copy(Bitmap.Config.ARGB_8888, /*mutable=*/false)
    } else image
    return src.toRgbByteArray()
}

public actual fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int> =
    image.width to image.height

public actual fun argbToPlatformImage(
    image: Tensor<FP16, Float>,
    ctx: ExecutionContext
): PlatformBitmapImage {
    // Tensor is NCHW according to this module's convention used in platformImageToArgb
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

    val bmp = Bitmap.createBitmap(width, height, Config.ARGB_8888)
    bmp.setPixels(pixels, 0, width, 0, 0, width, height)
    return bmp
}
