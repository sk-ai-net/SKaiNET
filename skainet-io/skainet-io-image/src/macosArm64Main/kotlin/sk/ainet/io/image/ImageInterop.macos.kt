@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package sk.ainet.io.image

import kotlinx.cinterop.*
import platform.AppKit.NSImage
import platform.CoreFoundation.CFDataCreate
import platform.CoreFoundation.kCFAllocatorDefault
import platform.CoreGraphics.*
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGSize
import platform.CoreGraphics.CGSizeMake
import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP16

actual typealias PlatformBitmapImage = NSImage

actual fun platformImageToArgb(image: PlatformBitmapImage, ctx: ExecutionContext): Tensor<FP16, Float> =
    data<FP16, Float>(ctx) {
        // Draw into RGBA8 buffer and read back as RGB
        val (w, h) = platformImageSize(image)
        val rgba = drawImageIntoRgbaBuffer(image, w, h)

        // Convert interleaved RGB bytes to CHW floats
        val rgbChw = FloatArray(w * h * 3)
        var p = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val hw = y * w + x
                val r = rgba[p].toUByte().toInt(); p++
                val g = rgba[p].toUByte().toInt(); p++
                val b = rgba[p].toUByte().toInt(); p++
                p++ // skip alpha
                rgbChw[hw] = r.toFloat()
                rgbChw[1 * h * w + hw] = g.toFloat()
                rgbChw[2 * h * w + hw] = b.toFloat()
            }
        }

        tensor<FP16, Float> {
            shape(1, 3, h, w) {
                fromArray(rgbChw)
            }
        }
    }

actual fun argbToPlatformImage(image: Tensor<FP16, Float>, ctx: ExecutionContext): PlatformBitmapImage {
    val shape = image.data.shape
    val channels = shape[1]
    val height = shape[2]
    val width = shape[3]

    // Prepare RGBA bytes
    val bytes = ByteArray(width * height * 4)
    var o = 0
    for (y in 0 until height) {
        for (x in 0 until width) {
            when (channels) {
                1 -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    bytes[o++] = v // R
                    bytes[o++] = v // G
                    bytes[o++] = v // B
                    bytes[o++] = (-1).toByte() // A = 255
                }
                3 -> {
                    val r = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    val g = image.data[0, 1, y, x].toInt().coerceIn(0, 255).toByte()
                    val b = image.data[0, 2, y, x].toInt().coerceIn(0, 255).toByte()
                    bytes[o++] = r
                    bytes[o++] = g
                    bytes[o++] = b
                    bytes[o++] = (-1).toByte() // A = 255
                }
                else -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    bytes[o++] = v
                    bytes[o++] = v
                    bytes[o++] = v
                    bytes[o++] = (-1).toByte()
                }
            }
        }
    }

    // Create CGImage from RGBA bytes and wrap into NSImage
    val cgImage = createCgImageFromRgba(bytes, width, height)
    val size: CValue<CGSize> = CGSizeMake(width.toDouble(), height.toDouble())
    return NSImage(cGImage = cgImage, size = size)
}

actual fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray {
    val (w, h) = platformImageSize(image)
    val rgba = drawImageIntoRgbaBuffer(image, w, h)
    val out = ByteArray(w * h * 3)
    var i = 0
    var p = 0
    while (p < rgba.size) {
        out[i++] = rgba[p]     // R
        out[i++] = rgba[p + 1] // G
        out[i++] = rgba[p + 2] // B
        p += 4 // skip A
    }
    return out
}

actual fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int> {
    val cg = image.CGImageForProposedRect(null, null, null)
    return if (cg != null) {
        CGImageGetWidth(cg).toInt() to CGImageGetHeight(cg).toInt()
    } else {
        // Fallback to point size if no CGImage representation (values in points)
        val sz = image.size
        val w = sz.useContents { width }
        val h = sz.useContents { height }
        w.toInt() to h.toInt()
    }
}

// Helpers
private fun drawImageIntoRgbaBuffer(image: NSImage, width: Int, height: Int): ByteArray = memScoped {
    val cg = image.CGImageForProposedRect(null, null, null)
        ?: error("NSImage has no CGImage representation")

    val colorSpace = CGColorSpaceCreateDeviceRGB()
        ?: error("Failed to create RGB color space")
    val bytesPerRow = width * 4
    val bitmapInfo: UInt = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value

    val buffer = ByteArray(width * height * 4)
    buffer.usePinned { pinned ->
        val ctx = CGBitmapContextCreate(
            pinned.addressOf(0),
            width.convert(),
            height.convert(),
            8.convert(),
            bytesPerRow.convert(),
            colorSpace,
            bitmapInfo
        ) ?: error("Failed to create bitmap context")

        // Flip the context vertically to match typical top-left origin drawing
        val rect = CGRectMake(0.0, 0.0, width.toDouble(), height.toDouble())
        // Draw without additional transforms; NSImage/CGImage coordinates are bottom-left origin.
        CGContextDrawImage(ctx, rect, cg)
    }
    buffer
}

private fun createCgImageFromRgba(bytes: ByteArray, width: Int, height: Int): CGImageRef = memScoped {
    val colorSpace = CGColorSpaceCreateDeviceRGB()
        ?: error("Failed to create RGB color space")
    val bytesPerRow = width * 4

    val provider = bytes.usePinned { pinned ->
        val cfData = CFDataCreate(
            kCFAllocatorDefault,
            pinned.addressOf(0).reinterpret(),
            bytes.size.convert()
        ) ?: error("Failed to create CFData")
        CGDataProviderCreateWithCFData(cfData)
            ?: error("Failed to create CGDataProvider")
    }

    val bitmapInfo: UInt = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value

    CGImageCreate(
        width.convert(),
        height.convert(),
        8.convert(),
        32.convert(),
        bytesPerRow.convert(),
        colorSpace,
        bitmapInfo,
        provider,
        null,
        true,
        CGColorRenderingIntent.kCGRenderingIntentDefault
    ) ?: error("Failed to create CGImage")
}
