package sk.ainet.lang.model.dnn.yolo

import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGContextRef
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGImageRef
import platform.CoreGraphics.CGRectMake
import platform.UIKit.UIImage
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

public actual object YoloPlatformPreprocess {
    public actual fun fromImage(image: Any, inputSize: Int, normalize: Boolean): YoloInput {
        require(image is UIImage) { "iOS preprocessing expects a UIImage" }
        val (cgImage, width, height) = requireNotNull(image.CGImage) { "UIImage must have CGImage backing" }.let { Triple(it, it.width.toInt(), it.height.toInt()) }

        // Simple resize preserving aspect ratio with padding
        val scale = minOf(inputSize / width.toFloat(), inputSize / height.toFloat())
        val newW = (width * scale).toInt()
        val newH = (height * scale).toInt()
        val padW = (inputSize - newW) / 2
        val padH = (inputSize - newH) / 2

        val colorSpace = CGColorSpaceCreateDeviceRGB()
        val bytesPerPixel = 4
        val bytesPerRow = newW * bytesPerPixel
        val buffer = kotlin.experimental.ExperimentalNativeApi.run { kotlin.native.allocArray<ByteVar>(newH * bytesPerRow) }
        val context: CGContextRef = platform.CoreGraphics.CGBitmapContextCreate(
            data = buffer,
            width = newW.toULong(),
            height = newH.toULong(),
            bitsPerComponent = 8u,
            bytesPerRow = bytesPerRow.toULong(),
            space = colorSpace,
            bitmapInfo = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value
        ) ?: error("Failed to create CGContext")

        // Draw scaled image into buffer
        val rect = CGRectMake(0.0, 0.0, newW.toDouble(), newH.toDouble())
        platform.CoreGraphics.CGContextDrawImage(context, rect, cgImage)

        val tensor = tensor<FP32, Float> {
            shape(1, 3, inputSize, inputSize) {
                init { idx ->
                    val c = idx[1]
                    val y = idx[2]
                    val x = idx[3]
                    val isPadX = x < padW || x >= padW + newW
                    val isPadY = y < padH || y >= padH + newH
                    if (isPadX || isPadY) return@init 0f
                    val srcX = x - padW
                    val srcY = y - padH
                    val offset = (srcY * bytesPerRow + srcX * bytesPerPixel)
                    val r = buffer[offset].toUByte().toFloat()
                    val g = buffer[offset + 1].toUByte().toFloat()
                    val b = buffer[offset + 2].toUByte().toFloat()
                    val scaleVal = if (normalize) 1f / 255f else 1f
                    when (c) {
                        0 -> r * scaleVal
                        1 -> g * scaleVal
                        else -> b * scaleVal
                    }
                }
            }
        }

        return YoloPreprocess.fromReadyTensor(
            tensor = tensor,
            originalWidth = width,
            originalHeight = height,
            inputSize = inputSize,
            padW = padW,
            padH = padH,
            scale = scale
        )
    }
}
