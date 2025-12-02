package sk.ainet.lang.model.dnn.yolo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

public actual object YoloPlatformPreprocess {
    public actual fun fromImage(image: Any, inputSize: Int, normalize: Boolean): YoloInput {
        require(image is Bitmap) { "Android preprocessing expects a Bitmap" }
        val lb = letterbox(image, inputSize)
        val tensor = bitmapToNchw(resized, normalize)
        return YoloPreprocess.fromReadyTensor(
            tensor = tensor,
            originalWidth = image.width,
            originalHeight = image.height,
            inputSize = inputSize,
            padW = lb.padW,
            padH = lb.padH,
            scale = lb.scale
        )
    }

    private fun bitmapToNchw(bmp: Bitmap, normalize: Boolean): sk.ainet.lang.tensor.Tensor<FP32, Float> {
        val width = bmp.width
        val height = bmp.height
        val pixels = IntArray(width * height)
        bmp.getPixels(pixels, 0, width, 0, 0, width, height)
        val scale = if (normalize) 1f / 255f else 1f
        return tensor<FP32, Float> {
            shape(1, 3, height, width) {
                init { idx ->
                    val c = idx[1]
                    val y = idx[2]
                    val x = idx[3]
                    val color = pixels[y * width + x]
                    val r = ((color shr 16) and 0xFF).toFloat()
                    val g = ((color shr 8) and 0xFF).toFloat()
                    val b = (color and 0xFF).toFloat()
                    when (c) {
                        0 -> r * scale
                        1 -> g * scale
                        else -> b * scale
                    }
                }
            }
        }
    }

    private data class LetterboxResult(val bitmap: Bitmap, val scale: Float, val padW: Int, val padH: Int)

    private fun letterbox(bmp: Bitmap, size: Int): LetterboxResult {
        val w = bmp.width
        val h = bmp.height
        val scale = minOf(size / w.toFloat(), size / h.toFloat())
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        val padW = (size - newW) / 2
        val padH = (size - newH) / 2

        val scaled = Bitmap.createScaledBitmap(bmp, newW, newH, true)
        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(scaled, padW.toFloat(), padH.toFloat(), null)
        return LetterboxResult(output, scale, padW, padH)
    }
}
