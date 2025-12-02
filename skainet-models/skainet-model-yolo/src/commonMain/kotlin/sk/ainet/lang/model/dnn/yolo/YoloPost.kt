package sk.ainet.lang.model.dnn.yolo

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

public data class Detection(
    val box: Box,
    val score: Float,
    val classId: Int,
    val label: String? = null
)

public data class Box(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    val width: Float get() = max(0f, x2 - x1)
    val height: Float get() = max(0f, y2 - y1)
}

public data class YoloInput(
    val tensor: Tensor<FP32, Float>,
    val originalWidth: Int,
    val originalHeight: Int,
    val letterboxScale: Float = 1f,
    val padW: Int = 0,
    val padH: Int = 0
)

internal object YoloDecoder {
    fun decode(
        heads: HeadOutputs,
        config: YoloConfig,
        inputMeta: YoloInput
    ): List<Detection> {
        val small = decodeHead(heads.small, stride = 8, config, inputMeta)
        val medium = decodeHead(heads.medium, stride = 16, config, inputMeta)
        val large = decodeHead(heads.large, stride = 32, config, inputMeta)
        return nms(small + medium + large, config)
    }

    private fun decodeHead(
        head: Tensor<FP32, Float>,
        stride: Int,
        config: YoloConfig,
        inputMeta: YoloInput
    ): List<Detection> {
        val (batch, channels, h, w) = head.shape.dimensions
        require(batch == 1) { "Only batch size 1 is supported for decode, got $batch" }
        val classes = config.numClasses
        require(channels == classes + 5) {
            "Head channels ($channels) must equal classes+5 (${classes + 5})"
        }
        val out = mutableListOf<Detection>()
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val base = floatArrayOf(
                    head.data.get(0, 0, y, x),
                    head.data.get(0, 1, y, x),
                    head.data.get(0, 2, y, x),
                    head.data.get(0, 3, y, x),
                    head.data.get(0, 4, y, x)
                )
                val obj = sigmoid(base[4])
                if (obj >= config.confThreshold) {
                    val clsScore = argMaxClass(head, y, x, classes)
                    val score = obj * clsScore.score
                    if (score >= config.confThreshold) {
                        val box = decodeBox(base, x, y, stride, inputMeta)
                        val label = config.classNames.getOrNull(clsScore.classId)
                        out += Detection(box, score, clsScore.classId, label)
                    }
                }
                x++
            }
            y++
        }
        return out
    }

    private data class ClassScore(val classId: Int, val score: Float)

    private fun argMaxClass(
        head: Tensor<FP32, Float>,
        y: Int,
        x: Int,
        classes: Int
    ): ClassScore {
        var bestScore = Float.NEGATIVE_INFINITY
        var bestId = -1
        var c = 0
        while (c < classes) {
            val v = head.data.get(0, 5 + c, y, x)
            if (v > bestScore) {
                bestScore = v
                bestId = c
            }
            c++
        }
        return ClassScore(bestId, sigmoid(bestScore))
    }

    private fun decodeBox(
        raw: FloatArray,
        gridX: Int,
        gridY: Int,
        stride: Int,
        meta: YoloInput
    ): Box {
        val x = (sigmoid(raw[0]) * 2f - 0.5f + gridX) * stride
        val y = (sigmoid(raw[1]) * 2f - 0.5f + gridY) * stride
        val w = (sigmoid(raw[2]) * 2f).let { it * it } * stride
        val h = (sigmoid(raw[3]) * 2f).let { it * it } * stride
        val halfW = w / 2f
        val halfH = h / 2f
        val lx = x - halfW
        val ty = y - halfH
        val rx = x + halfW
        val by = y + halfH

        // Remove letterbox padding and rescale to original dimensions
        val scale = meta.letterboxScale
        val padW = meta.padW
        val padH = meta.padH
        val mappedX1 = (lx - padW) / scale
        val mappedY1 = (ty - padH) / scale
        val mappedX2 = (rx - padW) / scale
        val mappedY2 = (by - padH) / scale

        val x1 = mappedX1.coerceIn(0f, meta.originalWidth.toFloat())
        val y1 = mappedY1.coerceIn(0f, meta.originalHeight.toFloat())
        val x2 = mappedX2.coerceIn(0f, meta.originalWidth.toFloat())
        val y2 = mappedY2.coerceIn(0f, meta.originalHeight.toFloat())
        return Box(x1, y1, x2, y2)
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x)))

    private fun nms(detections: List<Detection>, config: YoloConfig): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val keep = mutableListOf<Detection>()
        while (sorted.isNotEmpty() && keep.size < config.maxDetections) {
            val best = sorted.removeAt(0)
            keep += best
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (best.classId == other.classId) {
                    val iou = iou(best.box, other.box)
                    if (iou > config.iouThreshold) {
                        iterator.remove()
                    }
                }
            }
        }
        return keep
    }

    private fun iou(a: Box, b: Box): Float {
        val interX1 = max(a.x1, b.x1)
        val interY1 = max(a.y1, b.y1)
        val interX2 = min(a.x2, b.x2)
        val interY2 = min(a.y2, b.y2)
        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val interArea = interW * interH
        val union = a.width * a.height + b.width * b.height - interArea
        return if (union <= 0f) 0f else interArea / union
    }
}
