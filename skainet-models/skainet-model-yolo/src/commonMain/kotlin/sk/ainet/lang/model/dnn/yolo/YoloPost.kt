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
        head: HeadTensor,
        stride: Int,
        config: YoloConfig,
        inputMeta: YoloInput
    ): List<Detection> {
        val (batch, regChannels, h, w) = head.reg.shape.dimensions
        val classes = config.numClasses
        require(batch == 1) { "Only batch size 1 is supported for decode, got $batch" }
        require(regChannels == config.regMax * 4) {
            "Reg branch has $regChannels channels, expected ${config.regMax * 4}"
        }
        val clsShape = head.cls.shape.dimensions
        require(clsShape[1] == classes) { "Cls branch has ${clsShape[1]} channels, expected $classes" }

        val out = mutableListOf<Detection>()
        val bins = FloatArray(config.regMax) { it.toFloat() }
        var yIdx = 0
        while (yIdx < h) {
            var xIdx = 0
            while (xIdx < w) {
                val dist = FloatArray(4)
                var c = 0
                while (c < 4) {
                    val start = c * config.regMax
                    val logits = FloatArray(config.regMax) { k ->
                        head.reg.data.get(0, start + k, yIdx, xIdx)
                    }
                    val probs = softmax(logits)
                    var exp = 0f
                    var k = 0
                    while (k < config.regMax) {
                        exp += probs[k] * bins[k]
                        k++
                    }
                    dist[c] = exp * stride
                    c++
                }

                val clsScore = argMaxClass(head.cls, yIdx, xIdx, classes)
                if (clsScore.score >= config.confThreshold) {
                    val box = decodeBoxFromDistances(dist, xIdx, yIdx, stride, inputMeta)
                    val label = config.classNames.getOrNull(clsScore.classId)
                    out += Detection(box, clsScore.score, clsScore.classId, label)
                }
                xIdx++
            }
            yIdx++
        }
        return out
    }

    private data class ClassScore(val classId: Int, val score: Float)

    private fun argMaxClass(
        cls: Tensor<FP32, Float>,
        y: Int,
        x: Int,
        classes: Int
    ): ClassScore {
        var bestScore = Float.NEGATIVE_INFINITY
        var bestId = -1
        var c = 0
        while (c < classes) {
            val v = cls.data.get(0, c, y, x)
            if (v > bestScore) {
                bestScore = v
                bestId = c
            }
            c++
        }
        return ClassScore(bestId, sigmoid(bestScore))
    }

    private fun decodeBoxFromDistances(
        dist: FloatArray,
        gridX: Int,
        gridY: Int,
        stride: Int,
        meta: YoloInput
    ): Box {
        val centerX = (gridX + 0.5f) * stride
        val centerY = (gridY + 0.5f) * stride
        val x1 = centerX - dist[0]
        val y1 = centerY - dist[1]
        val x2 = centerX + dist[2]
        val y2 = centerY + dist[3]

        val scale = meta.letterboxScale
        val padW = meta.padW
        val padH = meta.padH
        val mappedX1 = (x1 - padW) / scale
        val mappedY1 = (y1 - padH) / scale
        val mappedX2 = (x2 - padW) / scale
        val mappedY2 = (y2 - padH) / scale

        val x1c = mappedX1.coerceIn(0f, meta.originalWidth.toFloat())
        val y1c = mappedY1.coerceIn(0f, meta.originalHeight.toFloat())
        val x2c = mappedX2.coerceIn(0f, meta.originalWidth.toFloat())
        val y2c = mappedY2.coerceIn(0f, meta.originalHeight.toFloat())
        return Box(x1c, y1c, x2c, y2c)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var max = logits.maxOrNull() ?: 0f
        var sum = 0f
        val exp = FloatArray(logits.size)
        var i = 0
        while (i < logits.size) {
            val e = kotlin.math.exp(logits[i] - max)
            exp[i] = e
            sum += e
            i++
        }
        var j = 0
        while (j < exp.size) {
            exp[j] /= sum
            j++
        }
        return exp
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
