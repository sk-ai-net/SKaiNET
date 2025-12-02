package sk.ainet.lang.model.dnn.yolo

/**
 * YOLO configuration used for decoding and thresholds.
 */
public data class YoloConfig(
    val numClasses: Int = 80,
    val inputSize: Int = 640,
    val confThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val maxDetections: Int = 300,
    val classNames: List<String> = emptyList()
)
