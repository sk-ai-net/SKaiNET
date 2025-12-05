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
    val classNames: List<String> = emptyList(),
    val regMax: Int = 16,
    /**
     * Channel multiplier used to size the backbone/neck to match ONNX weights.
     * Example: YOLOv8n exported from Ultralytics uses baseChannels=16 (width_mult=0.25),
     * while larger configs use higher numbers.
     */
    val baseChannels: Int = 32,
    /**
     * Depth multiplier controlling how many bottlenecks each C2f block uses.
     * Example: Ultralytics YOLOv8n uses ~0.33 which turns base counts [3,6,6,3] into [1,2,2,1].
     */
    val depthMultiple: Float = 1.0f
)
