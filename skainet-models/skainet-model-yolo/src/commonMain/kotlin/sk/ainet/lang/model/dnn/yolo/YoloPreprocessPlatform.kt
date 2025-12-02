package sk.ainet.lang.model.dnn.yolo

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Platform hooks for image-to-tensor conversion. Implementations should perform:
 * 1) resize+letterbox to [inputSize, inputSize] preserving aspect ratio
 * 2) convert to NCHW FP32 normalized to [0,1]
 * 3) return [YoloInput] with padding/scale metadata
 */
public expect object YoloPlatformPreprocess {
    public fun fromImage(
        image: Any,
        inputSize: Int = 640,
        normalize: Boolean = true
    ): YoloInput
}

/**
 * Reference pure tensor-based helper for already prepared raw pixel tensors (NCHW uint8 or float in [0,255]).
 */
public object YoloTensorPreprocess {
    public fun fromUint8(
        tensor: Tensor<FP32, Float>,
        originalWidth: Int,
        originalHeight: Int,
        inputSize: Int,
        padW: Int = 0,
        padH: Int = 0,
        scale: Float = 1f,
    ): YoloInput {
        // Assume tensor already holds pixel values; caller controls normalization.
        return YoloPreprocess.fromReadyTensor(tensor, originalWidth, originalHeight, inputSize, padW, padH, scale)
    }
}
