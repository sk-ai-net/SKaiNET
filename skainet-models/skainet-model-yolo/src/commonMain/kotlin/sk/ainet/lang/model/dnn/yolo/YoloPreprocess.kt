package sk.ainet.lang.model.dnn.yolo

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Preprocessing helpers for YOLOv8 inputs.
 *
 * For now we assume the caller supplies an already resized and normalized NCHW tensor
 * matching [YoloConfig.inputSize]. This keeps dependencies minimal and avoids
 * platform-specific image handling. Letterbox metadata is carried to map boxes
 * back to the original image size.
 */
public object YoloPreprocess {

    /**
     * Wraps a tensor that is already in the expected model shape [1, 3, inputSize, inputSize].
     *
     * @param tensor normalized tensor (values typically in [0,1])
     * @param originalWidth width of the source image
     * @param originalHeight height of the source image
     * @param inputSize target model resolution (defaults to config.inputSize)
     * @param padW horizontal padding applied during letterbox (pixels)
     * @param padH vertical padding applied during letterbox (pixels)
     * @param scale scaling factor used during letterbox (original -> model space)
     */
    public fun fromReadyTensor(
        tensor: Tensor<FP32, Float>,
        originalWidth: Int,
        originalHeight: Int,
        inputSize: Int,
        padW: Int = 0,
        padH: Int = 0,
        scale: Float = 1f
    ): YoloInput {
        require(tensor.shape.rank == 4) { "Expected NCHW tensor rank 4, got ${tensor.shape.rank}" }
        require(tensor.shape[0] == 1 && tensor.shape[1] == 3) {
            "Expected batch=1 and channels=3, got shape ${tensor.shape.dimensions.contentToString()}"
        }
        require(tensor.shape[2] == inputSize && tensor.shape[3] == inputSize) {
            "Tensor must already be resized to $inputSize, got ${tensor.shape[2]}x${tensor.shape[3]}"
        }
        return YoloInput(
            tensor = tensor,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            letterboxScale = scale,
            padW = padW,
            padH = padH
        )
    }
}
