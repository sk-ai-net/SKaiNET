package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/*
/**
 * Image normalization utilities for BCHW tensors with integer pixel values.
 */
public object ImageNormalization {
    /** Default max pixel value for 8-bit images */
    public const val DEFAULT_MAX_PIXEL: Float = 255f
}

/**
 * Normalize a BCHW tensor with integer pixel values to the [0, 1] range.
 *
 * Steps:
 * - Validates BCHW rank (4D)
 * - Builds a new FP32 tensor with the same shape where each element is value/maxValue
 */
public fun Tensor<Int32, Int>.normalizePixels(maxValue: Float = ImageNormalization.DEFAULT_MAX_PIXEL): Tensor<FP32, Float> {
    require(this.rank == 4) { "Expected BCHW tensor with rank 4 (BCHW), but got rank=${this.rank} and shape=${this.shape}" }

    val src = this
    val shape = src.shape

    // Build a new FP32 tensor by mapping each Int pixel to Float/Max
    return process<FP32,Float> {
        val a= tensor<FP32, Float> {
            shape(shape) {
                init { indices ->
                    val b = indices[0]
                    val c = indices[1]
                    val h = indices[2]
                    val w = indices[3]
                    (src.data[b, c, h, w].toFloat() / maxValue)
                }
            }
        }
         return@definition a
    }
}

/**
 * Alias: Normalize Int32 BCHW tensor to unit range [0,1].
 */
public fun Tensor<Int32, Int>.normalizeIntBCHWToUnitRange(maxValue: Float = ImageNormalization.DEFAULT_MAX_PIXEL): Tensor<FP32, Float> =
    this.normalizePixels(maxValue)


 */
