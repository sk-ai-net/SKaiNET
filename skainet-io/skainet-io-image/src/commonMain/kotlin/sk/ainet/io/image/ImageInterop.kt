package sk.ainet.io.image

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

expect class PlatformBitmapImage

// Convert Platform image to an RGB tensor with shape (1, 3, H, W)
expect fun platformImageToArgb(image: PlatformBitmapImage, ctx: ExecutionContext): Tensor<FP16, Float>

// Convert RGB tensor (1, 3/1, H, W) back to platform image
expect fun argbToPlatformImage(image: Tensor<FP16, Float>, ctx: ExecutionContext): PlatformBitmapImage

// RGB bytes in row-major order, 3 bytes per pixel
expect fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray

// Returns width to Pair.first and height to Pair.second
expect fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int>