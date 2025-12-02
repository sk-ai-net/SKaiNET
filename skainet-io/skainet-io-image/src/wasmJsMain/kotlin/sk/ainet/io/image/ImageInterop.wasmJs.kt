package sk.ainet.io.image

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

actual class PlatformBitmapImage

actual fun platformImageToArgb(image: PlatformBitmapImage, ctx: ExecutionContext): Tensor<FP16, Float> {
    throw NotImplementedError("platformImageToArgb is not implemented for Wasm JS yet")
}

actual fun argbToPlatformImage(image: Tensor<FP16, Float>, ctx: ExecutionContext): PlatformBitmapImage {
    throw NotImplementedError("argbToPlatformImage is not implemented for Wasm JS yet")
}

actual fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray {
    throw NotImplementedError("platformImageToRgbByteArray is not implemented for Wasm JS yet")
}

actual fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int> {
    throw NotImplementedError("platformImageSize is not implemented for Wasm JS yet")
}
