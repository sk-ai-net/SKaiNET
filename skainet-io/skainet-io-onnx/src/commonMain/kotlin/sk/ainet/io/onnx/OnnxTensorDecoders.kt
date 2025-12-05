package sk.ainet.io.onnx

import onnx.TensorProto
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.max

/**
 * Convert an ONNX TensorProto into an SKaiNET tensor.
 *
 * Currently supports float tensors (FLOAT). Raw data is assumed to be little-endian.
 */
public fun TensorProto.toTensor(ctx: ExecutionContext): Tensor<FP32, Float> {
    require(dataType == TensorProto.DataType.FLOAT.value) {
        "Unsupported ONNX tensor dtype=$dataType (only FLOAT is supported for now)"
    }
    val shape = Shape(*dims.map { it.toInt() }.toIntArray())
    val values: FloatArray = when {
        floatData.isNotEmpty() -> floatData.toFloatArray()
        rawData.array.isNotEmpty() -> rawData.array.toFloatArrayLE()
        else -> error("TensorProto has no data for tensor '${name.ifEmpty { "<unnamed>" }}'")
    }
    // If the tensor is shorter than the shape indicates (edge cases), pad with zeros to avoid crashes.
    val expectedSize = max(1, shape.dimensions.fold(1) { acc, d -> acc * d })
    val padded = if (values.size >= expectedSize) values else {
        FloatArray(expectedSize) { i -> values.getOrElse(i) { 0f } }
    }
    return ctx.fromFloatArray(shape, FP32::class, padded)
}

/** Helper: decode little-endian bytes into a FloatArray. */
private fun ByteArray.toFloatArrayLE(): FloatArray {
    require(size % 4 == 0) { "Raw tensor data length $size is not divisible by 4 for FLOAT" }
    val out = FloatArray(size / 4)
    var o = 0
    var i = 0
    while (i < size) {
        val bits =
            (this[i].toInt() and 0xFF) or
                ((this[i + 1].toInt() and 0xFF) shl 8) or
                ((this[i + 2].toInt() and 0xFF) shl 16) or
                ((this[i + 3].toInt() and 0xFF) shl 24)
        out[o++] = Float.fromBits(bits)
        i += 4
    }
    return out
}

/** Convenience to decode through the OnnxTensorView wrapper. */
public fun OnnxTensorView.toTensor(ctx: ExecutionContext): Tensor<FP32, Float> = proto.toTensor(ctx)
