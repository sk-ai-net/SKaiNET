package sk.ainet.io.core

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.*

/**
 * Adapter that bridges IO-layer TensorDescriptor/BufferView to skainet-lang TensorData.
 *
 * Notes:
 * - Current skainet-lang DenseTensorDataFactory supports Float (FP16/FP32) and Int32/Int8 via helpers.
 * - Zero-copy wrap is not available in TensorDataFactory API yet; wrap() will throw until support exists.
 */
class SkainetTensorFactoryAdapter(
    private val dataFactory: TensorDataFactory = DenseTensorDataFactory()
) : SkainetTensorFactory {

    override fun allocate(desc: TensorDescriptor): Any {
        val dtype = ioToLangDType(desc.dtype)
        val shape = Shape(desc.shape)
        // Use zeros as a neutral allocation; callers can fill using stream later.
        return when (dtype) {
            is FP32 -> dataFactory.zeros<FP32, Float>(shape, FP32::class)
            is FP16 -> dataFactory.zeros<FP16, Float>(shape, FP16::class)
            is Int32 -> dataFactory.zeros<Int32, Int>(shape, Int32::class)
            is Int8 -> dataFactory.zeros<Int8, Int>(shape, Int8::class)
            else -> throw IllegalArgumentException("Unsupported dtype for allocation: ${dtype.name}")
        }
    }

    override fun wrap(buffer: BufferView, desc: TensorDescriptor): Any {
        // Until TensorDataFactory offers wrap from raw bytes, we cannot truly zero-copy here.
        // Provide a copy-based fallback honoring dtype and endianness. If prefer true wrap, throw.
        if (!desc.isContiguous || desc.strides != null) {
            throw UnsupportedOperationException("Non-contiguous wrap is not supported by DenseTensorDataFactory")
        }
        // Copy path – read all bytes and construct appropriate TensorData
        val total = buffer.size.toInt()
        val bytes = ByteArray(total)
        buffer.readFully(bytes)
        val dtype = ioToLangDType(desc.dtype)
        val shape = Shape(desc.shape)
        return when (dtype) {
            is FP32 -> {
                // Assuming native endianness for now; endianness transform pipeline lives elsewhere per docs.
                val floats = ByteArrayToFloatArray(bytes)
                (dataFactory as DenseTensorDataFactory).fromFloatArray(shape, floats, FP32)
            }
            is FP16 -> {
                // No native FP16 buffer path; upcast to Float for Dense backend storage.
                val floats = FP16BytesToFloatArray(bytes)
                (dataFactory as DenseTensorDataFactory).fromFloatArray(shape, floats, FP16)
            }
            is Int32 -> {
                val ints = ByteArrayToIntArray(bytes)
                DenseTensorDataFactory().let { factory ->
                    // DenseTensorDataFactory has createIntTensorData via public API paths using TensorFactory methods
                    // Use full() + manual fill for minimal dependency on internal APIs
                    val td = factory.full<Int32>(shape, 0, Int32)
                    val buf = (td as? sk.ainet.lang.tensor.data.IntArrayTensorData<Int32>)?.buffer
                        ?: throw IllegalStateException("Expected IntArrayTensorData backing")
                    if (ints.size != buf.size) throw IllegalArgumentException("Element count mismatch for Int32: ${ints.size} != ${buf.size}")
                    ints.copyInto(destination = buf, startIndex = 0, endIndex = buf.size)
                    td
                }
            }
            is Int8 -> {
                // Dense backend models Int8 via Int array; convert bytes to ints
                val ints = bytes.map { it.toInt() }.toIntArray()
                DenseTensorDataFactory().let { factory ->
                    val td = factory.full<Int8>(shape, 0, Int8)
                    val buf = (td as? sk.ainet.lang.tensor.data.IntArrayTensorData<Int8>)?.buffer
                        ?: throw IllegalStateException("Expected IntArrayTensorData backing for Int8")
                    if (ints.size != buf.size) throw IllegalArgumentException("Element count mismatch for Int8: ${ints.size} != ${buf.size}")
                    ints.copyInto(destination = buf, startIndex = 0, endIndex = buf.size)
                    td
                }
            }
            else -> throw IllegalArgumentException("Unsupported dtype for wrap: ${dtype.name}")
        }
    }
}

// Decide whether to attempt wrap (zero/min-copy) vs copy based on descriptor and options.
fun shouldWrap(desc: TensorDescriptor, opts: MaterializeOptions, hasBufferView: Boolean): Boolean {
    if (!hasBufferView) return false
    if (!opts.preferZeroCopy) return false
    if (!desc.isContiguous) return false
    // if endianness mismatch and no transform configured, force copy
    val byteOrder = opts.byteOrderOverride ?: desc.endianness
    if (byteOrder != ByteOrder.NATIVE) return false
    return true
}

// Mapping utils between IO IoDType and skainet-lang DType
fun ioToLangDType(io: IoDType): DType = when (io) {
    IoDType.F32 -> FP32
    IoDType.F16 -> FP16
    IoDType.I32 -> Int32
    IoDType.I8 -> Int8
    else -> throw IllegalArgumentException("Unsupported IoDType in adapter: $io")
}

// Size of a single element in bytes for IO dtype
fun ioElementSize(io: IoDType): Int = when (io) {
    IoDType.F64, IoDType.I64, IoDType.U64 -> 8
    IoDType.F32, IoDType.I32, IoDType.U32 -> 4
    IoDType.F16, IoDType.BF16, IoDType.I16, IoDType.U16 -> 2
    IoDType.I8, IoDType.U8, IoDType.BOOL -> 1
}

// Utilities to convert raw bytes into typed arrays. Endianness handling is simplified; real pipeline should handle ByteOrder.
private fun ByteArrayToFloatArray(bytes: ByteArray): FloatArray {
    require(bytes.size % 4 == 0) { "Byte size not multiple of 4 for FP32: ${bytes.size}" }
    val out = FloatArray(bytes.size / 4)
    var i = 0; var j = 0
    while (i < bytes.size) {
        val b0 = bytes[i].toInt() and 0xFF
        val b1 = bytes[i + 1].toInt() and 0xFF
        val b2 = bytes[i + 2].toInt() and 0xFF
        val b3 = bytes[i + 3].toInt() and 0xFF
        val intBits = (b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0
        out[j++] = Float.fromBits(intBits)
        i += 4
    }
    return out
}

private fun FP16BytesToFloatArray(bytes: ByteArray): FloatArray {
    require(bytes.size % 2 == 0) { "Byte size not multiple of 2 for FP16: ${bytes.size}" }
    val out = FloatArray(bytes.size / 2)
    var i = 0; var j = 0
    while (i < bytes.size) {
        val b0 = bytes[i].toInt() and 0xFF
        val b1 = bytes[i + 1].toInt() and 0xFF
        val half = (b1 shl 8) or b0
        out[j++] = fp16ToFloat(half)
        i += 2
    }
    return out
}

private fun ByteArrayToIntArray(bytes: ByteArray): IntArray {
    require(bytes.size % 4 == 0) { "Byte size not multiple of 4 for Int32: ${bytes.size}" }
    val out = IntArray(bytes.size / 4)
    var i = 0; var j = 0
    while (i < bytes.size) {
        val b0 = bytes[i].toInt() and 0xFF
        val b1 = bytes[i + 1].toInt() and 0xFF
        val b2 = bytes[i + 2].toInt() and 0xFF
        val b3 = bytes[i + 3].toInt() and 0xFF
        val intVal = (b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0
        out[j++] = intVal
        i += 4
    }
    return out
}

// Half-precision to float conversion (IEEE 754)
private fun fp16ToFloat(h: Int): Float {
    val s = (h ushr 15) and 0x00000001
    var e = (h ushr 10) and 0x0000001f
    var f = h and 0x000003ff

    if (e == 0) {
        if (f == 0) {
            return Float.fromBits(s shl 31)
        } else {
            while ((f and 0x00000400) == 0) {
                f = f shl 1
                e -= 1
            }
            e += 1
            f = f and 0xFFFFFBFF.toInt()
        }
    } else if (e == 31) {
        if (f == 0) {
            return Float.fromBits((s shl 31) or 0x7f800000)
        } else {
            return Float.fromBits((s shl 31) or 0x7f800000 or (f shl 13))
        }
    }

    e = e + (127 - 15)
    f = f shl 13
    return Float.fromBits((s shl 31) or (e shl 23) or f)
}
