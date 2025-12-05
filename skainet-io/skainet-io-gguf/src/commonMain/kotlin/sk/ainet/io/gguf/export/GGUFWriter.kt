package sk.ainet.io.gguf.export

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUF_DEFAULT_ALIGNMENT
import sk.ainet.io.gguf.GGUF_MAGIC
import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.GGUF_VERSION
import sk.ainet.io.gguf.GGUFValueType
import sk.ainet.lang.tensor.Tensor
import kotlin.math.max

/** Options for writing GGUF bytes. */
public data class GGUFWriteOptions(
    val alignment: Int = GGUF_DEFAULT_ALIGNMENT
)

/** Result summary for a GGUF write. */
public data class GGUFWriteReport(
    val bytesWritten: Long,
    val tensorCount: Int,
    val kvCount: Int
)

/**
 * Minimal GGUF writer that consumes a [GgufWriteRequest] and emits GGUF v3 bytes.
 * Scope: supports scalar KV types (string/int/long/float/bool) and flat arrays of those scalars;
 * tensor payloads support FP32, Int8, and Int32. Other quantization types are rejected for now.
 */
public object GGUFWriter {

    /** Write into a byte array (most portable). */
    public fun writeToByteArray(
        request: GgufWriteRequest,
        options: GGUFWriteOptions = GGUFWriteOptions()
    ): Pair<GGUFWriteReport, ByteArray> {
        val writer = ByteWriter()
        val kvEntries = request.metadata.entries.sortedBy { it.key }.map { it.key to it.value }
        val tensorEntries = request.tensors.sortedBy { it.ggufName }

        // Pre-materialize tensor payloads and sizes for offsets.
        val payloads = tensorEntries.map { entry ->
            val bytes = materializeTensor(entry)
            val expectedSize = expectedTensorSize(entry, bytes.size)
            require(expectedSize == bytes.size) {
                "Tensor ${entry.ggufName} size mismatch: expected $expectedSize, got ${bytes.size}"
            }
            bytes
        }

        val offsets = computeOffsets(payloads.map { it.size }, options.alignment)

        // Header
        writer.writeUInt32(GGUF_MAGIC.toUInt())
        writer.writeUInt32(GGUF_VERSION.toUInt())
        writer.writeUInt64(tensorEntries.size.toULong())
        writer.writeUInt64(kvEntries.size.toULong())

        // KV section
        kvEntries.forEach { (key, value) ->
            writeKeyValue(writer, key, value)
        }

        // Tensor info section
        tensorEntries.forEachIndexed { idx, entry ->
            val offs = offsets[idx]
            writeTensorInfo(writer, entry, offs.toULong())
        }

        // Align to data section
        writer.align(options.alignment)
        val dataStart = writer.size()

        // Tensor payloads
        payloads.forEachIndexed { idx, bytes ->
            val desired = offsets[idx]
            val actual = writer.size() - dataStart
            if (actual < desired) {
                writer.pad(desired - actual)
            } else require(actual == desired) {
                "Tensor ${tensorEntries[idx].ggufName} offset mismatch: expected $desired, actual $actual"
            }
            writer.writeBytes(bytes)
        }

        val bytes = writer.toByteArray()
        val report = GGUFWriteReport(
            bytesWritten = bytes.size.toLong(),
            tensorCount = tensorEntries.size,
            kvCount = kvEntries.size
        )
        return report to bytes
    }

    // --- Helpers ---

    private fun writeKeyValue(writer: ByteWriter, key: String, value: Any) {
        val keyBytes = key.encodeToByteArray()
        writer.writeUInt64(keyBytes.size.toULong())
        writer.writeBytes(keyBytes)

        val (type, payloadWriter) = encodeValue(value)
        writer.writeUInt32(type.value.toUInt())
        payloadWriter(writer)
    }

    private fun encodeValue(value: Any): Pair<GGUFValueType, (ByteWriter) -> Unit> = when (value) {
        is String -> GGUFValueType.STRING to { w ->
            val bytes = value.encodeToByteArray()
            w.writeUInt64(bytes.size.toULong())
            w.writeBytes(bytes)
        }
        is Boolean -> GGUFValueType.BOOL to { w -> w.writeUInt8(if (value) 1 else 0) }
        is Int -> GGUFValueType.INT32 to { w -> w.writeInt32(value) }
        is Long -> GGUFValueType.INT64 to { w -> w.writeInt64(value) }
        is Float -> GGUFValueType.FLOAT32 to { w -> w.writeFloat32(value) }
        is Double -> GGUFValueType.FLOAT64 to { w -> w.writeFloat64(value) }
        is UInt -> GGUFValueType.UINT32 to { w -> w.writeUInt32(value) }
        is ULong -> GGUFValueType.UINT64 to { w -> w.writeUInt64(value) }
        is List<*> -> encodeArray(value)
        else -> error("Unsupported KV type for value '$value' (${value::class})")
    }

    private fun encodeArray(values: List<*>): Pair<GGUFValueType, (ByteWriter) -> Unit> {
        require(values.isNotEmpty()) { "GGUF arrays must be non-empty" }
        val first = values.first() ?: error("GGUF arrays cannot contain null elements")
        val (elemType, elemWriterFactory) = encodeScalar(first)
        // Validate homogeneity
        values.forEachIndexed { idx, v ->
            val e = v ?: error("GGUF arrays cannot contain null elements (index $idx)")
            val (t, _) = encodeScalar(e)
            require(t == elemType) { "Heterogeneous arrays not supported: $elemType vs $t" }
        }
        return GGUFValueType.ARRAY to { w ->
            w.writeUInt32(elemType.value.toUInt())
            w.writeUInt64(values.size.toULong())
            values.forEach { v ->
                elemWriterFactory(v!!, w)
            }
        }
    }

    private fun encodeScalar(value: Any): Pair<GGUFValueType, (Any, ByteWriter) -> Unit> = when (value) {
        is String -> GGUFValueType.STRING to { v, w ->
            val bytes = (v as String).encodeToByteArray()
            w.writeUInt64(bytes.size.toULong()); w.writeBytes(bytes)
        }
        is Boolean -> GGUFValueType.BOOL to { v, w -> w.writeUInt8(if (v as Boolean) 1 else 0) }
        is Int -> GGUFValueType.INT32 to { v, w -> w.writeInt32(v as Int) }
        is Long -> GGUFValueType.INT64 to { v, w -> w.writeInt64(v as Long) }
        is Float -> GGUFValueType.FLOAT32 to { v, w -> w.writeFloat32(v as Float) }
        is Double -> GGUFValueType.FLOAT64 to { v, w -> w.writeFloat64(v as Double) }
        is UInt -> GGUFValueType.UINT32 to { v, w -> w.writeUInt32(v as UInt) }
        is ULong -> GGUFValueType.UINT64 to { v, w -> w.writeUInt64(v as ULong) }
        else -> error("Unsupported array element type: ${value::class}")
    }

    private fun writeTensorInfo(writer: ByteWriter, entry: GgufTensorEntry, dataOffset: ULong) {
        val nameBytes = entry.ggufName.encodeToByteArray()
        writer.writeUInt64(nameBytes.size.toULong())
        writer.writeBytes(nameBytes)

        val dims = entry.shape
        writer.writeUInt32(dims.size.toUInt())
        dims.forEach { d ->
            require(d >= 0) { "Negative dimension in tensor ${entry.ggufName}" }
            writer.writeUInt64(d.toULong())
        }

        writer.writeUInt32(entry.quantization.value.toUInt())
        writer.writeUInt64(dataOffset)
    }

    private fun materializeTensor(entry: GgufTensorEntry): ByteArray {
        return when (entry.quantization) {
            GGMLQuantizationType.F32 -> materializeF32(entry)
            GGMLQuantizationType.I8 -> materializeI8(entry)
            GGMLQuantizationType.I32 -> materializeI32(entry)
            else -> error("Quantization ${entry.quantization} not supported for writing yet")
        }
    }

    private fun materializeF32(entry: GgufTensorEntry): ByteArray {
        val floatData = TensorFlatten.flattenFloats(entry.tensor)
        val out = ByteWriter()
        floatData.forEach { out.writeFloat32(it) }
        return out.toByteArray()
    }

    private fun materializeI8(entry: GgufTensorEntry): ByteArray {
        val bytes = TensorFlatten.flattenBytes(entry.tensor)
        return bytes
    }

    private fun materializeI32(entry: GgufTensorEntry): ByteArray {
        val ints = TensorFlatten.flattenInts(entry.tensor)
        val out = ByteWriter()
        ints.forEach { out.writeInt32(it) }
        return out.toByteArray()
    }

    private fun expectedTensorSize(entry: GgufTensorEntry, actualBytes: Int): Int {
        val (blockSize, typeSize) = GGML_QUANT_SIZES[entry.quantization]
            ?: error("Quantization ${entry.quantization} missing size metadata")
        val volume = entry.shape.fold(1L) { acc, d -> acc * max(1, d).toLong() }
        val expected = (volume * typeSize) / blockSize
        require(expected <= Int.MAX_VALUE) { "Tensor ${entry.ggufName} too large" }
        return expected.toInt()
    }

    private fun computeOffsets(sizes: List<Int>, alignment: Int): List<Int> {
        val result = mutableListOf<Int>()
        var offset = 0
        for (size in sizes) {
            val aligned = alignUp(offset, alignment)
            result.add(aligned)
            offset = aligned + size
        }
        return result
    }

    private fun alignUp(value: Int, alignment: Int): Int {
        if (alignment <= 0) return value
        val rem = value % alignment
        return if (rem == 0) value else value + (alignment - rem)
    }
}

// --- Byte writer ---

private class ByteWriter {
    private val bytes = mutableListOf<Byte>()

    fun size(): Int = bytes.size

    fun writeUInt8(v: Int) { bytes.add((v and 0xFF).toByte()) }
    fun writeBytes(data: ByteArray) { data.forEach { bytes.add(it) } }

    fun writeUInt32(v: UInt) {
        writeUInt8((v and 0xFFu).toInt())
        writeUInt8(((v shr 8) and 0xFFu).toInt())
        writeUInt8(((v shr 16) and 0xFFu).toInt())
        writeUInt8(((v shr 24) and 0xFFu).toInt())
    }

    fun writeInt32(v: Int) = writeUInt32(v.toUInt())

    fun writeUInt64(v: ULong) {
        var tmp = v
        repeat(8) {
            writeUInt8((tmp and 0xFFu).toInt())
            tmp = tmp shr 8
        }
    }

    fun writeInt64(v: Long) = writeUInt64(v.toULong())

    fun writeFloat32(v: Float) = writeUInt32(v.toBits().toUInt())
    fun writeFloat64(v: Double) = writeInt64(v.toBits())

    fun align(alignment: Int) {
        if (alignment <= 0) return
        val padding = (alignment - (bytes.size % alignment)) % alignment
        repeat(padding) { bytes.add(0) }
    }

    fun pad(count: Int) {
        repeat(count) { bytes.add(0) }
    }

    fun toByteArray(): ByteArray = bytes.toByteArray()
}

// --- Tensor flattening helpers (best-effort, supports FP32/Int8/Int32 shapes) ---

private object TensorFlatten {
    fun flattenFloats(tensor: Tensor<*, *>): FloatArray {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.FloatArrayTensorData<*>) {
            return data.buffer.copyOf()
        }
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = FloatArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (data.get(*idx) as Number).toFloat()
            bump(idx, dims)
        }
        return out
    }

    fun flattenInts(tensor: Tensor<*, *>): IntArray {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.IntArrayTensorData<*>) {
            return data.buffer.copyOf()
        }
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = IntArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (data.get(*idx) as Number).toInt()
            bump(idx, dims)
        }
        return out
    }

    fun flattenBytes(tensor: Tensor<*, *>): ByteArray {
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = ByteArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            val v = tensor.data.get(*idx)
            out[i] = when (v) {
                is Byte -> v
                is Number -> v.toInt().toByte()
                else -> error("Cannot convert element $v to Byte")
            }
            bump(idx, dims)
        }
        return out
    }

    private fun bump(idx: IntArray, dims: IntArray) {
        for (i in idx.indices.reversed()) {
            idx[i]++
            if (idx[i] < dims[i]) return
            idx[i] = 0
        }
    }
}
