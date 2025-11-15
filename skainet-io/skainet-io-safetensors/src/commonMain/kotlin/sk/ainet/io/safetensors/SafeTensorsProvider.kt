package sk.ainet.io.safetensors

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.io.core.*
import sk.ainet.io.core.buffer.ArrayBufferView
import sk.ainet.io.core.buffer.BufferView as BufferViewFactory
import sk.ainet.io.core.spi.FormatReaderProvider
import sk.ainet.io.core.spi.ProbeResult
import sk.ainet.io.core.spi.ProviderRegistry
import sk.ainet.io.core.stream.SimpleTensorStream

/** SafeTensors Format Provider (chapter 5). */
class SafeTensorsFormatProvider : FormatReaderProvider {
    override fun formatId(): String = FORMAT_ID

    override fun probe(source: TensorSource): ProbeResult {
        return when (source) {
            is TensorSource.Bytes -> probeBytes(source)
            is TensorSource.FilePath -> {
                // Ask platform to probe (JVM/Android may do header check). Otherwise, rely on extension hint.
                val platform = SafeTensorsPlatform.probeFilePath(source.path)
                if (platform != null) return platform
                if (source.path.endsWith(".safetensors", ignoreCase = true))
                    return ProbeResult.supported(ProbeResult.EXTENSION_HINT, formatId = FORMAT_ID, reason = "extension")
                else
                    return ProbeResult.unsupported("extension not matching .safetensors", FORMAT_ID)
            }
            is TensorSource.Url -> ProbeResult.unsupported("no HTTP support yet", FORMAT_ID)
        }
    }

    private fun probeBytes(bytes: TensorSource.Bytes): ProbeResult {
        val data = bytes.bytes
        if (data.size < 8) return ProbeResult.unsupported("too small", FORMAT_ID)
        val headerSize = readU64LE(data, 0)
        val total = 8L + headerSize
        if (headerSize < 2 || total > data.size) return ProbeResult.unsupported("invalid header size", FORMAT_ID)
        val hdr = try {
            val headerJson = data.copyOfRange(8, (8 + headerSize).toInt()).decodeToString()
            Json.parseToJsonElement(headerJson)
        } catch (t: Throwable) {
            return ProbeResult.unsupported("invalid JSON header: ${t.message}", FORMAT_ID)
        }
        // Basic plausibility: at least one tensor entry
        val obj = hdr as? JsonObject ?: return ProbeResult.unsupported("header is not object", FORMAT_ID)
        val entries = obj.entries.filter { it.key != META_KEY }
        if (entries.isEmpty()) return ProbeResult.unsupported("no tensors in header", FORMAT_ID)
        return ProbeResult.supported(ProbeResult.HEADER_STRONG, version = "1", formatId = FORMAT_ID, reason = "header ok")
    }

    override fun open(source: TensorSource): CloseableTensorArchive {
        return when (source) {
            is TensorSource.Bytes -> SafeTensorsArchive.fromBytes(source)
            is TensorSource.FilePath -> {
                val bytes = SafeTensorsPlatform.readFileToBytes(source.path)
                SafeTensorsArchive.fromBytes(TensorSource.Bytes(bytes, source.path))
            }
            else -> throw IllegalArgumentException("SafeTensors: unsupported source ${source::class.simpleName}")
        }
    }

    companion object {
        const val FORMAT_ID = "safetensors"
        private const val META_KEY = "__metadata__"
    }
}

/**
 * In-memory SafeTensors archive based on a ByteArray.
 */
private class SafeTensorsArchive(
    private val sourceBytes: ByteArray,
    private val headerSize: Long,
    private val tensors: List<TensorDescriptor>,
    private val index: Map<String, TensorEntry>,
    private val globalMeta: Map<String, String>
) : CloseableTensorArchive {

    override fun metadata(): ArchiveMetadata = ArchiveMetadata(
        formatId = SafeTensorsFormatProvider.FORMAT_ID,
        version = "1",
        globalMetadata = globalMeta,
        tensorCount = tensors.size,
        totalBytes = sourceBytes.size.toLong(),
        checksums = null
    )

    override fun list(): List<TensorDescriptor> = tensors

    override fun get(name: String): TensorHandle {
        val e = index[name] ?: throw NoSuchElementException("Tensor '$name' not found")
        return SafeTensorHandle(sourceBytes, headerSize, e.descriptor, e.offset, e.length)
    }

    override fun close() { /* nothing to close for in-memory */ }

    companion object {
        fun fromBytes(src: TensorSource.Bytes): SafeTensorsArchive {
            val bytes = src.bytes
            val headerSize = readU64LE(bytes, 0)
            val headerJson = bytes.copyOfRange(8, (8 + headerSize).toInt()).decodeToString()
            val parsed = parseHeader(headerJson)
            val list = ArrayList<TensorDescriptor>(parsed.entries.size)
            val map = HashMap<String, TensorEntry>(parsed.entries.size)
            parsed.entries.forEach { (name, meta) ->
                val desc = TensorDescriptor(
                    name = name,
                    dtype = meta.dtype,
                    shape = meta.shape,
                    strides = null,
                    byteSize = meta.length,
                    endianness = ByteOrder.LITTLE_ENDIAN,
                    isContiguous = true,
                    extras = meta.extras
                )
                list += desc
                map[name] = TensorEntry(desc, meta.offset, meta.length)
            }
            return SafeTensorsArchive(bytes, headerSize, list, map, parsed.globalMeta)
        }
    }
}

private data class TensorEntry(
    val descriptor: TensorDescriptor,
    val offset: Long,
    val length: Long
)

private class SafeTensorHandle(
    private val sourceBytes: ByteArray,
    private val headerSize: Long,
    private val desc: TensorDescriptor,
    private val dataOffset: Long,
    private val dataLength: Long
) : TensorHandle {
    override fun descriptor(): TensorDescriptor = desc

    override fun stream(window: ReadWindow?): TensorStream {
        val fullView = asBufferView() ?: error("BufferView expected for in-memory bytes")
        val view = if (window == null) fullView else fullView.slice(window.offset, window.length)
        return SimpleTensorStream(view)
    }

    override fun asBufferView(): BufferView? {
        // Point a view at the exact data region for this tensor within the original bytes
        val start = (8 + headerSize + dataOffset).toInt()
        val len = dataLength.toInt()
        // Use ArrayBufferView directly to avoid copying
        return ArrayBufferView(sourceBytes, start, len, desc.endianness)
    }

    override fun materialize(factory: SkainetTensorFactory, opts: MaterializeOptions): Any {
        val view = asBufferView()
        val wantWrap = shouldWrap(desc, opts, view != null)
        if (wantWrap && view != null) {
            // Optionally verify checksum by tapping the buffer if provided (no extra copy)
            sk.ainet.io.core.util.ChecksumUtil.maybeVerifyChecksum(view, desc.extras, opts)
            return factory.wrap(view, desc)
        }
        // Copy path: read via stream to avoid assuming backing
        val total = desc.byteSize.toInt()
        val bytes = ByteArray(total)
        stream(null).use { s ->
            var off = 0
            while (off < total) {
                val n = s.read(bytes, off, total - off)
                if (n <= 0) break
                off += n
            }
            require(off == total) { "Unexpected EOF while reading tensor ${desc.name}: read $off of $total bytes" }
        }
        // Endianness handling: transform if needed
        val sourceOrder = opts.byteOrderOverride ?: desc.endianness
        val elemSize = ioElementSize(desc.dtype)
        val bv = sk.ainet.io.core.util.ensureNativeOrder(ArrayBufferView(bytes, 0, bytes.size, sourceOrder), sourceOrder, elemSize)
        sk.ainet.io.core.util.ChecksumUtil.maybeVerifyChecksum(bv, desc.extras, opts)
        return factory.wrap(bv, desc)
    }
}

private data class ParsedHeader(
    val entries: Map<String, ParsedTensorMeta>,
    val globalMeta: Map<String, String>
)

private data class ParsedTensorMeta(
    val dtype: IoDType,
    val shape: IntArray,
    val offset: Long,
    val length: Long,
    val extras: Map<String, String>
)

private fun parseHeader(json: String): ParsedHeader {
    val root = Json.parseToJsonElement(json).jsonObject
    val globalMeta = (root["__metadata__"] as? JsonObject)?.let { metaObj ->
        metaObj.mapValues { (_, v) -> v.jsonPrimitive.content }
    } ?: emptyMap()

    val entries = mutableMapOf<String, ParsedTensorMeta>()
    for ((name, el) in root) {
        if (name == "__metadata__") continue
        val obj = el as? JsonObject ?: continue
        val dtypeStr = obj["dtype"]?.jsonPrimitive?.content ?: continue
        val dtype = mapDType(dtypeStr) ?: continue
        val shapeArr = obj["shape"]?.jsonArray?.map { it.jsonPrimitive.content.toInt() }?.toIntArray() ?: continue
        val offs = obj["data_offsets"]?.jsonArray ?: continue
        if (offs.size != 2) continue
        val start = offs[0].jsonPrimitive.content.toLong()
        val end = offs[1].jsonPrimitive.content.toLong()
        val length = end - start
        val extras: Map<String, String> = obj.filterKeys { it !in setOf("dtype", "shape", "data_offsets") }
            .mapValues { (_, v) -> v.toCompactString() }
        entries[name] = ParsedTensorMeta(dtype, shapeArr, start, length, extras)
    }
    return ParsedHeader(entries, globalMeta)
}

private fun JsonElement.toCompactString(): String = when (this) {
    is JsonObject -> this.toString()
    else -> this.jsonPrimitive.content
}

private fun readU64LE(arr: ByteArray, offset: Int): Long {
    var res = 0L
    for (i in 0 until 8) {
        res = res or ((arr[offset + i].toLong() and 0xFF) shl (8 * i))
    }
    return res
}

private fun mapDType(safe: String): IoDType? = when (safe.lowercase()) {
    "f64" -> IoDType.F64
    "f32" -> IoDType.F32
    "f16" -> IoDType.F16
    "bf16", "bfloat16" -> IoDType.BF16
    "i64" -> IoDType.I64
    "i32" -> IoDType.I32
    "i16" -> IoDType.I16
    "i8" -> IoDType.I8
    "u64" -> IoDType.U64
    "u32" -> IoDType.U32
    "u16" -> IoDType.U16
    "u8" -> IoDType.U8
    "bool", "bool8" -> IoDType.BOOL
    else -> null
}

/** Auto-register provider in runtime registry for discovery on all platforms. */
@Suppress("unused")
private object SafeTensorsProviderAutoRegister {
    init {
        ProviderRegistry.register(SafeTensorsFormatProvider())
    }
}
