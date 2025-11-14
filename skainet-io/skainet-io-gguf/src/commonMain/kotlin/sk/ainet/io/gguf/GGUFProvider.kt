package sk.ainet.io.gguf

import sk.ainet.io.core.*
import sk.ainet.io.core.buffer.ArrayBufferView
import sk.ainet.io.core.spi.FormatReaderProvider
import sk.ainet.io.core.spi.ProbeResult
import sk.ainet.io.core.spi.ProviderRegistry
import sk.ainet.io.core.stream.SimpleTensorStream

/** GGUF Format Provider (chapter 6). */
class GGUFFormatProvider : FormatReaderProvider {
    override fun formatId(): String = FORMAT_ID

    override fun probe(source: TensorSource): ProbeResult {
        return when (source) {
            is TensorSource.Bytes -> probeBytes(source)
            is TensorSource.FilePath -> {
                val hint = if (source.path.endsWith(".gguf", ignoreCase = true))
                    ProbeResult.supported(ProbeResult.EXTENSION_HINT, formatId = FORMAT_ID, reason = "extension")
                else ProbeResult.unsupported("extension not matching .gguf", FORMAT_ID)
                // Keep unsupported unless bytes are provided, to avoid accidental selection without header validation
                if (hint.supported) hint else ProbeResult.unsupported("no header validation for FilePath in common", FORMAT_ID)
            }
            is TensorSource.Url -> ProbeResult.unsupported("no HTTP support yet", FORMAT_ID)
        }
    }

    private fun probeBytes(bytes: TensorSource.Bytes): ProbeResult {
        val data = bytes.bytes
        if (data.size < 8) return ProbeResult.unsupported("too small", FORMAT_ID)
        // magic: 4 bytes little-endian should equal GGUF_MAGIC
        val magic = ((data[0].toInt() and 0xFF)) or
                ((data[1].toInt() and 0xFF) shl 8) or
                ((data[2].toInt() and 0xFF) shl 16) or
                ((data[3].toInt() and 0xFF) shl 24)
        if (magic.toUInt() != GGUF_MAGIC) return ProbeResult.unsupported("bad magic", FORMAT_ID)
        val version = (data[4].toInt() and 0xFF) or
                ((data[5].toInt() and 0xFF) shl 8) or
                ((data[6].toInt() and 0xFF) shl 16) or
                ((data[7].toInt() and 0xFF) shl 24)
        val supported = version in READER_SUPPORTED_VERSIONS
        return if (supported) ProbeResult.supported(ProbeResult.HEADER_STRONG, version = version.toString(), formatId = FORMAT_ID, reason = "header ok")
        else ProbeResult.unsupported("unsupported version $version", FORMAT_ID)
    }

    override fun open(source: TensorSource): CloseableTensorArchive {
        return when (source) {
            is TensorSource.Bytes -> GGUFArchive.fromBytes(source)
            else -> throw IllegalArgumentException("GGUF: only TensorSource.Bytes supported in common; got ${source::class.simpleName}")
        }
    }

    companion object {
        const val FORMAT_ID = "gguf"
    }
}

private class GGUFArchive(
    private val sourceBytes: ByteArray,
    private val reader: GGUFReader,
    private val tensors: List<TensorDescriptor>,
    private val index: Map<String, TensorEntry>,
    private val globalMeta: Map<String, String>
) : CloseableTensorArchive {

    override fun metadata(): ArchiveMetadata = ArchiveMetadata(
        formatId = GGUFFormatProvider.FORMAT_ID,
        version = (reader.fields["GGUF.version"]?.parts?.getOrNull(0)?.getOrNull(0) as? UInt)?.toInt()?.toString(),
        globalMetadata = globalMeta,
        tensorCount = tensors.size,
        totalBytes = sourceBytes.size.toLong(),
        checksums = null
    )

    override fun list(): List<TensorDescriptor> = tensors

    override fun get(name: String): TensorHandle {
        val e = index[name] ?: throw NoSuchElementException("Tensor '$name' not found")
        return GGUFTensorHandle(sourceBytes, e.descriptor, e.offset, e.length)
    }

    override fun close() { /* in-memory */ }

    companion object {
        fun fromBytes(src: TensorSource.Bytes): GGUFArchive {
            val bytes = src.bytes
            val reader = GGUFReader(bytes, loadTensorData = false)

            // Build global metadata from reader.fields (stringifiable values)
            val meta = reader.fields.mapValues { (_, field) ->
                // Try to extract human-readable string for single string field; else fall back to toString
                try {
                    if (field.types.size == 1 && field.types[0] == GGUFValueType.STRING) {
                        val data = field.parts[field.data[0]] as List<UByte>
                        data.toUByteArray().toByteArray().decodeToString()
                    } else field.parts[field.data[0]].toString()
                } catch (_: Throwable) {
                    field.parts.toString()
                }
            }

            val list = ArrayList<TensorDescriptor>(reader.tensors.size)
            val map = HashMap<String, TensorEntry>(reader.tensors.size)
            for (t in reader.tensors) {
                val (ioDType, extras) = mapDTypeForDescriptor(t.tensorType)
                val desc = TensorDescriptor(
                    name = t.name,
                    dtype = ioDType,
                    shape = t.shape.map { it.toInt() }.toIntArray(),
                    strides = null,
                    byteSize = t.nBytes.toLong(),
                    endianness = ByteOrder.LITTLE_ENDIAN,
                    isContiguous = true,
                    extras = extras
                )
                list += desc
                map[t.name] = TensorEntry(desc, t.dataOffset.toLong(), t.nBytes.toLong())
            }
            return GGUFArchive(bytes, reader, list, map, meta)
        }

        private fun mapDTypeForDescriptor(q: GGMLQuantizationType): Pair<IoDType, Map<String, String>> = when (q) {
            GGMLQuantizationType.F64 -> IoDType.F64 to emptyMap()
            GGMLQuantizationType.F32 -> IoDType.F32 to emptyMap()
            GGMLQuantizationType.F16 -> IoDType.F16 to emptyMap()
            GGMLQuantizationType.BF16 -> IoDType.BF16 to emptyMap()
            GGMLQuantizationType.I64 -> IoDType.I64 to emptyMap()
            GGMLQuantizationType.I32 -> IoDType.I32 to emptyMap()
            GGMLQuantizationType.I16 -> IoDType.I16 to emptyMap()
            GGMLQuantizationType.I8 -> IoDType.I8 to emptyMap()
            // Non-native/quantized types: represent as bytes and keep ggml type in extras
            else -> IoDType.U8 to mapOf(
                "ggml.quantized" to "true",
                "ggml.type" to q.name
            )
        }
    }
}

private data class TensorEntry(
    val descriptor: TensorDescriptor,
    val offset: Long,
    val length: Long
)

private class GGUFTensorHandle(
    private val sourceBytes: ByteArray,
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
        val start = dataOffset.toInt()
        val len = dataLength.toInt()
        return ArrayBufferView(sourceBytes, start, len, desc.endianness)
    }

    override fun materialize(factory: SkainetTensorFactory, opts: MaterializeOptions): Any {
        val view = asBufferView()
        val wantWrap = shouldWrap(desc, opts, view != null)
        if (wantWrap && view != null) {
            sk.ainet.io.core.util.ChecksumUtil.maybeVerifyChecksum(view, desc.extras, opts)
            return factory.wrap(view, desc)
        }
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
        val sourceOrder = opts.byteOrderOverride ?: desc.endianness
        val elemSize = ioElementSize(desc.dtype)
        val bv = sk.ainet.io.core.util.ensureNativeOrder(ArrayBufferView(bytes, 0, bytes.size, sourceOrder), sourceOrder, elemSize)
        sk.ainet.io.core.util.ChecksumUtil.maybeVerifyChecksum(bv, desc.extras, opts)
        return factory.wrap(bv, desc)
    }
}

/** Auto-register provider in runtime registry for discovery on all platforms. */
@Suppress("unused")
private object GGUFProviderAutoRegister {
    init { ProviderRegistry.register(GGUFFormatProvider()) }
}
