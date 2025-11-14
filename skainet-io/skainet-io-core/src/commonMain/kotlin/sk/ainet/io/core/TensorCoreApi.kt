package sk.ainet.io.core

/**
 * Core IO API (interfaces and minimal types)
 * These are format-agnostic abstractions living in skainet-io-core.
 * They intentionally avoid platform-specific types to remain multiplatform-friendly.
 */

// 1.1 TensorSource abstraction (minimal set, extendable later)
sealed interface TensorSource {
    /** A human-friendly identifier for logging/debugging. */
    val id: String

    /** Local file by path. Platforms may interpret path accordingly. */
    data class FilePath(val path: String) : TensorSource { override val id: String = path }

    /** HTTP/HTTPS URL resource. Range support is negotiated by providers. */
    data class Url(val url: String) : TensorSource { override val id: String = url }

    /** In-memory bytes, primarily for tests and small payloads. */
    data class Bytes(val bytes: ByteArray, override val id: String = "<bytes>") : TensorSource
}

// 1.8 Supporting types first: ReadWindow, ByteOrder, BufferView, TensorStream
/** Byte order used by the underlying serialized tensor bytes. */
enum class ByteOrder { LITTLE_ENDIAN, BIG_ENDIAN, NATIVE }

/**
 * A logical window for ranged reads. When null, it implies the full tensor content.
 */
data class ReadWindow(val offset: Long, val length: Long)

/**
 * A lightweight, possibly zero-copy view over a byte buffer.
 * Implementations may be mmap-backed, direct native buffers, or simple arrays.
 */
interface BufferView {
    val size: Long
    val byteOrder: ByteOrder

    /** Return a slice view; must be within [0, size]. */
    fun slice(offset: Long, length: Long): BufferView

    /** Read the entire view into the destination array. Returns number of bytes read. */
    fun readFully(dst: ByteArray, dstOffset: Int = 0, length: Int = dst.size - dstOffset): Int
}

/**
 * A single-consumer stream for chunked reading of tensor bytes. Implementations should not be reused after close().
 */
interface TensorStream : AutoCloseable {
    /** Read up to [length] bytes into [dst] starting at [dstOffset]. Returns -1 on EOF. */
    fun read(dst: ByteArray, dstOffset: Int = 0, length: Int = dst.size - dstOffset): Int

    override fun close()
}

// 1.4 TensorDescriptor
/** Minimal IO-level dtype set; mapping to skainet-lang DType is handled separately. */
enum class IoDType {
    F64, F32, F16, BF16,
    I64, I32, I16, I8,
    U64, U32, U16, U8,
    BOOL
}

data class TensorDescriptor(
    val name: String,
    val dtype: IoDType,
    val shape: IntArray,
    val strides: IntArray? = null,
    val byteSize: Long,
    val endianness: ByteOrder = ByteOrder.NATIVE,
    val isContiguous: Boolean = strides == null,
    val extras: Map<String, String> = emptyMap()
)

// 1.6 ArchiveMetadata

data class ArchiveMetadata(
    val formatId: String,
    val version: String? = null,
    val globalMetadata: Map<String, String> = emptyMap(),
    val tensorCount: Int,
    val totalBytes: Long,
    val checksums: Map<String, String>? = null // by tensor name when available
)

// 1.7 MaterializeOptions

data class MaterializeOptions(
    val preferZeroCopy: Boolean = true,
    val validateChecksum: Boolean = false,
    val deviceHint: String? = null,
    val memoryHint: String? = null,
    val byteOrderOverride: ByteOrder? = null,
    val fallbackToCopy: Boolean = true
)

// Forward-declare a minimal factory contract to decouple from skainet-lang while enabling materialize().
interface SkainetTensorFactory {
    // Allocate a new tensor compatible with [desc] and return an opaque handle instance.
    fun allocate(desc: TensorDescriptor): Any

    // Wrap an existing buffer view as a tensor when layout/contiguity allow. Should throw if not supported.
    fun wrap(buffer: BufferView, desc: TensorDescriptor): Any
}

// 1.5 TensorHandle
interface TensorHandle {
    fun descriptor(): TensorDescriptor

    /**
     * Open a stream for this tensor's bytes. If [window] is null, the full content is streamed.
     */
    fun stream(window: ReadWindow? = null): TensorStream

    /** Return a zero/min-copy buffer view when possible, else null. */
    fun asBufferView(): BufferView?

    /** Materialize into a concrete tensor via the provided factory and options. */
    fun materialize(factory: SkainetTensorFactory, opts: MaterializeOptions = MaterializeOptions()): Any
}

// 1.2 CloseableTensorArchive
interface CloseableTensorArchive : AutoCloseable {
    fun metadata(): ArchiveMetadata
    fun list(): List<TensorDescriptor>
    fun get(name: String): TensorHandle
    override fun close()
}

// 1.3 TensorReader facade
interface TensorReader {
    fun open(source: TensorSource): CloseableTensorArchive
}
