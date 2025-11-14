package sk.ainet.io.core.buffer

import sk.ainet.io.core.BufferView
import sk.ainet.io.core.ByteOrder
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path

/**
 * JVM BufferView backed by a memory-mapped file region.
 * Note: This view is read-only and assumes the file does not change during the lifetime of the view.
 */
class MMapBufferView private constructor(
    private val buffer: ByteBuffer,
    override val byteOrder: ByteOrder,
) : BufferView, JvmByteBufferBacked {

    init { buffer.clear() }

    override val size: Long get() = buffer.limit().toLong()

    override fun slice(offset: Long, length: Long): BufferView {
        require(offset >= 0 && length >= 0 && offset + length <= size) { "slice out of bounds" }
        val dup = buffer.duplicate()
        dup.position(offset.toInt())
        dup.limit((offset + length).toInt())
        return MMapBufferView(dup.slice(), byteOrder)
    }

    override fun readFully(dst: ByteArray, dstOffset: Int, length: Int): Int {
        require(dstOffset >= 0 && length >= 0 && dstOffset + length <= dst.size) { "dest out of bounds" }
        val dup = buffer.duplicate()
        dup.position(0)
        dup.limit(length)
        dup.get(dst, dstOffset, length)
        return length
    }

    override fun asByteBuffer(): ByteBuffer = buffer.asReadOnlyBuffer()

    companion object {
        /** Map a region of the file as a read-only BufferView. */
        fun mapRegion(path: String, offset: Long, length: Long, byteOrder: ByteOrder = ByteOrder.NATIVE): MMapBufferView {
            require(offset >= 0 && length >= 0) { "negative offset/length" }
            RandomAccessFile(Path.of(path).toFile(), "r").use { raf ->
                val channel: FileChannel = raf.channel
                val mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, length)
                // Byte order is a logical property here; we don't set ByteBuffer order to avoid accidental typed gets.
                return MMapBufferView(mapped, byteOrder)
            }
        }
    }
}
