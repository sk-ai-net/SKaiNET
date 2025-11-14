package sk.ainet.io.core.stream

import sk.ainet.io.core.TensorStream
import sk.ainet.io.core.BufferView

/**
 * Simple single-consumer TensorStream that reads from a BufferView by copying into caller-provided buffers.
 * Optional hooks allow observing bytes for checksum updates and a close callback.
 */
class SimpleTensorStream(
    private val source: BufferView,
    private val onBytes: ((ByteArray, Int, Int) -> Unit)? = null,
    private val onClose: (() -> Unit)? = null,
) : TensorStream {
    private var position: Long = 0
    private var closed: Boolean = false

    override fun read(dst: ByteArray, dstOffset: Int, length: Int): Int {
        check(!closed) { "Stream is closed" }
        if (position >= source.size) return -1
        val toRead = minOf(length.toLong(), (dst.size - dstOffset).toLong(), (source.size - position)).toInt()
        if (toRead <= 0) return 0
        val slice = source.slice(position, toRead.toLong())
        val n = slice.readFully(dst, dstOffset, toRead)
        onBytes?.invoke(dst, dstOffset, n)
        position += n
        return n
    }

    override fun close() {
        if (!closed) {
            closed = true
            onClose?.invoke()
        }
    }
}
