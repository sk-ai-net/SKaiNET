package sk.ainet.io.core.buffer

import sk.ainet.io.core.BufferView
import sk.ainet.io.core.ByteOrder
import java.nio.ByteBuffer

/** Marker for JVM BufferView implementations that can expose a ByteBuffer directly. */
internal interface JvmByteBufferBacked {
    fun asByteBuffer(): ByteBuffer
}

/** JVM-only helper to obtain a ByteBuffer view when available. */
fun BufferView.asByteBufferOrNull(): ByteBuffer? = when (this) {
    is JvmByteBufferBacked -> this.asByteBuffer()
    is ArrayBufferView -> ByteBuffer.wrap(this.backing, this.offset, this.length)
    else -> null
}
