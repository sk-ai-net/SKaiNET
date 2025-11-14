package sk.ainet.io.core.buffer

import sk.ainet.io.core.BufferView
import sk.ainet.io.core.ByteOrder

/**
 * A simple ByteArray-backed BufferView with slicing and copying support.
 */
class ArrayBufferView(
    internal val backing: ByteArray,
    internal val offset: Int = 0,
    internal val length: Int = backing.size - offset,
    override val byteOrder: ByteOrder = ByteOrder.NATIVE,
) : BufferView {

    init {
        require(offset >= 0) { "offset < 0" }
        require(length >= 0) { "length < 0" }
        require(offset + length <= backing.size) { "slice out of bounds: ${offset}+${length} > ${backing.size}" }
    }

    override val size: Long get() = length.toLong()

    override fun slice(offset: Long, length: Long): BufferView {
        require(offset >= 0 && length >= 0) { "negative offset/length" }
        val o = offset.toInt()
        val l = length.toInt()
        require(o + l <= this.length) { "slice out of bounds: ${o}+${l} > ${this.length}" }
        return ArrayBufferView(backing, this.offset + o, l, byteOrder)
    }

    override fun readFully(dst: ByteArray, dstOffset: Int, length: Int): Int {
        require(dstOffset >= 0 && length >= 0 && dstOffset + length <= dst.size) { "dest out of bounds" }
        if (length == 0) return 0
        require(length <= this.length) { "requested length exceeds view size" }
        backing.copyInto(destination = dst, destinationOffset = dstOffset, startIndex = offset, endIndex = offset + length)
        return length
    }

    fun toByteArrayCopy(): ByteArray = backing.copyOfRange(offset, offset + length)
}

/** Factory helpers */
fun BufferView(bytes: ByteArray, byteOrder: ByteOrder = ByteOrder.NATIVE): BufferView =
    ArrayBufferView(bytes, 0, bytes.size, byteOrder)
