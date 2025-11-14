package sk.ainet.io.core.util

import sk.ainet.io.core.BufferView
import sk.ainet.io.core.ByteOrder
import sk.ainet.io.core.buffer.ArrayBufferView

/**
 * Returns a BufferView whose content matches the native byte order for the given elementSize.
 * If [sourceOrder] equals ByteOrder.NATIVE, returns [src] as-is. Otherwise performs a copy with per-element byte swap.
 * Supported element sizes: 1, 2, 4, 8. For others, falls back to raw copy without swap.
 */
fun ensureNativeOrder(src: BufferView, sourceOrder: ByteOrder, elementSize: Int): BufferView {
    val native = ByteOrder.NATIVE
    if (sourceOrder == native || elementSize == 1) return src
    val total = src.size.toInt()
    val bytes = ByteArray(total)
    src.readFully(bytes)
    when (elementSize) {
        2 -> swapInPlace(bytes, 2)
        4 -> swapInPlace(bytes, 4)
        8 -> swapInPlace(bytes, 8)
        else -> { /* no-op */ }
    }
    return ArrayBufferView(bytes, 0, bytes.size, native)
}

private fun swapInPlace(arr: ByteArray, elem: Int) {
    require(arr.size % elem == 0) { "Buffer size ${arr.size} not multiple of element size $elem" }
    var i = 0
    while (i < arr.size) {
        var l = 0
        var r = elem - 1
        while (l < r) {
            val tmp = arr[i + l]
            arr[i + l] = arr[i + r]
            arr[i + r] = tmp
            l++
            r--
        }
        i += elem
    }
}
