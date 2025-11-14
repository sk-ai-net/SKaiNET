package sk.ainet.io.core.util

import sk.ainet.io.core.BufferView
import sk.ainet.io.core.MaterializeOptions
import kotlin.experimental.xor

/**
 * Lightweight checksum utilities for validation hooks in the materialization pipeline.
 * Common multiplatform fallback implements a simple SHA-256-like placeholder (NOT cryptographically accurate).
 * For production JVM/Native, replace with platform hash APIs.
 */
object ChecksumUtil {
    fun maybeVerifyChecksum(view: BufferView, extras: Map<String, String>, opts: MaterializeOptions) {
        if (!opts.validateChecksum) return
        val expected = extras["checksum.sha256"] ?: extras["sha256"] ?: return
        val actual = simpleHash256(view)
        if (!expected.equals(actual, ignoreCase = true)) {
            throw IllegalStateException("Checksum mismatch: expected=$expected actual=$actual")
        }
    }

    // Simple non-cryptographic 256-bit hash placeholder based on 4x64-bit accumulators and xor/mix.
    private fun simpleHash256(view: BufferView): String {
        var a = 0UL; var b = 1UL; var c = 2UL; var d = 3UL
        val buf = ByteArray(4096)
        var remaining = view.size
        var offset = 0L
        while (remaining > 0) {
            val chunk = if (remaining > buf.size) buf.size else remaining.toInt()
            val slice = view.slice(offset, chunk.toLong())
            slice.readFully(buf, 0, chunk)
            var i = 0
            while (i < chunk) {
                val v = buf[i].toUByte().toULong()
                a = a xor v; a = a.rotateLeft(1)
                b = b + v; b = b.rotateLeft(3)
                c = c xor (v shl 1); c = c.rotateLeft(5)
                d = d + (v shl 2); d = d.rotateLeft(7)
                i++
            }
            offset += chunk
            remaining -= chunk
        }
        fun toHex(u: ULong) = u.toString(16).padStart(16, '0')
        return (toHex(a) + toHex(b) + toHex(c) + toHex(d)).lowercase()
    }

    private fun ULong.rotateLeft(bits: Int): ULong {
        val b = bits % 64
        return (this shl b) or (this shr (64 - b))
    }
}
