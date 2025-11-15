package sk.ainet.io.safetensors

import sk.ainet.io.core.spi.ProbeResult
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

actual object SafeTensorsPlatform {
    actual fun probeFilePath(path: String): ProbeResult? {
        try {
            val f = File(path)
            if (!f.exists() || !f.isFile) return ProbeResult.unsupported("not a file", SafeTensorsFormatProvider.FORMAT_ID)
            FileChannel.open(f.toPath(), StandardOpenOption.READ).use { ch ->
                if (ch.size() < 8) return ProbeResult.unsupported("too small", SafeTensorsFormatProvider.FORMAT_ID)
                val hdrSizeBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                ch.read(hdrSizeBuf, 0)
                hdrSizeBuf.flip()
                val headerSize = hdrSizeBuf.long
                if (headerSize < 2) return ProbeResult.unsupported("invalid header size", SafeTensorsFormatProvider.FORMAT_ID)
                val totalHeader = 8L + headerSize
                if (totalHeader > ch.size()) return ProbeResult.unsupported("header beyond EOF", SafeTensorsFormatProvider.FORMAT_ID)
                // Try to parse JSON header bytes for plausibility
                val hdrBytes = ByteArray(headerSize.toInt())
                val bb = ByteBuffer.wrap(hdrBytes)
                ch.read(bb, 8)
                val text = hdrBytes.decodeToString()
                // very light check: must start with '{'
                if (text.trimStart().startsWith("{")) {
                    return ProbeResult.supported(ProbeResult.HEADER_STRONG, version = "1", formatId = SafeTensorsFormatProvider.FORMAT_ID, reason = "header ok")
                }
                return ProbeResult.unsupported("header not json object", SafeTensorsFormatProvider.FORMAT_ID)
            }
        } catch (_: Throwable) {
            return ProbeResult.unsupported("exception while probing", SafeTensorsFormatProvider.FORMAT_ID)
        }
    }

    actual fun readFileToBytes(path: String): ByteArray {
        val f = File(path)
        require(f.exists() && f.isFile) { "File not found: $path" }
        return f.readBytes()
    }
}
