package sk.ainet.io.core.http

import sk.ainet.io.core.TensorStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * A simple HTTP Range-based TensorStream for JVM. It performs sequential range GET requests of [chunkSize] bytes.
 * This is minimal and intended for testing/prototyping. Production code should use a robust HTTP client.
 */
class RangeHttpTensorStream(
    private val url: String,
    private val totalSize: Long?,
    private val chunkSize: Int = 1 shl 20, // 1 MiB
    private val maxRetries: Int = 2,
    private val backoffMillis: Long = 200,
) : TensorStream {

    private var position: Long = 0
    private var closed: Boolean = false
    private var currentChunk: ByteArray = ByteArray(0)
    private var chunkPos: Int = 0

    override fun read(dst: ByteArray, dstOffset: Int, length: Int): Int {
        check(!closed) { "Stream closed" }
        if (totalSize != null && position >= totalSize) return -1

        if (chunkPos >= currentChunk.size) {
            if (!fetchNextChunk()) return -1
        }
        val toCopy = minOf(length, currentChunk.size - chunkPos)
        if (toCopy <= 0) return 0
        System.arraycopy(currentChunk, chunkPos, dst, dstOffset, toCopy)
        chunkPos += toCopy
        position += toCopy
        return toCopy
    }

    private fun fetchNextChunk(): Boolean {
        val start = position
        val endExclusive = if (totalSize == null) start + chunkSize else minOf(totalSize, start + chunkSize.toLong())
        if (totalSize != null && start >= totalSize) return false
        val rangeHeader = "bytes=$start-${endExclusive - 1}"
        var attempt = 0
        while (true) {
            try {
                val conn = URL(url).openConnection() as HttpURLConnection
                conn.requestMethod = "GET"
                conn.setRequestProperty("Range", rangeHeader)
                conn.instanceFollowRedirects = true
                conn.connectTimeout = 10_000
                conn.readTimeout = 30_000
                val code = conn.responseCode
                if (code == 206 || code == 200) {
                    val len = conn.contentLength
                    val data = conn.inputStream.use(InputStream::readAllBytes)
                    currentChunk = data
                    chunkPos = 0
                    return data.isNotEmpty()
                } else if (code in 300..399) {
                    // simple redirect handling
                    val loc = conn.getHeaderField("Location")
                    if (!loc.isNullOrBlank()) {
                        return followRedirectAndFetch(loc, rangeHeader)
                    }
                } else if (code == 416) {
                    return false
                }
                throw RuntimeException("HTTP $code for range $rangeHeader")
            } catch (e: Exception) {
                if (attempt++ >= maxRetries) throw e
                Thread.sleep(backoffMillis * attempt)
            }
        }
    }

    private fun followRedirectAndFetch(newUrl: String, rangeHeader: String): Boolean {
        var attempt = 0
        while (true) {
            try {
                val conn = URL(newUrl).openConnection() as HttpURLConnection
                conn.requestMethod = "GET"
                conn.setRequestProperty("Range", rangeHeader)
                conn.connectTimeout = 10_000
                conn.readTimeout = 30_000
                val code = conn.responseCode
                if (code == 206 || code == 200) {
                    val data = conn.inputStream.use(InputStream::readAllBytes)
                    currentChunk = data
                    chunkPos = 0
                    return data.isNotEmpty()
                }
                throw RuntimeException("HTTP $code after redirect for range $rangeHeader")
            } catch (e: Exception) {
                if (attempt++ >= maxRetries) throw e
                Thread.sleep(200L * attempt)
            }
        }
    }

    override fun close() {
        closed = true
        currentChunk = ByteArray(0)
    }
}
