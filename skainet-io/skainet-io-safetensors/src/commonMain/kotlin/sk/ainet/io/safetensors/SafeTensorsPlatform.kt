package sk.ainet.io.safetensors

import sk.ainet.io.core.spi.ProbeResult

/**
 * Platform hooks used by SafeTensors provider. JVM/Android may implement file handling.
 * Other targets can return null/throw to indicate unsupported FilePath handling.
 */
expect object SafeTensorsPlatform {
    /** Return a ProbeResult for a local file path, or null if not supported on this platform. */
    fun probeFilePath(path: String): ProbeResult?

    /** Read entire file into memory as ByteArray, or throw if not supported. */
    fun readFileToBytes(path: String): ByteArray
}
