package sk.ainet.exec.tensor.ops

internal object JvmCpuBackendConfig {
    private const val ENV_VECTOR = "SKAINET_CPU_VECTOR_ENABLED"
    private const val PROP_VECTOR = "skainet.cpu.vector.enabled"
    private const val ENV_BLAS = "SKAINET_CPU_BLAS_ENABLED"
    private const val PROP_BLAS = "skainet.cpu.blas.enabled"

    val vectorEnabled: Boolean
        get() = readFlag(ENV_VECTOR, PROP_VECTOR, default = true)
    val blasEnabled: Boolean
        get() = readFlag(ENV_BLAS, PROP_BLAS, default = false)

    private fun readFlag(envKey: String, propKey: String, default: Boolean): Boolean {
        val sysProp = runCatching { System.getProperty(propKey) }.getOrNull()
            ?.toBooleanStrictOrNull()
        val env = runCatching { System.getenv(envKey) }.getOrNull()
            ?.toBooleanStrictOrNull()
        return sysProp ?: env ?: default
    }
}
