package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    val jdkOk = isJdk21Plus()
    val vectorAvailable = jdkOk && isVectorApiAvailable()
    val useVector = (JvmCpuBackendConfig.vectorEnabled && vectorAvailable)
    return if (useVector) {
        { factory -> DefaultCpuOpsJvm(factory) }
    } else {
        // Note: BLAS acceleration not yet implemented; falling back to DefaultCpuOps
        { factory -> DefaultCpuOps(factory) }
    }
}

private fun isVectorApiAvailable(): Boolean {
    return runCatching {
        Class.forName("jdk.incubator.vector.FloatVector")
        Class.forName("jdk.incubator.vector.VectorSpecies")
        true
    }.getOrElse { false }
}

private fun isJdk21Plus(): Boolean {
    // Prefer Runtime.version() when available (Java 9+)
    val runtimeVersion = runCatching {
        val runtimeClass = Class.forName("java.lang.Runtime")
        val versionMethod = runtimeClass.getMethod("version")
        val versionObj = versionMethod.invoke(Runtime.getRuntime())
        val featureMethod = versionObj.javaClass.getMethod("feature")
        featureMethod.invoke(versionObj) as Int
    }.getOrNull()
    if (runtimeVersion != null) return runtimeVersion >= 21

    // Fallback to parsing java.specification.version
    val spec = System.getProperty("java.specification.version") ?: return false
    return spec.toIntOrNull()?.let { it >= 21 } ?: run {
        // Handle versions like "1.8", "11", "21.0.1"
        val major = spec.split('.', '-').firstOrNull()?.toIntOrNull() ?: return@run false
        major >= 21
    }
}
