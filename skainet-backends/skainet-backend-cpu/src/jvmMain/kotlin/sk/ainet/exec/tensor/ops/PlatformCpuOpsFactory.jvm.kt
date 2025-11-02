package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    val vectorAvailable = isVectorApiAvailable()
    val useAccelerated = (JvmCpuBackendConfig.vectorEnabled && vectorAvailable) || JvmCpuBackendConfig.blasEnabled
    return if (useAccelerated) {
        { factory -> DefaultCpuOpsJvm(factory) }
    } else {
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
