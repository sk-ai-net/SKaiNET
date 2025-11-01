package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    val useAccelerated = JvmCpuBackendConfig.vectorEnabled || JvmCpuBackendConfig.blasEnabled
    return if (useAccelerated) {
        { factory -> DefaultCpuOpsJvm(factory) }
    } else {
        { factory -> DefaultCpuOps(factory) }
    }
}
