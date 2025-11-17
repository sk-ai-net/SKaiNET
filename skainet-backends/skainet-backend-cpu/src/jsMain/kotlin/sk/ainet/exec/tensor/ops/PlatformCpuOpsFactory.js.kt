package sk.ainet.exec.tensor.ops

internal actual fun platformDefaultCpuOpsFactory(): (sk.ainet.lang.tensor.data.TensorDataFactory) -> sk.ainet.lang.tensor.ops.TensorOps =
    { factory -> DefaultCpuOps(factory) }
