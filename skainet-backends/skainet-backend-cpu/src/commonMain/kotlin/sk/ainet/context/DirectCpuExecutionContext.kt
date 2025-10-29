package sk.ainet.context

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.exec.tensor.ops.DefaultCpuOps

public class DirectCpuExecutionContext(
    override val executionStats: ExecutionStats = ExecutionStats(),
) : ExecutionContext {
    private val _memoryInfo = MemoryInfo(
        totalMemory = 0,
        usedMemory = 0,
        freeMemory = 0,
        usagePercentage = 0.0
    )

    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory()
    override val memoryInfo: MemoryInfo
        get() = _memoryInfo

    override val ops: TensorOps
        get() = DefaultCpuOps(tensorDataFactory)
}
