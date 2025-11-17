package sk.ainet.bench

import org.openjdk.jmh.annotations.*
import java.util.concurrent.TimeUnit
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class MatmulBench {
    @Param("256", "512", "1024")
    var size: Int = 512

    @Param("true", "false")
    var vectorEnabled: String = "true"

    @Param("true", "false")
    var blasEnabled: String = "true"

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext

    private lateinit var a: VoidOpsTensor<FP32, Float>
    private lateinit var b: VoidOpsTensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        System.setProperty("skainet.cpu.vector.enabled", vectorEnabled)
        System.setProperty("skainet.cpu.blas.enabled", blasEnabled)
        ctx = DirectCpuExecutionContext()
        val n = size
        val shapeA = Shape(n, n)
        val shapeB = Shape(n, n)
        val arrA = FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f }
        val arrB = FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f }
        val dataA = dataFactory.fromFloatArray<FP32, Float>(shapeA, FP32::class, arrA)
        val dataB = dataFactory.fromFloatArray<FP32, Float>(shapeB, FP32::class, arrB)
        a = VoidOpsTensor(dataA, FP32::class)
        b = VoidOpsTensor(dataB, FP32::class)
    }

    @Benchmark
    fun matmul_fp32_square(): Any {
        return ctx.ops.matmul(a, b)
    }
}
