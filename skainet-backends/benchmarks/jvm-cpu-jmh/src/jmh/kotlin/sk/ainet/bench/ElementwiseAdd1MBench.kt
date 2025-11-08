package sk.ainet.bench

import org.openjdk.jmh.annotations.*
import java.util.concurrent.TimeUnit
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class ElementwiseAdd1MBench {
    @Param("true", "false")
    var vectorEnabled: String = "true"

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext

    private lateinit var a: VoidOpsTensor<FP32, Float>
    private lateinit var b: VoidOpsTensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        // Toggle vector path via system property
        System.setProperty("skainet.cpu.vector.enabled", vectorEnabled)
        // Leave BLAS as-is for elementwise
        ctx = DirectCpuExecutionContext()
        val n = 1_000_000
        val shape = Shape(n)
        val arrA = FloatArray(n) { it.toFloat() * 0.5f }
        val arrB = FloatArray(n) { 1f }
        val dataA = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrA)
        val dataB = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrB)
        a = VoidOpsTensor(dataA, FP32::class)
        b = VoidOpsTensor(dataB, FP32::class)
    }

    @Benchmark
    fun add_1M_fp32(): Any {
        // Return the result tensor so JIT cannot dead-code eliminate work
        return ctx.ops.add(a, b)
    }
}
