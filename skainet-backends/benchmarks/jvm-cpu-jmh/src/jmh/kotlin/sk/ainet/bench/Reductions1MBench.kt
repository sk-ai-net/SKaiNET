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
open class Reductions1MBench {
    @Param("true", "false")
    var vectorEnabled: String = "true"

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext

    private lateinit var a: VoidOpsTensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        System.setProperty("skainet.cpu.vector.enabled", vectorEnabled)
        ctx = DirectCpuExecutionContext()
        val n = 1_000_000
        val shape = Shape(n)
        val arrA = FloatArray(n) { (it % 1024).toFloat() * 0.25f }
        val dataA = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrA)
        a = VoidOpsTensor(dataA, FP32::class)
    }

    @Benchmark
    fun sum_1M_fp32(): Any {
        return ctx.ops.sum(a)
    }

    @Benchmark
    fun mean_1M_fp32(): Any {
        return ctx.ops.mean(a)
    }
}
