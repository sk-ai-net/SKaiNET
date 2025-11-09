package sk.ainet.exec.tensor.ops

import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32

class DefaultCpuOpsJvmBroadcastPropertyTest {
    private val dataFactory = DenseTensorDataFactory()

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun scalar_broadcast_and_last_dim_bias_match_scalar_reference() {
        val rng = Random(42)

        repeat(25) {
            // Vector + scalar
            val species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
            val len = (1..3).random(rng) * species + rng.nextInt(0, species) // random tail
            val a = FloatArray(len) { i -> (i - len / 2).toFloat() / 10f }
            val bScalar = floatArrayOf(rng.nextInt(-3, 4).toFloat())

            val tA = floatTensor(Shape(len), a)
            val tB = floatTensor(Shape(1), bScalar)

            val jvmOps = DefaultCpuOpsJvm(dataFactory)
            val scalarOps = DefaultCpuOps(dataFactory)

            val rJvm = jvmOps.add(tA, tB)
            val rScalar = scalarOps.add(tA, tB)

            val outJvm = (rJvm.data as FloatArrayTensorData<*>).buffer
            val outScalar = (rScalar.data as FloatArrayTensorData<*>).buffer
            for (i in outJvm.indices) {
                assertEquals(outScalar[i], outJvm[i], 1e-6f)
            }
        }

        repeat(10) {
            // Last-dimension bias broadcasting: (B,H,W) + (1,W)
            val b = 2
            val h = 3
            val w = 7
            val big = FloatArray(b * h * w) { idx -> ((idx % 11) - 5).toFloat() }
            val bias = FloatArray(w) { i -> (i - 3).toFloat() / 7f }

            val tBig = floatTensor(Shape(b, h, w), big)
            val tBias = floatTensor(Shape(1, w), bias)

            val jvmOps = DefaultCpuOpsJvm(dataFactory)
            val scalarOps = DefaultCpuOps(dataFactory)

            val rJvm = jvmOps.add(tBig, tBias)
            val rScalar = scalarOps.add(tBig, tBias)

            val outJvm = (rJvm.data as FloatArrayTensorData<*>).buffer
            val outScalar = (rScalar.data as FloatArrayTensorData<*>).buffer
            for (i in outJvm.indices) {
                assertEquals(outScalar[i], outJvm[i], 1e-6f)
            }
        }
    }

    private fun floatTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }
}
