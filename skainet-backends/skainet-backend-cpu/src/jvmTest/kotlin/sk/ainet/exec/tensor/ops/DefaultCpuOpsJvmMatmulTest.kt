package sk.ainet.exec.tensor.ops

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32

class DefaultCpuOpsJvmMatmulTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(dataFactory)

    @AfterTest
    fun teardown() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun matmul2x3x3x2MatchesReference() {
        val aValues = floatArrayOf(
            1f, 2f, 3f,
            4f, 5f, 6f
        )
        val bValues = floatArrayOf(
            7f, 8f,
            9f, 10f,
            11f, 12f
        )
        val a = tensor(Shape(2, 3), aValues)
        val b = tensor(Shape(3, 2), bValues)

        val result = ops.matmul(a, b)
        val resultBuffer = (result.data as FloatArrayTensorData<*>).buffer

        val expected = floatArrayOf(
            58f, 64f,
            139f, 154f
        )
        expected.forEachIndexed { index, value ->
            assertEquals(value, resultBuffer[index], 1e-5f)
        }
    }

    private fun tensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }
}
