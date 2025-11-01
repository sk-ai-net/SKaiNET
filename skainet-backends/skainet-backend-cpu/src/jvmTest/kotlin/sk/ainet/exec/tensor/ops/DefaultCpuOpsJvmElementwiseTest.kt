package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32

class DefaultCpuOpsJvmElementwiseTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(dataFactory)

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun addVectors_usesVectorApiForFloat32() {
        val length = FloatVector.SPECIES_PREFERRED.length() * 2 + 3
        val a = floatTensor(FloatArray(length) { it.toFloat() })
        val b = floatTensor(FloatArray(length) { (it * 2).toFloat() })

        val result = ops.add(a, b)
        val resultData = result.data as FloatArrayTensorData<*>

        for (i in 0 until length) {
            assertEquals(a.data[i], i.toFloat())
            assertEquals(b.data[i], (i * 2).toFloat())
            assertEquals((i + i * 2).toFloat(), resultData.buffer[i])
        }
        assertTrue(result.shape == a.shape)
    }

    @Test
    fun relu_appliesVectorizedClamp() {
        val input = floatTensor(floatArrayOf(-2f, -0.5f, 0f, 1.5f, 3f))
        val result = ops.relu(input)
        val resultData = result.data as FloatArrayTensorData<*>
        val expected = floatArrayOf(0f, 0f, 0f, 1.5f, 3f)
        expected.forEachIndexed { index, value ->
            assertEquals(value, resultData.buffer[index])
        }
    }

    private fun floatTensor(values: FloatArray): VoidOpsTensor<FP32, Float> {
        val shape = Shape(values.size)
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }
}
