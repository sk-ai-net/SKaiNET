package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.execute.context.computation
import sk.ainet.execute.context.dsl.tensor
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.pprint

class DefaultCpuOpsElementwiseTest {
    private val dataFactory = DenseTensorDataFactory()
    private val cpuOpsF = DefaultCpuOps(dataFactory)
    private val cpuOpsI = DefaultCpuOps(dataFactory)


    // Helpers to build input tensors with data
    private fun fTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    private fun iTensor(shape: Shape, values: IntArray): VoidOpsTensor<Int32, Int> {
        val data = dataFactory.fromIntArray<Int32, Int>(shape, Int32::class, values)
        return VoidOpsTensor(data, Int32::class)
    }

    @Test
    fun add_fp32_sameShape() {
        val a = fTensor(Shape(3), floatArrayOf(1f, 2f, 3f))
        val b = fTensor(Shape(3), floatArrayOf(4f, 5f, 6f))
        val r = cpuOpsF.add(a, b)
        assertEquals(Shape(3), r.shape)
        assertEquals(FP32::class, r.dtype)
        assertEquals(5f, r.data[0])
        assertEquals(7f, r.data[1])
        assertEquals(9f, r.data[2])
    }

    @Test
    fun context_add_fp32_sameShape() {
        // Simulate an exec context: use DirectCpuExecutionContext and the tensor DSL

        val ctx = DirectCpuExecutionContext()

        // dsl for execution context
        computation(ctx) {
            val a = tensor<FP32, Float> {
                shape(1) { ones() }
            }
            val b = tensor<FP32, Float> {
                shape(3) {
                    init { indices ->
                        // Initialize with meaningful pattern: batch*1000 + channel*100 + height*10 + width
                        indices[0].toFloat() * 3
                    }

                }
            }
            println(a.pprint())
            println("+")
            println(b.pprint())
            val c = a + b
            println(c.pprint())
            // Broadcasting: (1) + (1,1,1) => (2,2,2)
            assertEquals(Shape(3), c.shape)
            assertEquals(FP32::class, c.dtype)
            assertEquals(1f, c.data[0])
            assertEquals(4f, c.data[1])
            assertEquals(7f, c.data[2])
        }
    }

    @Test
    fun add_int32_sameShape() {
        val a = iTensor(Shape(3), intArrayOf(1, 2, 3))
        val b = iTensor(Shape(3), intArrayOf(4, 5, 6))
        val r = cpuOpsI.add(a, b)
        assertEquals(Shape(3), r.shape)
        assertEquals(Int32::class, r.dtype)
        assertEquals(5, r.data[0])
        assertEquals(7, r.data[1])
        assertEquals(9, r.data[2])
    }

    @Test
    fun subtract_and_multiply_fp32() {
        val a = fTensor(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f))
        val b = fTensor(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val sub = cpuOpsF.subtract(a, b)
        val mul = cpuOpsF.multiply(a, b)
        // subtract
        assertEquals(4f, sub.data[0, 0])
        assertEquals(4f, sub.data[0, 1])
        assertEquals(4f, sub.data[1, 0])
        assertEquals(4f, sub.data[1, 1])
        // multiply
        assertEquals(5f, mul.data[0, 0])
        assertEquals(12f, mul.data[0, 1])
        assertEquals(21f, mul.data[1, 0])
        assertEquals(32f, mul.data[1, 1])
    }

    @Test
    fun divide_fp32() {
        val a = fTensor(Shape(3), floatArrayOf(8f, 9f, 10f))
        val b = fTensor(Shape(3), floatArrayOf(2f, 3f, 5f))
        val r = cpuOpsF.divide(a, b)
        assertEquals(4f, r.data[0])
        assertEquals(3f, r.data[1])
        assertEquals(2f, r.data[2])
    }

    @Test
    fun broadcasting_scalar_vector_fp32() {
        val a = fTensor(Shape(1), floatArrayOf(2f)) // acts as scalar
        val b = fTensor(Shape(4), floatArrayOf(1f, 2f, 3f, 4f))
        val r = cpuOpsF.multiply(a, b)
        assertEquals(Shape(4), r.shape)
        assertEquals(2f, r.data[0])
        assertEquals(4f, r.data[1])
        assertEquals(6f, r.data[2])
        assertEquals(8f, r.data[3])
    }

    @Test
    fun broadcasting_vector_matrix_fp32() {
        val a = fTensor(
            Shape(2, 3), floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f
            )
        )
        val b = fTensor(Shape(3), floatArrayOf(10f, 1f, 0.5f))
        val r = cpuOpsF.add(a, b)
        assertEquals(Shape(2, 3), r.shape)
        // row 0
        assertEquals(11f, r.data[0, 0])
        assertEquals(3f, r.data[0, 1])
        assertEquals(3.5f, r.data[0, 2])
        // row 1
        assertEquals(14f, r.data[1, 0])
        assertEquals(6f, r.data[1, 1])
        assertEquals(6.5f, r.data[1, 2])
    }
}
