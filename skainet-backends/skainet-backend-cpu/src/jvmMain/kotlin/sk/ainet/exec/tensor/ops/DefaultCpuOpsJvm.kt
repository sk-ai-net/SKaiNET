package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.data.TensorDataFactory

internal class DefaultCpuOpsJvm(
    dataFactory: TensorDataFactory,
) : DefaultCpuOpsBase(dataFactory) {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.add(y) }) { x, y -> x + y }?.let { return it }
        return super.add(a, b)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.sub(y) }) { x, y -> x - y }?.let { return it }
        return super.subtract(a, b)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.mul(y) }) { x, y -> x * y }?.let { return it }
        return super.multiply(a, b)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.div(y) }) { x, y -> x / y }?.let { return it }
        return super.divide(a, b)
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        vectorFloatUnary(tensor, { vector ->
            val zero = FloatVector.zero(floatSpecies)
            vector.max(zero)
        }, { value ->
            if (value < 0f) 0f else value
        })?.let { return it }
        return super.relu(tensor)
    }

    private fun <T : DType, V> vectorFloatBinary(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        vectorOp: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float
    ): Tensor<T, V>? {
        if (!supportsFloatOps(a, b)) return null

        val aData = a.data as? FloatArrayTensorData<T> ?: return null
        val bData = b.data as? FloatArrayTensorData<T> ?: return null
        val volume = a.shape.volume
        val outBuffer = FloatArray(volume)
        val speciesLen = floatSpecies.length()
        var index = 0
        val loopBound = floatSpecies.loopBound(volume)

        while (index < loopBound) {
            val va = FloatVector.fromArray(floatSpecies, aData.buffer, index)
            val vb = FloatVector.fromArray(floatSpecies, bData.buffer, index)
            vectorOp(va, vb).intoArray(outBuffer, index)
            index += speciesLen
        }
        while (index < volume) {
            outBuffer[index] = scalarOp(aData.buffer[index], bData.buffer[index])
            index++
        }

        val outData = DenseFloatArrayTensorData<T>(Shape(a.shape.dimensions.copyOf()), outBuffer)
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
    }

    private fun <T : DType, V> vectorFloatUnary(
        tensor: Tensor<T, V>,
        vectorOp: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float
    ): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val tensorData = tensor.data as? FloatArrayTensorData<T> ?: return null
        val volume = tensor.shape.volume
        val outBuffer = FloatArray(volume)
        val speciesLen = floatSpecies.length()
        var index = 0
        val loopBound = floatSpecies.loopBound(volume)

        while (index < loopBound) {
            val vec = FloatVector.fromArray(floatSpecies, tensorData.buffer, index)
            vectorOp(vec).intoArray(outBuffer, index)
            index += speciesLen
        }
        while (index < volume) {
            outBuffer[index] = scalarOp(tensorData.buffer[index])
            index++
        }

        val outData = DenseFloatArrayTensorData<T>(Shape(tensor.shape.dimensions.copyOf()), outBuffer)
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, tensor.dtype)
    }

    private fun <T : DType> supportsFloatOps(a: Tensor<T, *>, b: Tensor<T, *>): Boolean {
        return supportsFloatOps(a) &&
            a.dtype == b.dtype &&
            a.shape == b.shape
    }

    private fun <T : DType> supportsFloatOps(tensor: Tensor<T, *>): Boolean {
        val dtype = tensor.dtype
        return (dtype == FP32::class || dtype == FP16::class)
    }
}
