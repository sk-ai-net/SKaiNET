package sk.ainet.context

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

public interface ExecutionContext {
    public val ops: TensorOps


    public val tensorDataFactory: TensorDataFactory

    public fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): Tensor<T, V> {
        val data = tensorDataFactory.full<T, V>(shape, dtype, value)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> zeros(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.zeros<T, V>(shape, dtype)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> ones(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.ones<T, V>(shape, dtype)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> fromData(data: TensorData<T, V>, dtype: KClass<T>): Tensor<T, V> = OpsBoundTensor(
        data, dtype,
        ops
    )

    public fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromFloatArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromIntArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }


    // runtime information
    public val memoryInfo: MemoryInfo
    public val executionStats: ExecutionStats
}
