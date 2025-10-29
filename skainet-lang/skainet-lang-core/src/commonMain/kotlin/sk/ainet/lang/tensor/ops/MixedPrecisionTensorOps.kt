package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

public interface MixedPrecisionTensorOps<V> : TensorOps {
    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V>
    
    public fun <T1 : DType, T2 : DType> addMixed(
        a: Tensor<T1, V>,
        b: Tensor<T2, V>
    ): Tensor<out DType, V>
}