package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

public interface TensorFactory<T: DType, V> {
    public fun constructTensor(data:T): Tensor<T, V>
    public fun zeros(shape:Shape): Tensor<T, V>
    public fun ones(shape:Shape): Tensor<T, V>

}