package sk.ainet.io.mapper

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

interface ModelValuesMapper<T : DType, V> {
    fun mapToModel(model: Module<T, V>, wandb: Map<String, Tensor<T, V>>)
}