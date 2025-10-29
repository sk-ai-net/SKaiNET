package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.pprint
import sk.ainet.lang.types.DType


public fun <T : DType, V> Tensor<T, V>.pprint(): String = data.pprint()