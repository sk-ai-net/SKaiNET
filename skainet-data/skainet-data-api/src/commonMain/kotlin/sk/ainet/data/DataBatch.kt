package sk.ainet.data

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public data class DataBatch<T : DType, V>(val x: Array<Tensor<T, V>>, val y: Tensor<T, V>) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DataBatch<*, *>

        if (!x.contentEquals(other.x)) return false
        if (y != other.y) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.contentHashCode()
        result = 31 * result + y.hashCode()
        return result
    }
}
