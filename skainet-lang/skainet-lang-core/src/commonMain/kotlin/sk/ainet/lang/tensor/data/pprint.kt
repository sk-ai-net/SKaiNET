package sk.ainet.lang.tensor.data

import sk.ainet.lang.types.DType

private fun <T : DType, V> TensorData<T, V>.printMatrix(): String {
    val rows = this.shape[0]
    val cols = this.shape[1]
    val sb = StringBuilder()
    sb.append("[")
    for (i in 0 until rows) {
        if (i > 0) sb.append("\n ") // newline + space for readability
        sb.append("[ ")
        for (j in 0 until cols) {
            if (j > 0) sb.append(", ")
            sb.append(this[i, j].toString())
        }
        sb.append(" ]")
        if (i == rows - 1) sb.append("]")
    }
    return sb.toString()
}


public fun <T : DType, V> TensorData<T, V>.pprint(): String {
    return when (this.shape.rank) {
        0 -> {
            // Scalar - single value
            this.toString()
        }

        1 -> {
            // Vector - horizontal representation with parentheses
            val sb = StringBuilder()
            sb.append("[ ")
            for (i in 0 until this.shape[0]) {
                if (i > 0) sb.append(", ")
                sb.append(this[i].toString())
            }
            sb.append(" ]")
            sb.toString()
        }

        2 -> {
            printMatrix()
        }

        else -> {
            // Higher rank tensors - use toString
            this.toString()
        }
    }
}