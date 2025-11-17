package sk.ainet.lang.model.dnn

public data class WeightAndBias(val name: String, val weights: FloatArray, val bias: FloatArray) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as WeightAndBias

        if (name != other.name) return false
        if (!weights.contentEquals(other.weights)) return false
        if (!bias.contentEquals(other.bias)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + weights.contentHashCode()
        result = 31 * result + bias.contentHashCode()
        return result
    }
}