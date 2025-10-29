package sk.ainet.lang.tensor

public data class Shape(val dimensions: IntArray) {
    public companion object {
        public operator fun invoke(vararg dimensions: Int): Shape {
            return Shape(dimensions.copyOf())
        }
    }

    val volume: Int
        get() = dimensions.fold(1) { a, x -> a * x }

    val rank: Int
        get() = dimensions.size

    public fun index(indices: IntArray): Int {
        assert(
            { indices.size == dimensions.size },
            { "`indices.size` must be ${dimensions.size}: ${indices.size}" })
        return dimensions.zip(indices).fold(0) { a, x ->
            assert(
                { 0 <= x.second && x.second < x.first },
                { "Illegal index: indices = ${indices}, shape = $dimensions" })
            a * x.first + x.second
        }
    }

    public operator fun get(index: Int): Int {
        return dimensions[index]
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Shape) return false

        return dimensions.contentEquals(other.dimensions)
    }

    override fun hashCode(): Int {
        return dimensions.contentHashCode()
    }

    override fun toString(): String {
        // Create a string representation of the dimensions array
        val dimensionsString = dimensions.joinToString(separator = " x ", prefix = "[", postfix = "]")
        // Return the formatted string including dimensions and volume
        return "Shape: Dimensions = $dimensionsString, Size (Volume) = $volume"
    }
}

internal inline fun assert(value: () -> Boolean, lazyMessage: () -> Any) {
    if (!value()) {
        val message = lazyMessage()
        throw AssertionError(message)
    }
}