package sk.ainet.lang.tensor.dsl

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

@TensorDsl
public interface TensorsValueScope<T : DType, V> {
    public val shape: Shape
    public val dtype: KClass<T>
    public val executionContext: ExecutionContext

    // Basic fills
    public fun zeros(): Tensor<T, V> = executionContext.zeros(shape, dtype)
    public fun ones(): Tensor<T, V> = executionContext.ones(shape, dtype)
    public fun full(value: Number): Tensor<T, V> = executionContext.full(shape, dtype, value)

    // Factories from primitive arrays
    public fun from(vararg data: Float): Tensor<T, V> = fromArray(data.toTypedArray().toFloatArray())
    public fun fromList(data: List<Float>): Tensor<T, V> = fromArray(data.toFloatArray())
    public fun fromArray(data: FloatArray): Tensor<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        return executionContext.fromFloatArray(shape, dtype, data)
    }

    public fun from(vararg data: Int): Tensor<T, V> = fromArray(data.toTypedArray().toIntArray())
    public fun fromIntList(data: List<Int>): Tensor<T, V> = fromArray(data.toIntArray())
    public fun fromArray(data: IntArray): Tensor<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        return executionContext.fromIntArray(shape, dtype, data)
    }

    // Custom initializers
    public fun init(generator: (indices: IntArray) -> V): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.init(shape, dtype, generator)
        return executionContext.fromData(data, dtype)
    }

    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.randomInit(shape, dtype, generator, random)
        return executionContext.fromData(data, dtype)
    }

    /** Advanced initialization with custom random distribution. */
    public fun random(initBlock: (Shape) -> Tensor<T, V>): Tensor<T, V> = initBlock(shape)

    // Distributions
    public fun randn(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.randn<T, V>(shape, dtype, mean, std, random)
        return executionContext.fromData(data, dtype)
    }

    public fun uniform(min: Float = 0.0f, max: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.uniform<T, V>(shape, dtype, min, max, random)
        return executionContext.fromData(data, dtype)
    }
}