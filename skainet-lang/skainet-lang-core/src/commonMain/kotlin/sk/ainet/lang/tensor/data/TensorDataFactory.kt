package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Interface for tensor data factory used by the DSL
 */
public interface TensorDataFactory {
    public fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V>
    public fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): TensorData<T, V>
    public fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): TensorData<T, V>
    public fun <T : DType, V> randn(
        shape: Shape,
        dtype: KClass<T>,
        mean: Float,
        std: Float,
        random: Random
    ): TensorData<T, V>

    public fun <T : DType, V> uniform(
        shape: Shape,
        dtype: KClass<T>,
        min: Float,
        max: Float,
        random: Random
    ): TensorData<T, V>

    public fun <T : DType, V> init(
        shape: Shape,
        dtype: KClass<T>,
        generator: (indices: IntArray) -> V
    ): TensorData<T, V>

    public fun <T : DType, V> randomInit(
        shape: Shape,
        dtype: KClass<T>,
        generator: (random: Random) -> V,
        random: Random
    ): TensorData<T, V>

    public fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): TensorData<T, V>

    public fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): TensorData<T, V>

    public fun <T : DType, V> fromByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): TensorData<T, V>
}

/**
 * Global registry for tensor data factories, enabling factory management for different precision types.
 *
 * This registry implements Task 1.2.3: Add factory management for different precision types.
 */
public object TensorFactoryRegistry {
    private val factories = mutableMapOf<DType, TensorDataFactory>()

    public fun <T : DType, V> registerFactory(dtype: T, factory: TensorDataFactory) {
        factories[dtype] = factory
    }

    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> getFactory(dtype: T): TensorDataFactory {
        return factories[dtype] as? TensorDataFactory
            ?: throw IllegalArgumentException("No factory registered for dtype: ${dtype.name}")
    }

    public fun hasFactory(dtype: DType): Boolean = factories.containsKey(dtype)
}
