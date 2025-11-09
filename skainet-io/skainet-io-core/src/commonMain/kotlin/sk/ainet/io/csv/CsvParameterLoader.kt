package sk.ainet.io.csv

import kotlinx.io.Source
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import sk.ainet.io.ParametersLoader
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

class JsonParametersLoader(private val handleSource: () -> Source) :
    ParametersLoader {
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        handleSource().use { source: Source ->
            val json = Json { ignoreUnknownKeys = true }
            val params: List<Parameter> = json.decodeFromString(source.readString())

            params.forEach { p ->
                val name = p.unique_parameter_name
                val t = p.tensor
                val shape = Shape(*t.shape.toIntArray())


                @Suppress("UNCHECKED_CAST")
                val tensor: Tensor<T, V> = when (dtype) {
                    FP32::class, FP16::class -> {
                        val floatValues = FloatArray(t.values.size) { i -> t.values[i].toFloat() }
                        ctx.fromFloatArray<T, Float>(shape, dtype, floatValues)
                    }

                    Int32::class -> {
                        val intValues = IntArray(t.values.size) { i -> t.values[i].toInt() }
                        ctx.fromIntArray<T, Int>(shape, dtype, intValues)
                    }

                    else -> error("Unsupported dtype for CsvParametersLoader: ${dtype.simpleName}")
                } as Tensor<T, V>

                onTensorLoaded(name, tensor)
            }
        }
    }
}