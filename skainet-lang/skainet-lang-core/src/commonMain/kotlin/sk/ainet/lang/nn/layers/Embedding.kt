package sk.ainet.lang.nn.layers

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.NeuralNetworkExecutionContext
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.slice
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Embedding layer: maps integer token ids to dense vectors using a lookup table (weights).
 *
 * Notes/Constraints:
 * - Due to current Module<T,V> design, input and output dtypes must be the same T.
 *   Indices are therefore read from input tensor's data as Number and coerced to Int.
 *   Caller must ensure integer-valued contents by contract.
 * - Implements minimal gather using slice() + concat() provided by TensorOps.
 * - paddingIdx (if provided): rows equal to paddingIdx are zeroed by subtracting the row from itself.
 */
public class Embedding<T : DType, V>(
    public val numEmbeddings: Int,
    public val embeddingDim: Int,
    initWeight: Tensor<T, V>,
    public val paddingIdx: Int? = null,
    override val name: String = "Embedding"
) : Module<T, V>(), ModuleParameters<T, V> {

    public companion object {
        internal fun <T : DType, V> initWeight(
            ctx: NeuralNetworkExecutionContext,
            dtype: KClass<T>,
            numEmbeddings: Int,
            embeddingDim: Int,
            mean: Float,
            std: Float,
            random: Random
        ): Tensor<T, V> {
            val data = ctx.tensorDataFactory.randn<T, V>(Shape(numEmbeddings, embeddingDim), dtype, mean, std, random)
            return ctx.fromData(data, dtype)
        }
    }

    /** Convenience constructor taking [EmbeddingParams] with explicit weight. */
    public constructor(
        params: EmbeddingParams,
        initWeight: Tensor<T, V>,
        name: String = "Embedding"
    ) : this(
        numEmbeddings = params.numEmbeddings,
        embeddingDim = params.embeddingDim,
        initWeight = initWeight,
        paddingIdx = params.paddingIdx,
        name = name
    )

    /**
     * Default-initializing constructor: creates the weight tensor internally using a standard normal init.
     * Uses mean=0 and small std (0.1) similar to typical Linear initializers.
     */
    public constructor(
        ctx: NeuralNetworkExecutionContext,
        dtype: KClass<T>,
        numEmbeddings: Int,
        embeddingDim: Int,
        paddingIdx: Int? = null,
        name: String = "Embedding",
        mean: Float = 0.0f,
        std: Float = 0.1f,
        random: Random = Random.Default
    ) : this(
        numEmbeddings = numEmbeddings,
        embeddingDim = embeddingDim,
        initWeight = Companion.initWeight<T, V>(ctx, dtype, numEmbeddings, embeddingDim, mean, std, random),
        paddingIdx = paddingIdx,
        name = name
    )

    /** Params-based default-initializing constructor. */
    public constructor(
        ctx: NeuralNetworkExecutionContext,
        dtype: KClass<T>,
        params: EmbeddingParams,
        name: String = "Embedding",
        mean: Float = 0.0f,
        std: Float = 0.1f,
        random: Random = Random.Default
    ) : this(
        ctx = ctx,
        dtype = dtype,
        numEmbeddings = params.numEmbeddings,
        embeddingDim = params.embeddingDim,
        paddingIdx = params.paddingIdx,
        name = name,
        mean = mean,
        std = std,
        random = random
    )

    init {
        // Validate weight shape: [numEmbeddings, embeddingDim]
        val wShape = initWeight.shape.dimensions
        require(initWeight.rank == 2 && wShape[0] == numEmbeddings && wShape[1] == embeddingDim) {
            "Embedding($name): weight shape must be [numEmbeddings, embeddingDim]=[${numEmbeddings}, ${embeddingDim}], but was ${initWeight.shape}"
        }
        if (paddingIdx != null) {
            require(paddingIdx >= 0 && paddingIdx < numEmbeddings) {
                "Embedding($name): paddingIdx must be in [0, $numEmbeddings), was $paddingIdx"
            }
        }
    }

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", initWeight)
    )

    override val modules: List<Module<T, V>>
        get() = emptyList()

    private fun gatherRow(ops: TensorOps, weight: Tensor<T, V>, index: Int): Tensor<T, V> {
        // slice out a single row with shape [1, embeddingDim]
        val row = weight.slice<T, V>(listOf(
            Slice.Range<T, V>(index, index + 1),
            Slice.All<T, V>()
        ))
        // If padding, turn into zeros by row - row
        return if (paddingIdx != null && index == paddingIdx) {
            ops.subtract(row, row)
        } else row
    }

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        val weight = (params[0] as ModuleParameter.WeightParameter<T, V>).value
        val ops = weight.ops

        fun coerceToInt(v: Any?): Int {
            return when (v) {
                is Number -> {
                    val f = v.toFloat()
                    val i = v.toInt()
                    if (i.toFloat() != f) {
                        throw sk.ainet.lang.tensor.NonIntegralIndexException(
                            "Embedding($name): non-integral index value $f encountered. Migrate to integer tensors or call asIndices(strict=false)."
                        )
                    }
                    i
                }
                else -> error("Embedding($name): input values must be numeric indices, but got ${v?.let { it::class.simpleName }}")
            }
        }

        return when (input.rank) {
            // Unbatched: [L] -> [L, D]
            1 -> {
                val L = input.shape[0]
                val rows = ArrayList<Tensor<T, V>>(L)
                for (i in 0 until L) {
                    val idx = coerceToInt(input.data[i])
                    require(idx in 0 until numEmbeddings) {
                        "Embedding($name): index out of range at position $i: $idx not in [0, $numEmbeddings)"
                    }
                    rows += gatherRow(ops, weight, idx)
                }
                // concat along dim 0 to get [L, D]
                ops.concat(rows, 0)
            }
            // Batched: [N, L] -> [N, L, D]
            2 -> {
                val N = input.shape[0]
                val L = input.shape[1]
                val flatRows = ArrayList<Tensor<T, V>>(N * L)
                for (n in 0 until N) {
                    for (l in 0 until L) {
                        val idx = coerceToInt(input.data[n, l])
                        require(idx in 0 until numEmbeddings) {
                            "Embedding($name): index out of range at position ($n,$l): $idx not in [0, $numEmbeddings)"
                        }
                        flatRows += gatherRow(ops, weight, idx)
                    }
                }
                // [N*L, D] then reshape to [N, L, D]
                val concatenated = ops.concat(flatRows, 0)
                concatenated.reshape(Shape(N, L, embeddingDim))
            }
            else -> error("Embedding($name): input shape ${input.shape} not supported; expected [L] or [N, L]")
        }
    }
}
