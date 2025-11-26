package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Stable, serializable identity for a tensor within a recording window/session.
 * The `id` must be unique per recording session; policy is owned by [TraceSession].
 */
public data class TensorRef(public val id: String)

/**
 * Minimal operation trace model as specified in exec-prd.md FR1.
 */
public data class OpTrace(
    val opType: String,
    val inputs: List<TensorRef>,
    val outputs: List<TensorRef>,
    val attributes: Map<String, Any?> = emptyMap()
)

/**
 * A lightweight session to convert runtime tensors to stable [TensorRef] ids
 * and optionally resolve them back for diagnostics within the same run.
 *
 * Notes:
 * - Keys are held strongly; intended for short-lived recording windows.
 * - ID policy: sequential IDs (t0, t1, ...), deterministic within the session.
 */
public class TraceSession {
    private val tensorToRef = mutableMapOf<Any, TensorRef>()
    private val refToTensor = mutableMapOf<String, Any>()
    private var nextId = 0

    /** Return existing or create a new TensorRef for the given tensor. */
    public fun <T : DType, V> refOf(tensor: Tensor<T, V>): TensorRef {
        return tensorToRef.getOrPut(tensor) {
            val id = "t${nextId++}"
            val ref = TensorRef(id)
            refToTensor[id] = tensor
            ref
        }
    }

    /** Batch conversion helper. */
    public fun <T : DType, V> refsOf(tensors: List<Tensor<T, V>>): List<TensorRef> = tensors.map { refOf(it) }

    /** Diagnostics helper: best-effort resolve a TensorRef back to the runtime tensor (if still present). */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> resolve(ref: TensorRef): Tensor<T, V>? = refToTensor[ref.id] as? Tensor<T, V>
}

/**
 * Attribute helpers for common ops: attaches shapes and dtypes.
 * Designed to be simple and serializable, usable by both sinks and offline conversions.
 */
public object OpAttributeFactory {
    /** Generic: capture shapes and dtypes for an arbitrary set of inputs/outputs. */
    public fun <T : DType, V> shapesAndDTypes(
        inputs: List<Tensor<T, V>>, outputs: List<Tensor<T, V>>
    ): Map<String, Any?> = mapOf(
        "inputShapes" to inputs.map { it.shape.dimensions.toList() },
        "outputShapes" to outputs.map { it.shape.dimensions.toList() },
        "inputDTypes" to inputs.map { it.dtype.simpleName() },
        "outputDTypes" to outputs.map { it.dtype.simpleName() }
    )

    /** Binary op convenience (e.g., add, mul). */
    public fun <T : DType, V> binary(
        a: Tensor<T, V>, b: Tensor<T, V>, out: Tensor<T, V>
    ): Map<String, Any?> = shapesAndDTypes(listOf(a, b), listOf(out))

    /** Unary op convenience (e.g., relu, sigmoid). */
    public fun <T : DType, V> unary(
        x: Tensor<T, V>, y: Tensor<T, V>
    ): Map<String, Any?> = shapesAndDTypes(listOf(x), listOf(y))

    /** Conv2d op attributes: shapes/dtypes + stride/padding/dilation/groups and bias flag. */
    public fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        out: Tensor<T, V>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Map<String, Any?> = shapesAndDTypes(listOf(input, weight) + listOfNotNull(bias), listOf(out)) + mapOf(
        "stride" to listOf(stride.first, stride.second),
        "padding" to listOf(padding.first, padding.second),
        "dilation" to listOf(dilation.first, dilation.second),
        "groups" to groups,
        "hasBias" to (bias != null)
    )
}

// Small helper to get a simple type name consistently across platforms
private fun <T : DType> kotlin.reflect.KClass<T>.simpleName(): String = this.simpleName ?: this.toString()
