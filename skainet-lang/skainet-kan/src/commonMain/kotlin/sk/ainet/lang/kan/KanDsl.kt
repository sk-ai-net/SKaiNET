package sk.ainet.lang.kan

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.NetworkDsl
import sk.ainet.lang.nn.dsl.NetworkDslItem
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.BiasScope
import sk.ainet.lang.nn.dsl.BiasScopeImpl
import sk.ainet.lang.nn.dsl.WeightsScope
import sk.ainet.lang.nn.dsl.WeightsScopeImpl
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * DSL surface for configuring a Kolmogorov–Arnold Network layer.
 */
@NetworkDsl
public interface KAN<T : DType, V> : NetworkDslItem {
    public var outputDim: Int
    public var gridSize: Int
    public var degree: Int
    public var useBias: Boolean
    public var useResidual: Boolean
    public var baseActivation: (Tensor<T, V>) -> Tensor<T, V>
    public fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>)
    public fun basis(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>)
    public fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>)
}

public class KanDslImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    private val dtype: KClass<T>,
    initialOutputDim: Int,
    private val id: String
) : KAN<T, V> {
    override var outputDim: Int = initialOutputDim
    override var gridSize: Int = 16
    override var degree: Int = 3
    override var useBias: Boolean = true
    override var useResidual: Boolean = false
    override var baseActivation: (Tensor<T, V>) -> Tensor<T, V> = { it }
    public var weightsInit: WeightsScope<T, V>.(Shape) -> Tensor<T, V> = { randn(std = 0.02f) }
    public var basisInit: WeightsScope<T, V>.(Shape) -> Tensor<T, V> = { uniform(min = -0.5f, max = 0.5f) }
    public var biasInit: BiasScope<T, V>.(Shape) -> Tensor<T, V> = { zeros() }

    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        weightsInit = initBlock
    }

    override fun basis(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        basisInit = initBlock
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        biasInit = initBlock
    }

    public fun create(
        inputDim: Int,
        mixingWeights: Tensor<T, V>,
        basis: Tensor<T, V>,
        bias: Tensor<T, V>?
    ): KanLayer<T, V> = KanLayer(
        config = KanConfig(
            inputDim = inputDim,
            outputDim = outputDim,
            gridSize = gridSize,
            degree = degree,
            useBias = useBias,
            useResidual = useResidual
        ),
        baseActivation = baseActivation,
        initMixingWeights = mixingWeights,
        initBasis = basis,
        initBias = bias,
        name = id
    )
}

/**
 * Adds a KAN layer to the existing network DSL.
 *
 * Note: The current implementation expands inputs with a learnable basis per
 * feature and mixes them via a linear projection. The `degree` parameter is
 * reserved for future spline/basis variants and is not yet used.
 */
public inline fun <reified T : DType, V> NeuralNetworkDsl<T, V>.kanLayer(
    outputDim: Int,
    gridSize: Int = 16,
    degree: Int = 3,
    useBias: Boolean = true,
    useResidual: Boolean = false,
    id: String = "",
    noinline baseActivation: (Tensor<T, V>) -> Tensor<T, V> = { it },
    content: KAN<T, V>.() -> Unit = {}
) {
    val resolvedId = id.ifEmpty {
        when (this) {
            is NeuralNetworkDslImpl<*, *> -> "kan-${this.modules.size}"
            is StageImpl<*, *> -> "kan-${this.modules.size}"
            else -> "kan"
        }
    }
    val impl = KanDslImpl<T, V>(
        executionContext = executionContext,
        dtype = T::class,
        initialOutputDim = outputDim,
        id = resolvedId
    )
    impl.gridSize = gridSize
    impl.degree = degree
    impl.useBias = useBias
    impl.useResidual = useResidual
    impl.baseActivation = baseActivation
    impl.content()

    require(outputDim > 0) { "KAN layer requires outputDim > 0." }
    require(gridSize > 0) { "KAN layer requires gridSize > 0." }
    val inputDim = when (this) {
        is NeuralNetworkDslImpl<T, V> -> this.lastDimension
        is StageImpl<T, V> -> this.lastDimension
        else -> error("KAN layers are supported only on the default SKaiNET DSL implementations")
    }
    require(inputDim > 0) { "KAN layer requires a known input dimension; call input(...) before kanLayer()." }

    val weightsShape = Shape(outputDim, inputDim * gridSize)
    val basisShape = Shape(inputDim, gridSize)
    val biasShape = Shape(outputDim)
    val weightsScope = WeightsScopeImpl<T, V>(executionContext, weightsShape, T::class)
    val weightsTensor = impl.weightsInit.invoke(weightsScope, weightsShape)
    val basisScope = WeightsScopeImpl<T, V>(executionContext, basisShape, T::class)
    val basisTensor = impl.basisInit.invoke(basisScope, basisShape)
    val biasTensor = if (impl.useBias) {
        val biasScope = BiasScopeImpl<T, V>(executionContext, biasShape, T::class)
        impl.biasInit.invoke(biasScope, biasShape)
    } else null

    when (this) {
        is NeuralNetworkDslImpl<T, V> -> {
            this.modules += impl.create(inputDim, weightsTensor, basisTensor, biasTensor)
            this.lastDimension = outputDim
        }

        is StageImpl<T, V> -> {
            this.modules += impl.create(inputDim, weightsTensor, basisTensor, biasTensor)
            this.lastDimension = outputDim
        }

        else -> error("KAN layers are supported only on the default SKaiNET DSL implementations")
    }
}
