package sk.ainet.lang.kan

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.dsl.BiasScope
import sk.ainet.lang.nn.dsl.BiasScopeImpl
import sk.ainet.lang.nn.dsl.WeightsScope
import sk.ainet.lang.nn.dsl.WeightsScopeImpl
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Public NN module alias for Kolmogorov–Arnold Network (AKN).
 * This exposes the same implementation the DSL uses (KanLayer) for direct creation without the DSL.
 */
public typealias AknConfig = KanConfig
public typealias Akn<T, V> = KanLayer<T, V>

/**
 * Factory to create an AKN module directly (without the kanLayer DSL helper).
 * Mirrors defaults and initializer hooks used by the DSL.
 */
public fun <T : DType, V> createAkn(
    executionContext: ExecutionContext,
    dtype: KClass<T>,
    inputDim: Int,
    outputDim: Int,
    gridSize: Int = 16,
    degree: Int = 3,
    useBias: Boolean = true,
    useResidual: Boolean = false,
    name: String = "akn",
    baseActivation: (Tensor<T, V>) -> Tensor<T, V> = { it },
    weightsInit: WeightsScope<T, V>.(Shape) -> Tensor<T, V> = { randn(std = 0.02f) },
    basisInit: WeightsScope<T, V>.(Shape) -> Tensor<T, V> = { uniform(min = -0.5f, max = 0.5f) },
    biasInit: BiasScope<T, V>.(Shape) -> Tensor<T, V> = { zeros() }
): Akn<T, V> {
    require(outputDim > 0) { "AKN requires outputDim > 0." }
    require(gridSize > 0) { "AKN requires gridSize > 0." }
    require(inputDim > 0) { "AKN requires inputDim > 0." }

    val weightsShape = Shape(outputDim, inputDim * gridSize)
    val basisShape = Shape(inputDim, gridSize)
    val biasShape = Shape(outputDim)

    val wScope = WeightsScopeImpl<T, V>(executionContext, weightsShape, dtype)
    val bScope = WeightsScopeImpl<T, V>(executionContext, basisShape, dtype)
    val weights = weightsInit.invoke(wScope, weightsShape)
    val basis = basisInit.invoke(bScope, basisShape)
    val bias = if (useBias) {
        val biasScope = BiasScopeImpl<T, V>(executionContext, biasShape, dtype)
        biasInit.invoke(biasScope, biasShape)
    } else null

    val config = KanConfig(
        inputDim = inputDim,
        outputDim = outputDim,
        gridSize = gridSize,
        degree = degree,
        useBias = useBias,
        useResidual = useResidual
    )

    return KanLayer(
        config = config,
        baseActivation = baseActivation,
        initMixingWeights = weights,
        initBasis = basis,
        initBias = bias,
        name = name
    )
}
