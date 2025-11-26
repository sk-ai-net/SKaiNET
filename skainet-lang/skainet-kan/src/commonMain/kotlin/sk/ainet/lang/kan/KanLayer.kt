package sk.ainet.lang.kan

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.topology.bias
import sk.ainet.lang.nn.topology.weights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.PI
import kotlin.math.max

/**
 * Configuration for a Kolmogorov–Arnold Network layer.
 */
public data class KanConfig(
    val inputDim: Int,
    val outputDim: Int,
    val gridSize: Int = 16,
    val degree: Int = 3,
    val useBias: Boolean = true,
    val useResidual: Boolean = false,
    val regularization: KanRegularization = KanRegularization(),
    val gridMin: Float = 0.0f,
    val gridMax: Float = (PI / 2).toFloat()
)

/**
 * Simple regularization hints for future spline/mixing penalties.
 */
public data class KanRegularization(
    val splineSmoothness: Double = 0.0,
    val sparsity: Double = 0.0
)

/**
 * Stub implementation for a Kolmogorov–Arnold Network layer.
 *
 * The forward path intentionally throws until the spline basis and mixing
 * logic are implemented. The class still wires into the DSL so models can
 * be composed before kernels land.
 */
public class KanLayer<T : DType, V>(
    public val config: KanConfig,
    public val baseActivation: (Tensor<T, V>) -> Tensor<T, V>,
    initMixingWeights: Tensor<T, V>,
    initBasis: Tensor<T, V>,
    initBias: Tensor<T, V>?,
    override val name: String
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = listOfNotNull(
        ModuleParameter.WeightParameter("$name.mixing_weight", initMixingWeights),
        ModuleParameter.WeightParameter("$name.basis", initBasis),
        initBias?.let { ModuleParameter.BiasParameter("$name.bias", it) }
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = input.ops
        val weightParams = params.filterIsInstance<ModuleParameter.WeightParameter<T, V>>()
        val mixing = weightParams.first { it.name.contains("mixing", ignoreCase = true) }.value
        val basis = weightParams.first { it.name.contains("basis", ignoreCase = true) }.value
        val bias = params.filterIsInstance<ModuleParameter.BiasParameter<T, V>>().firstOrNull()?.value

        // Collapse higher-rank inputs to [batch, features] to align with the mixing step.
        val base = when (input.rank) {
            1 -> ops.unsqueeze(input, 0)
            2 -> input
            else -> ops.flatten(input, startDim = 1, endDim = -1)
        }

        // Hat basis evaluation per input dimension:
        val batch = base.shape.dimensions[0]
        val step = if (config.gridSize > 1) (config.gridMax - config.gridMin) / (config.gridSize - 1) else config.gridMax - config.gridMin
        val centersData = FloatArray(config.inputDim * config.gridSize) { idx ->
            val g = idx % config.gridSize
            config.gridMin + step * g
        }
        val centers = ctx.fromFloatArray<T, V>(Shape(config.inputDim, config.gridSize), basis.dtype, centersData)

        val baseExp = ops.unsqueeze(base, base.rank) // [batch, in, 1]
        val centersExp = ops.unsqueeze(centers, dim = 0) // [1, in, grid]
        val diff = ops.subtract(baseExp, centersExp)
        val absDiff = ops.add(ops.relu(diff), ops.relu(ops.subtract(centersExp, baseExp)))

        val widthScalar = max(step, 1e-6f)
        val widthTensor = ctx.full<T, V>(Shape(1, 1, 1), basis.dtype, widthScalar)
        val absOverW = ops.divide(absDiff, widthTensor)
        val oneTensor = ctx.full<T, V>(Shape(1, 1, 1), basis.dtype, 1.0f)
        val hat = ops.relu(ops.subtract(oneTensor, absOverW)) // [batch, in, grid]

        val basisExp = ops.unsqueeze(basis, dim = 0)
        val weightedHat = ops.multiply(hat, basisExp)
        val expanded = ops.reshape(weightedHat, Shape(batch, config.inputDim * config.gridSize))

        val mixed = ops.matmul(expanded, ops.transpose(mixing))
        val biased = if (bias != null) {
            val biasAdjusted = if (bias.rank == 1) ops.unsqueeze(bias, 0) else bias
            ops.add(mixed, biasAdjusted)
        } else {
            mixed
        }
        val activated = baseActivation(biased)

        return when {
            input.rank == 1 -> ops.squeeze(activated, dim = 0)
            config.useResidual && config.outputDim == config.inputDim -> ops.add(activated, base)
            else -> activated
        }
    }
}
