package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Grouped Convolution layer.
 * 
 * In grouped convolution, the input channels are divided into groups, and each group
 * is convolved separately with its own set of filters. This reduces the number of
 * parameters and computational complexity while potentially improving feature learning
 * by encouraging different groups to learn different types of features.
 * 
 * When groups = 1, this is equivalent to standard convolution.
 * When groups = inChannels, this is equivalent to depthwise convolution.
 * 
 * This is essentially a convenience wrapper around Conv2d with explicit group handling
 * and additional utility methods for grouped convolution operations.
 * 
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels/filters
 * @param kernelSize Size of the convolving kernel (height, width)
 * @param groups Number of groups to divide channels into
 * @param stride Stride of the convolution (default: 1, 1)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param dilation Spacing between kernel elements (default: 1, 1)
 * @param bias Whether to add a learnable bias to the output (default: true)
 * @param name Name of the module
 * @param initWeights Initial weights tensor
 * @param initBias Initial bias tensor (if bias is true)
 */
public class GroupedConv2d<T : DType, V>(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Pair<Int, Int>,
    public val groups: Int,
    public val stride: Pair<Int, Int> = 1 to 1,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val dilation: Pair<Int, Int> = 1 to 1,
    public val bias: Boolean = true,
    override val name: String = "GroupedConv2d",
    initWeights: Tensor<T, V>,
    initBias: Tensor<T, V>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {
    
    init {
        require(inChannels > 0) { "inChannels must be positive" }
        require(outChannels > 0) { "outChannels must be positive" }
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
        require(dilation.first > 0 && dilation.second > 0) { "dilation must be positive" }
        require(groups > 0) { "groups must be positive" }
        require(inChannels % groups == 0) { "inChannels must be divisible by groups" }
        require(outChannels % groups == 0) { "outChannels must be divisible by groups" }
    }

    // Grouped convolution is implemented using the base Conv2d with groups parameter
    private val conv = Conv2d<T, V>(
        inChannels = inChannels,
        outChannels = outChannels,
        kernelSize = kernelSize,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        bias = bias,
        name = name,
        initWeights = initWeights,
        initBias = initBias
    )

    override val params: List<ModuleParameter<T, V>>
        get() = conv.params

    override val modules: List<Module<T, V>>
        get() = listOf(conv)

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            conv.forward(input, ctx)
        }

    /**
     * Calculates the output size for a given input size.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        return conv.outputSize(inputSize)
    }

    /**
     * Returns the number of input channels per group.
     */
    public fun inputChannelsPerGroup(): Int = inChannels / groups

    /**
     * Returns the number of output channels per group.
     */
    public fun outputChannelsPerGroup(): Int = outChannels / groups

    /**
     * Returns the total number of parameters in this layer.
     */
    public fun parameterCount(): Long {
        val weightParams = (inChannels / groups) * (outChannels / groups) * kernelSize.first * kernelSize.second * groups
        val biasParams = if (bias) outChannels else 0
        return (weightParams + biasParams).toLong()
    }

    /**
     * Returns the parameter reduction ratio compared to a standard convolution (groups = 1).
     */
    public fun parameterReduction(): Double {
        val standardParams = inChannels * outChannels * kernelSize.first * kernelSize.second + if (bias) outChannels else 0
        val groupedParams = parameterCount()
        return standardParams.toDouble() / groupedParams.toDouble()
    }

    /**
     * Returns the computational complexity reduction ratio compared to standard convolution.
     * This is the same as parameter reduction for convolution operations.
     */
    public fun computationalReduction(): Double = parameterReduction()

    /**
     * Checks if this is a depthwise convolution (groups == inChannels).
     */
    public fun isDepthwise(): Boolean = groups == inChannels

    /**
     * Checks if this is a standard convolution (groups == 1).
     */
    public fun isStandard(): Boolean = groups == 1

    /**
     * Returns information about the grouped convolution configuration.
     */
    public fun groupInfo(): GroupedConvInfo {
        return GroupedConvInfo(
            groups = groups,
            inputChannelsPerGroup = inputChannelsPerGroup(),
            outputChannelsPerGroup = outputChannelsPerGroup(),
            parameterCount = parameterCount(),
            parameterReduction = parameterReduction(),
            isDepthwise = isDepthwise(),
            isStandard = isStandard()
        )
    }
}

/**
 * Data class containing information about grouped convolution configuration.
 */
public data class GroupedConvInfo(
    val groups: Int,
    val inputChannelsPerGroup: Int,
    val outputChannelsPerGroup: Int,
    val parameterCount: Long,
    val parameterReduction: Double,
    val isDepthwise: Boolean,
    val isStandard: Boolean
)