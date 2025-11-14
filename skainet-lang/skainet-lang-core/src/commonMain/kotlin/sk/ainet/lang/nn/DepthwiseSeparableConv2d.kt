package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Depthwise Separable Convolution layer.
 * 
 * This layer applies a depthwise convolution followed by a pointwise convolution.
 * It's more efficient than standard convolution as it reduces the number of parameters
 * and computational complexity while maintaining similar representational power.
 * 
 * The operation is split into two parts:
 * 1. Depthwise convolution: applies a single filter per input channel (groups = inChannels)
 * 2. Pointwise convolution: 1x1 convolution to combine the depthwise outputs
 * 
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels/filters
 * @param kernelSize Size of the convolving kernel for depthwise convolution (height, width)
 * @param stride Stride of the convolution (default: 1, 1)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param dilation Spacing between kernel elements (default: 1, 1)
 * @param bias Whether to add learnable bias to both depthwise and pointwise layers (default: true)
 * @param name Name of the module
 * @param initDepthwiseWeights Initial weights for depthwise convolution
 * @param initPointwiseWeights Initial weights for pointwise convolution
 * @param initDepthwiseBias Initial bias for depthwise convolution (if bias is true)
 * @param initPointwiseBias Initial bias for pointwise convolution (if bias is true)
 */
public class DepthwiseSeparableConv2d<T : DType, V>(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = 1 to 1,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val dilation: Pair<Int, Int> = 1 to 1,
    public val bias: Boolean = true,
    override val name: String = "DepthwiseSeparableConv2d",
    initDepthwiseWeights: Tensor<T, V>,
    initPointwiseWeights: Tensor<T, V>,
    initDepthwiseBias: Tensor<T, V>? = null,
    initPointwiseBias: Tensor<T, V>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {
    
    init {
        require(inChannels > 0) { "inChannels must be positive" }
        require(outChannels > 0) { "outChannels must be positive" }
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
        require(dilation.first > 0 && dilation.second > 0) { "dilation must be positive" }
    }

    // Depthwise convolution: one filter per input channel
    private val depthwiseConv = Conv2d<T, V>(
        inChannels = inChannels,
        outChannels = inChannels, // Same number of output channels as input
        kernelSize = kernelSize,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = inChannels, // Key: groups = inChannels for depthwise
        bias = bias,
        name = "${name}.depthwise",
        initWeights = initDepthwiseWeights,
        initBias = initDepthwiseBias
    )
    
    // Pointwise convolution: 1x1 convolution to combine depthwise outputs
    private val pointwiseConv = Conv2d<T, V>(
        inChannels = inChannels,
        outChannels = outChannels,
        kernelSize = 1 to 1, // 1x1 kernel for pointwise
        stride = 1 to 1,
        padding = 0 to 0,
        dilation = 1 to 1,
        groups = 1, // Standard convolution for pointwise
        bias = bias,
        name = "${name}.pointwise",
        initWeights = initPointwiseWeights,
        initBias = initPointwiseBias
    )

    override val params: List<ModuleParameter<T, V>>
        get() = depthwiseConv.params + pointwiseConv.params

    override val modules: List<Module<T, V>>
        get() = listOf(depthwiseConv, pointwiseConv)

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            // First apply depthwise convolution
            val depthwiseOutput = depthwiseConv.forward(input, ctx)
            
            // Then apply pointwise convolution
            val pointwiseOutput = pointwiseConv.forward(depthwiseOutput, ctx)
            
            pointwiseOutput
        }

    /**
     * Calculates the output size for the entire depthwise separable convolution.
     * The output size is determined by the depthwise convolution since pointwise uses 1x1 kernels.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        return depthwiseConv.outputSize(inputSize)
    }

    /**
     * Returns the total number of parameters in this layer.
     * This is typically much less than a standard convolution with the same input/output dimensions.
     */
    public fun parameterCount(): Long {
        val depthwiseParams = inChannels * kernelSize.first * kernelSize.second + if (bias) inChannels else 0
        val pointwiseParams = inChannels * outChannels + if (bias) outChannels else 0
        return (depthwiseParams + pointwiseParams).toLong()
    }

    /**
     * Returns the parameter reduction ratio compared to a standard convolution.
     */
    public fun parameterReduction(): Double {
        val standardParams = inChannels * outChannels * kernelSize.first * kernelSize.second + if (bias) outChannels else 0
        val separableParams = parameterCount()
        return standardParams.toDouble() / separableParams.toDouble()
    }
}