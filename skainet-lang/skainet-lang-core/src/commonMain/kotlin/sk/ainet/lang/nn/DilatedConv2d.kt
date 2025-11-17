package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Dilated (Atrous) Convolution layer.
 * 
 * Dilated convolution introduces gaps (holes) between the kernel elements, effectively
 * increasing the receptive field without increasing the number of parameters or
 * computational cost. This is particularly useful for semantic segmentation and
 * other tasks where capturing multi-scale context is important.
 * 
 * The dilation parameter controls the spacing between kernel elements:
 * - dilation = (1, 1): standard convolution
 * - dilation = (2, 2): skip every other pixel
 * - dilation = (4, 4): skip every 4th pixel
 * 
 * Dilated convolution is also known as "atrous convolution" (from the French
 * word "à trous" meaning "with holes").
 * 
 * This is essentially a convenience wrapper around Conv2d with explicit dilation
 * handling and additional utility methods for dilated convolution operations.
 * 
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels/filters
 * @param kernelSize Size of the convolving kernel (height, width)
 * @param dilation Spacing between kernel elements (dilation rate)
 * @param stride Stride of the convolution (default: 1, 1)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param groups Number of groups for grouped dilated convolution (default: 1)
 * @param bias Whether to add a learnable bias to the output (default: true)
 * @param name Name of the module
 * @param initWeights Initial weights tensor
 * @param initBias Initial bias tensor (if bias is true)
 */
public class DilatedConv2d<T : DType, V>(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Pair<Int, Int>,
    public val dilation: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = 1 to 1,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val groups: Int = 1,
    public val bias: Boolean = true,
    override val name: String = "DilatedConv2d",
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

    // Dilated convolution is implemented using the base Conv2d with dilation parameter
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
     * Returns the effective kernel size considering dilation.
     * This is the size of the receptive field that the dilated kernel covers.
     */
    public fun effectiveKernelSize(): Pair<Int, Int> {
        val (kernelHeight, kernelWidth) = kernelSize
        val (dilationHeight, dilationWidth) = dilation
        
        val effectiveHeight = kernelHeight + (kernelHeight - 1) * (dilationHeight - 1)
        val effectiveWidth = kernelWidth + (kernelWidth - 1) * (dilationWidth - 1)
        
        return effectiveHeight to effectiveWidth
    }

    /**
     * Returns the receptive field size at the input level.
     * This shows how much of the input each output pixel can "see".
     */
    public fun receptiveFieldSize(): Pair<Int, Int> = effectiveKernelSize()

    /**
     * Returns the total number of parameters in this layer.
     * Dilation doesn't change the parameter count compared to standard convolution.
     */
    public fun parameterCount(): Long {
        val weightParams = (inChannels / groups) * (outChannels / groups) * kernelSize.first * kernelSize.second * groups
        val biasParams = if (bias) outChannels else 0
        return (weightParams + biasParams).toLong()
    }

    /**
     * Returns the receptive field expansion ratio compared to standard convolution.
     * This shows how much larger the receptive field is due to dilation.
     */
    public fun receptiveFieldExpansion(): Double {
        val (standardHeight, standardWidth) = kernelSize
        val (effectiveHeight, effectiveWidth) = effectiveKernelSize()
        
        val standardArea = standardHeight * standardWidth
        val effectiveArea = effectiveHeight * effectiveWidth
        
        return effectiveArea.toDouble() / standardArea.toDouble()
    }

    /**
     * Checks if this is a standard convolution (dilation == (1, 1)).
     */
    public fun isStandard(): Boolean = dilation == 1 to 1

    /**
     * Returns information about the dilated convolution configuration.
     */
    public fun dilationInfo(): DilatedConvInfo {
        val effectiveKernel = effectiveKernelSize()
        return DilatedConvInfo(
            dilation = dilation,
            kernelSize = kernelSize,
            effectiveKernelSize = effectiveKernel,
            receptiveFieldSize = effectiveKernel,
            receptiveFieldExpansion = receptiveFieldExpansion(),
            parameterCount = parameterCount(),
            isStandard = isStandard()
        )
    }

    /**
     * Creates a multi-scale dilated convolution block with different dilation rates.
     * This is commonly used in semantic segmentation (e.g., DeepLab's ASPP).
     */
    public companion object {
        /**
         * Creates multiple dilated convolutions with different dilation rates
         * for multi-scale context aggregation.
         */
        public fun <T : DType, V> createMultiScale(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Pair<Int, Int>,
            dilations: List<Pair<Int, Int>>,
            stride: Pair<Int, Int> = 1 to 1,
            padding: Pair<Int, Int> = 0 to 0,
            groups: Int = 1,
            bias: Boolean = true,
            namePrefix: String = "DilatedConv2d",
            weightsFactory: (Int) -> Tensor<T, V>,
            biasFactory: ((Int) -> Tensor<T, V>)? = null
        ): List<DilatedConv2d<T, V>> {
            return dilations.mapIndexed { index, dilation ->
                DilatedConv2d<T, V>(
                    inChannels = inChannels,
                    outChannels = outChannels,
                    kernelSize = kernelSize,
                    dilation = dilation,
                    stride = stride,
                    padding = padding,
                    groups = groups,
                    bias = bias,
                    name = "${namePrefix}_dil_${dilation.first}x${dilation.second}",
                    initWeights = weightsFactory(index),
                    initBias = biasFactory?.invoke(index)
                )
            }
        }
    }
}

/**
 * Data class containing information about dilated convolution configuration.
 */
public data class DilatedConvInfo(
    val dilation: Pair<Int, Int>,
    val kernelSize: Pair<Int, Int>,
    val effectiveKernelSize: Pair<Int, Int>,
    val receptiveFieldSize: Pair<Int, Int>,
    val receptiveFieldExpansion: Double,
    val parameterCount: Long,
    val isStandard: Boolean
)