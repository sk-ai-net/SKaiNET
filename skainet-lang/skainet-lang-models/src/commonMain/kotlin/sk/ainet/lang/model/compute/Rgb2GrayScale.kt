package sk.ainet.lang.model.compute

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.types.FP32

/**
 * Color Space Transformation: RGB → Grayscale using fixed luminance weights.
 *
 * Implements the equivalent of NumPy code:
 *   weights = [0.2989, 0.5870, 0.1140]
 *   gray = (H*W x 3) @ weights → reshape(H, W)
 *
 * In SKainNET DSL this is realized as a 1x1 Conv2D with inChannels=3 and outChannels=1
 * whose kernel is fixed to the weights above and with no bias.
 *
 * Expected input shape: (N, 3, H, W)
 * Output shape: (N, 1, H, W)
 */
public class Rgb2GrayScale : Model<FP32, Float> {


    private val modelImpl: Module<FP32, Float> = definition {
        network {
            sequential {
                // 1x1 convolution performing per-pixel dot product with RGB weights
                conv2d(
                    outChannels = 1,
                    kernelSize = 1 to 1,
                    stride = 1 to 1,
                    padding = 0 to 0,
                    dilation = 1 to 1,
                    groups = 1,
                    bias = false,
                    id = "rgb2gray"
                ) {
                    inChannels = 3
                    // weights shape = (out=1, in=3, kH=1, kW=1)
                    weights { shape ->
                        fromArray(floatArrayOf(0.2989f, 0.5870f, 0.1140f))
                    }
                }
            }
        }
    }

    override fun model(executionContext: ExecutionContext): Module<FP32, Float> = modelImpl
    override fun modelCard(): ModelCard {
        TODO("Not yet implemented")
    }


    /*
    override fun modelCard(): String =
        "RGB→Grayscale via fixed 1x1 Conv2D with weights [0.2989, 0.5870, 0.1140] and no bias. " +
                "Input: (N,3,H,W). Output: (N,1,H,W)."

     */
}