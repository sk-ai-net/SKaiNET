package sk.ainet.lang.nn.cnn

import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.network

import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Constructs a Convolutional Neural Network (CNN) tailored for the MNIST dataset using a DSL-based network builder.
 *
 * This model consists of two convolutional blocks followed by a flattening stage and two dense (fully connected) layers.
 * It is designed to classify handwritten digits (0–9) from grayscale 28x28 pixel images.
 *
 * The architecture is as follows:
 *
 * - **Stage: "conv1"**
 *   - 2D Convolution with:
 *     - 16 output channels
 *     - 5x5 kernel
 *     - stride of 1
 *     - padding of 2
 *   - ReLU activation
 *   - 2x2 MaxPooling with stride of 2
 *
 * - **Stage: "conv2"**
 *   - 2D Convolution with:
 *     - 32 output channels
 *     - 5x5 kernel
 *     - stride of 1
 *     - padding of 2
 *   - ReLU activation
 *   - 2x2 MaxPooling with stride of 2
 *
 * - **Stage: "flatten"**
 *   - Flattens the tensor for dense layer input
 *
 * - **Stage: "dense"**
 *   - Fully connected layer with 128 units
 *   - ReLU activation
 *
 * - **Stage: "output"**
 *   - Fully connected layer with 10 output units (for 10 MNIST classes)
 *   - Softmax activation over dimension 1 to produce class probabilities
 *
 * @return A [Module] representing the constructed CNN model
 */
public class MnistCnn() : Model {

    public override fun <T : DType, V> model(): Module<FP32, Float> = model
    override fun modelCard(): String {
        TODO("Not yet implemented")
    }


    private val model = sk.ainet.lang.nn.definition<FP32, Float> {
        network {
            sequential {
                // Stage: "conv1"
                stage("conv1") {
                    conv2d(outChannels = 16, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu1") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }
                
                // Stage: "conv2"  
                stage("conv2") {
                    conv2d(outChannels = 32, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu2") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }
                
                // Stage: "flatten"
                stage("flatten") {
                    flatten()
                }
                
                // Stage: "dense"
                stage("dense") {
                    dense(outputDimension = 128)
                    activation(id = "relu3") { tensor -> tensor.relu() }
                }
                
                // Stage: "output"
                stage("output") {
                    dense(outputDimension = 10)
                    activation(id = "softmax") { tensor -> tensor.softmax(dim = 1) }
                }
            }
        }
    }
}

