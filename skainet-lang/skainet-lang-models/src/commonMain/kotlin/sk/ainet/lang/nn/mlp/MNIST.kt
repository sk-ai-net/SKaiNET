package sk.ainet.lang.nn.mlp

import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
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
public class MnistMpl() : Model {

    public override fun <T : DType, V> model(): Module<FP32, Float> = model
    override fun modelCard(): String {
        TODO("Not yet implemented")
    }


    private val model = definition<FP32, Float> {
        network {
            sequential {
                // Note: CNN layers (conv2d, maxPool2d) are not yet implemented in the DSL
                // This is a simplified MLP version for MNIST classification
                stage("input") {
                    flatten() // Flatten 28x28 input to 784
                }
                stage("hidden1") {
                    dense(128) {
                        activation = { tensor -> with(tensor) { relu() } }
                    }
                }
                stage("hidden2") {
                    dense(64) {
                        activation = { tensor -> with(tensor) { relu() } }
                    }
                }
                stage("output") {
                    dense(10) {
                        activation = { tensor -> with(tensor) { softmax(1) } }
                    }
                }
            }
        }
    }
}

