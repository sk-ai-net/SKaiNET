package sk.ainet.lang.nn.mlp

import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

public class SinusApproximator() : Model {
    public override fun <T : DType, V> model(): Module<FP32, Float> = model


    private val model = definition<FP32, Float> {
        network {
            input(1)  // Single input for x value

            // First hidden layer: 1 -> 16 neurons
            dense(16) {
                // Weights: 16x1 matrix - explicitly defined values
                weights {
                    from(
                        0.5f, -0.3f, 0.8f, -0.2f, 0.6f, -0.4f, 0.7f, -0.1f,
                        0.9f, -0.5f, 0.3f, -0.7f, 0.4f, -0.6f, 0.2f, -0.8f
                    )
                }

                // Bias: 16 values - explicitly defined
                bias {
                    from(
                        0.1f, -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f,
                        -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f, -0.1f
                    )
                }

                activation = { tensor -> with(tensor) { relu() } }
            }

            // Second hidden layer: 16 -> 16 neurons
            dense(16) {
                // Weights: 16x16 matrix - explicitly defined values
                weights {
                    from(
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.2f,
                        -0.1f,
                        0.5f
                    )
                }

                // Bias: 16 values - explicitly defined
                bias {
                    from(
                        0.05f, -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f,
                        -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f, -0.05f
                    )
                }

                activation = { tensor -> with(tensor) { relu() } }
            }

            // Output layer: 16 -> 1 neuron
            dense(1) {
                // Weights: 1x16 matrix - explicitly defined values
                weights {
                    from(
                        0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f, 0.2f, -0.4f,
                        0.1f, -0.5f, 0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f
                    )
                }

                // Bias: single value - explicitly defined
                bias {
                    from(0.0f)
                }

                // No activation for output layer (linear output)
            }
        }
    }

    override fun modelCard(): String {
        return model.describe(Shape(1), FP32::class)
    }
}