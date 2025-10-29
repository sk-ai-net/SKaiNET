package sk.ainet.sk.ainet.exec.tensor.models

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.execute.context.computation
import sk.ainet.execute.context.dsl.tensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32
import kotlin.test.Test

class SinusApproximatorTest {

    fun createModel(computation: ExecutionContext): Module<FP32, Float> {
        return definition() {
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
    }

    @Test
    fun testSinusApproximator() {
        val ctx = DirectCpuExecutionContext()
        computation<Float>(ctx) { computation ->
            // Create a simple input tensor compatible with the model's expected input size (1)
            val inputTensor = tensor<FP32, Float> {
                // Using shape(1, 1) to represent a single scalar input in 2D form
                shape(1, 1) { init { 0f } }
            }
            val b = inputTensor.t()
            val model = createModel(computation)
            model(inputTensor)
        }
    }
}