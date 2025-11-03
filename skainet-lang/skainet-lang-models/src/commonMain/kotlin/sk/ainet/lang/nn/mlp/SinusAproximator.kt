package sk.ainet.lang.nn.mlp

import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.mlp.pretrained.SinusApproximatorWandB
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

public class SinusApproximator() : Model {

    private val sinusApproximatorWandB: SinusApproximatorWandB = SinusApproximatorWandB()

    public override fun <T : DType, V> model(): Module<FP32, Float> = model

    private val model = definition<FP32, Float>() {
        network {
            input(1, "input")  // Single input for x value

            // First hidden layer: 1 -> 16 neurons
            dense(16, "hidden-1") {
                // Weights: 16x1 matrix - explicitly defined values
                weights {
                    fromArray(
                        sinusApproximatorWandB.getLayer1WandB("").weights
                    )
                }
                // Bias: 16 values - explicitly defined
                bias {
                    fromArray(
                        sinusApproximatorWandB.getLayer1WandB("").bias
                    )
                }
                activation = { tensor -> with(tensor) { relu() } }
            }

            // Second hidden layer: 16 -> 16 neurons
            dense(16, "hidden-2") {
                // Weights: 16x16 matrix - explicitly defined values
                weights {
                    fromArray(
                        sinusApproximatorWandB.getLayer2WandB("").weights
                    )
                }
                // Bias: 16 values - explicitly defined
                bias {
                    fromArray(
                        sinusApproximatorWandB.getLayer2WandB("").bias
                    )
                }
                activation = { tensor -> with(tensor) { relu() } }
            }

            // Output layer: 16 -> 1 neuron
            dense(1, "output") {
                // Weights: 1x16 matrix - explicitly defined values
                weights {
                    fromArray(
                        sinusApproximatorWandB.getLayer3WandB("").weights
                    )
                }

                // Bias: single value - explicitly defined
                bias {
                    fromArray(
                        sinusApproximatorWandB.getLayer3WandB("").bias
                    )
                }

                // No activation for output layer (linear output)
            }
        }
    }

    override fun modelCard(): String {
        return model.describe(Shape(1), FP32::class)
    }
}