package sk.ainet.lang.model.dnn.mlp

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.dnn.mlp.pretrained.SinusApproximatorWandB
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

public class SinusApproximator() : Model<FP32, Float> {

    private val sinusApproximatorWandB: SinusApproximatorWandB = SinusApproximatorWandB()

    override fun model(executionContext: ExecutionContext): Module<FP32, Float> = definition<FP32, Float> {
        network(executionContext) {
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

    override fun modelCard(): ModelCard {
        TODO("Not yet implemented")
    }

    /*
    override fun modelCard(): String {
        // Build with a default context to generate a model card/description
        val defaultModel = buildModel(DefaultNeuralNetworkExecutionContext())
        return defaultModel.describe(Shape(1), FP32::class)
    }

     */
}