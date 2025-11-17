package sk.ainet.lang.model.dnn.mlp

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.model.dnn.mlp.pretrained.SinusApproximatorWandB
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.model.ModelIndexEntry
import sk.ainet.lang.model.ModelResult
import sk.ainet.lang.model.Task
import sk.ainet.lang.model.Dataset
import sk.ainet.lang.model.Metric
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32

public class SinusApproximator() : Model<FP32, Float, FloatArray, FloatArray> {

    private val sinusApproximatorWandB: SinusApproximatorWandB = SinusApproximatorWandB()

    public fun model(executionContext: ExecutionContext): Module<FP32, Float> = definition {
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

    override fun create(executionContext: ExecutionContext): Module<FP32, Float> = model(executionContext)

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: FloatArray,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): FloatArray {
        val inputTensor = data<FP32, Float>(executionContext) {
            tensor {
                // Using shape(1, 1) to represent a single scalar input in 2D form
                shape(1, 1) {
                    fromArray(inputValue)
                }
            }
        }
        val outputTensor = module.forward(inputTensor, executionContext)
        return listOf(outputTensor.data[0, 0]).toFloatArray()
    }

    override fun modelCard(): ModelCard {
        // Provide a fully-populated ModelCard with meaningful defaults for this simple numeric regression model
        return ModelCard(
            license = "MIT",
            libraryName = "skainet",
            pipelineTag = "regression",
            language = listOf("en"),
            modalities = listOf("numeric"),
            baseModel = "",
            contextLength = 1,
            datasets = listOf("synthetic-sine"),
            metrics = listOf("mse"),
            modelIndex = listOf(
                ModelIndexEntry(
                    name = "SinusApproximator",
                    results = listOf(
                        ModelResult(
                            task = Task(type = "regression"),
                            dataset = Dataset(name = "synthetic-sine", type = "synthetic"),
                            metrics = listOf(
                                Metric(name = "MSE", value = 0.0)
                            )
                        )
                    )
                )
            ),
            intendedUse = "Approximates y = sin(x) for scalar inputs in [-pi, pi]. Use for demos, tests, and educational purposes.",
            limitations = "Trained on synthetic data; accuracy outside training range may degrade. Not a general-purpose function approximator."
        )
    }
}