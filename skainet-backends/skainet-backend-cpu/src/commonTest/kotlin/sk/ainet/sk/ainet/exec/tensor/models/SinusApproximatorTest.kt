package sk.ainet.sk.ainet.exec.tensor.models

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.execute.context.computation
import sk.ainet.lang.nn.definition
import sk.ainet.lang.model.dnn.mlp.pretrained.SinusApproximatorWandB
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.pprint
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

private val sinusApproximatorWandB: SinusApproximatorWandB = SinusApproximatorWandB()


fun createModel(context: ExecutionContext) = definition<FP32, Float> {
    network(context) {
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
        activation("relu-1") { tensor -> with(tensor) { relu() } }

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
        activation("relu-2") { tensor -> with(tensor) { relu() } }

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


class SinusApproximatorTest {


    @Test
    fun testSinusApproximator() {
        val ctx = DirectCpuExecutionContext()
        val result = computation<Float>(ctx) { _ ->
            // Create a simple input tensor compatible with the model's expected input size (1)
            val inputTensor = data<FP32, Float>(ctx) {
                tensor<FP32, Float> {
                    // Using shape(1, 1) to represent a single scalar input in 2D form
                    shape(2, 1) {
                        fromArray(
                            floatArrayOf(0f, (PI / 2.0f).toFloat())
                        )
                    }
                }
            }
            val model = createModel(ctx)
            val result = model(inputTensor)
            //print(result.pprint())
            result.data[0,0]
        }
    }


    class SineNN(private val ctx: ExecutionContext) {
        val model_ = createModel(ctx)
        fun calcSine(angle: Float): Float {
            return computation(ctx) { computation ->
                // Create a simple input tensor compatible with the model's expected input size (1)
                val inputTensor = data<FP32, Float>(ctx) {
                    tensor<FP32, Float>() {
                        // Using shape(1, 1) to represent a single scalar input in 2D form
                        shape(2, 1) {
                            fromArray(
                                floatArrayOf(0f, (PI / 2.0f).toFloat())
                            )
                        }
                    }
                }

                val result = model_(inputTensor)
                print(result.pprint())
                result.data[0,0]
            }
        }
    }

    @Test
    fun testSinusApproximatorWithWeights() {
        val ctx = DirectCpuExecutionContext()
        val model = createModel(ctx)

        val rng = Random(42)
        val samples = 10
        val maxError = 0.1f

        repeat(samples) {
            val x: Float = (rng.nextFloat() * (PI.toFloat() / 2f))

            val predicted = computation(ctx) { _ ->
                val inputTensor = data<FP32, Float>(ctx) {
                    tensor<FP32, Float> {
                        // batch=1, features=1
                        shape(1, 1) {
                            fromArray(floatArrayOf(x))
                        }
                    }
                }
                val out = model(inputTensor)
                out.data[0, 0]
            }

            val expected = sin(x)
            val diff = abs(predicted - expected)
            assertTrue(
                diff <= maxError,
                message = "sin approximation error too high for x=$x: predicted=$predicted expected=$expected diff=$diff (max=$maxError)"
            )
        }
    }
}