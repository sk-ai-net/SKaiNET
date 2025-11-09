package sk.ainet.lang.model.dnn.mlp

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Simple MLP for MNIST as a models-as-code example.
 */
public class MnistMpl : Model<FP32, Float> {

    override fun model(executionContext: ExecutionContext): Module<FP32, Float> = definition {
        network<FP32, Float>(executionContext) {
            sequential {
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

    /*
    fun mdodelCard(): String {
        val defaultCtx = DefaultNeuralNetworkExecutionContext()
        //val net = model<FP32, Float>(defaultCtx)
        return net.describe(Shape(28, 28), FP32::class)
    }

     */


    override fun modelCard(): ModelCard {
        TODO("Not yet implemented")
    }
}

