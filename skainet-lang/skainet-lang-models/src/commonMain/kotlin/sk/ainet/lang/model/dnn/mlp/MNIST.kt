package sk.ainet.lang.model.dnn.mlp

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.FP32

/**
 * Simple MLP for MNIST as a models-as-code example.
 */
public class MnistMpl : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    private fun buildModel(executionContext: ExecutionContext): Module<FP32, Float> = definition {
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

    // Backward-compatible helper for old call sites
    public fun model(executionContext: ExecutionContext): Module<FP32, Float> = create(executionContext)

    override fun create(executionContext: ExecutionContext): Module<FP32, Float> = buildModel(executionContext)

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): Tensor<FP32, Float> {
        reportProgress(0, 1, "starting mnist-mlp forward")
        val out = module.forward(inputValue, executionContext)
        reportProgress(1, 1, "done")
        return out
    }

    override fun modelCard(): ModelCard {
        return ModelCard(
            license = "apache-2.0",
            libraryName = "skainet",
            pipelineTag = "image-classification",
            language = listOf("en"),
            modalities = listOf("vision"),
            baseModel = "",
            contextLength = 0,
            datasets = listOf("mnist"),
            metrics = emptyList(),
            modelIndex = emptyList(),
            intendedUse = "MNIST MLP: Flatten(28x28)-> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)",
            limitations = ""
        )
    }
}

