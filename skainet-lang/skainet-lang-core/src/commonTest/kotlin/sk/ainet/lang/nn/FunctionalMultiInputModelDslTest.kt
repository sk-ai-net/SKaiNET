package sk.ainet.lang.nn

import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Verifies building the multi-input model using the Network DSL for conv/pool/flatten parts.
 * Pipeline: Conv2d(1->16, 3x3, s=1, p=0) -> MaxPool2d(2x2, s=2) -> Flatten -> Concat(aux vector) -> Dense(32) -> Dense(1)
 */
class FunctionalMultiInputModelDslTest {
    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun make(shape: Shape, fill: Float = 0f): Tensor<FP32, Float> =
        tensor(ctx, FP32::class) { tensor { shape(shape) { full(fill) } } }

    private fun buildMultiInputModelDsl(): Functional<FP32, Float> {
        // CNN branch built using DSL
        val cnnBranch = sequential<FP32, Float> {
            // Explicitly set inChannels because shape inference is not automatic yet
            conv2d(outChannels = 16, kernelSize = 3 to 3, stride = 1 to 1, padding = 0 to 0) {
                inChannels = 1
            }
            maxPool2d(kernelSize = 2 to 2, stride = 2 to 2, padding = 0 to 0)
            flatten { /* keep batch */ }
        }

        return Functional.of(
            inputs = listOf(
                FuncInput(name = "matrixInput", dimensions = intArrayOf(1, 28, 28)),
                FuncInput(name = "vectorInput", dimensions = intArrayOf(10))
            )
        ) { args, exec ->
            val matrix = args["matrixInput"]
            val vector = args["vectorInput"]

            val flat = cnnBranch.forward(matrix, exec)
            val merged = flat.ops.concat(listOf(flat, vector), dim = -1)

            // Dense 1: 2714 -> 32
            val inFeat1 = merged.shape.dimensions.last()
            val outFeat1 = 32
            val w1 = exec.zeros<FP32, Float>(Shape(outFeat1, inFeat1), FP32::class)
            val b1 = exec.zeros<FP32, Float>(Shape(outFeat1), FP32::class)
            val d1 = merged.matmul(w1.t()) + b1

            // Dense 2: 32 -> 1
            val outFeat2 = 1
            val w2 = exec.zeros<FP32, Float>(Shape(outFeat2, outFeat1), FP32::class)
            val b2 = exec.zeros<FP32, Float>(Shape(outFeat2), FP32::class)
            val d2 = d1.matmul(w2.t()) + b2

            d2
        }
    }

    @Test
    fun singleComputation_multiInput_forward_shape_with_dsl() {
        val model = buildMultiInputModelDsl()
        val batch = 2
        val matrixInput = make(Shape(batch, 1, 28, 28), fill = 1f)
        val vectorInput = make(Shape(batch, 10), fill = 0.5f)

        val out = model.forward(
            mapOf(
                "matrixInput" to matrixInput,
                "vectorInput" to vectorInput
            ),
            ctx
        )

        // Expected shape after pipeline ends with Dense(1)
        assertEquals(Shape(batch, 1), out.shape)
    }
}