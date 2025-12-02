package sk.ainet.lang.nn

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.plus
import kotlin.test.Test
import kotlin.test.assertEquals

class FunctionalMultiInputModelTest {
    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun make(shape: Shape, fill: Float = 0f): Tensor<FP32, Float> =
        tensor(ctx, FP32::class) { tensor { shape(shape) { full(fill) } } }

    private fun buildMultiInputModel(): Functional<FP32, Float> {
        // Mirror the user-proposed single computation: conv -> pool -> flatten -> concat -> dense -> dense
        return Functional.of(
            inputs = listOf(
                FuncInput(name = "matrixInput", dimensions = intArrayOf(1, 28, 28)),
                FuncInput(name = "vectorInput", dimensions = intArrayOf(10))
            )
        ) { args, exec ->
            val matrix = args["matrixInput"] // expected shape: (N, C=1, H=28, W=28)
            val vector = args["vectorInput"] // expected shape: (N, 10)

            // Conv2D params
            val inC = 1
            val outC = 16
            val kH = 3
            val kW = 3
            val stride = 1 to 1
            val padding = 0 to 0
            // Create shape-only weights/bias
            val w1: Tensor<FP32, Float> = exec.zeros(Shape(outC, inC, kH, kW), FP32::class)
            val b1: Tensor<FP32, Float> = exec.zeros<FP32, Float>(Shape(outC), FP32::class)

            // Conv2D + ReLU (relu is no-op for shape)
            val convOut = matrix.ops.conv2d(
                input = matrix,
                weight = w1,
                bias = b1,
                stride = stride,
                padding = padding,
                dilation = 1 to 1,
                groups = 1
            )

            // MaxPool 2x2
            val pooled = convOut.ops.maxPool2d(convOut, kernelSize = 2 to 2, stride = 2 to 2, padding = 0 to 0)

            // Flatten keep batch
            val flat = pooled.ops.flatten(pooled, startDim = 1, endDim = -1)

            // Concat flattened conv features with the vector along last dim
            val merged = flat.ops.concat(listOf(flat, vector), dim = -1)

            // Dense 1: out=32
            val inFeat1 = merged.shape.dimensions.last()
            val outFeat1 = 32
            val wDense1: Tensor<FP32, Float> = exec.zeros<FP32, Float>(Shape(outFeat1, inFeat1), FP32::class)
            val bDense1: Tensor<FP32, Float> = exec.zeros<FP32, Float>(Shape(outFeat1), FP32::class)
            val mm1 = merged.matmul(wDense1.t())
            val dense1 = mm1 + bDense1

            // Dense 2: out=1
            val inFeat2 = outFeat1
            val outFeat2 = 1
            val wDense2: Tensor<FP32, Float> = exec.zeros<FP32, Float>(Shape(outFeat2, inFeat2), FP32::class)
            val bDense2: Tensor<FP32, Float> = exec.zeros<FP32, Float>(Shape(outFeat2), FP32::class)
            val mm2 = dense1.matmul(wDense2.t())
            val output = mm2 + bDense2

            output
        }
    }

    @Test
    fun singleComputation_multiInput_forward_shape() {
        val model = buildMultiInputModel()
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

        // Compute expected spatial after conv(3x3, stride=1, no pad): (28-3+1)=26; after 2x2 pool/2 -> 13
        // Channels = 16; flattened = 16*13*13 = 2704; merged with vector 10 -> 2714; Dense1 -> 32; Dense2 -> 1
        assertEquals(Shape(batch, 1), out.shape)
    }
}
