package sk.ainet.lang.kan

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class AknModuleIntegrationTest {

    private val ctx = DefaultNeuralNetworkExecutionContext()

    @Test
    fun `akn module builds and runs without DSL`() {
        val inputDim = 4
        val outputDim = 3
        val grid = 2

        val akn = createAkn<FP32, Float>(
            executionContext = ctx,
            dtype = FP32::class,
            inputDim = inputDim,
            outputDim = outputDim,
            gridSize = grid,
            name = "akn-test",
            weightsInit = { ones() },
            basisInit = { ones() },
            biasInit = { zeros() }
        )

        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(inputDim), FP32::class,
            floatArrayOf(1f, 2f, 3f, 4f)
        )

        val output = akn.forward(input, ctx)
        assertEquals(Shape(outputDim), output.shape, "AKN output should match requested outputDim")
    }
}
