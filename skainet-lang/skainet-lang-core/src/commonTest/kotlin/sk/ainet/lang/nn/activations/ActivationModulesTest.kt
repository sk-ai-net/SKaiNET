package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActivationModulesTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun createTensor(shape: Shape) = VoidOpsTensor(
        dataFactory.zeros<FP32, Float>(shape, FP32::class),
        FP32::class
    )

    @Test
    fun testSiLU_Forward_PreservesShape() {
        val activation = SiLU<FP32, Float>()
        val input = createTensor(Shape(2, 3, 4))

        val output = activation.forward(input, ctx)

        assertEquals(input.shape, output.shape)
        assertEquals(input.dtype, output.dtype)
        assertEquals("SiLU", activation.name)
        assertTrue(activation.modules.isEmpty())
    }

    @Test
    fun testGELU_Forward_PreservesShape() {
        val activation = GELU<FP32, Float>()
        val input = createTensor(Shape(2, 3, 4))

        val output = activation.forward(input, ctx)

        assertEquals(input.shape, output.shape)
        assertEquals(input.dtype, output.dtype)
        assertEquals("GELU", activation.name)
        assertTrue(activation.modules.isEmpty())
    }

    @Test
    fun testSigmoid_Forward_PreservesShape() {
        val activation = Sigmoid<FP32, Float>()
        val input = createTensor(Shape(2, 3, 4))

        val output = activation.forward(input, ctx)

        assertEquals(input.shape, output.shape)
        assertEquals(input.dtype, output.dtype)
        assertEquals("Sigmoid", activation.name)
        assertTrue(activation.modules.isEmpty())
    }

    @Test
    fun testActivations_CustomName() {
        val customSiLU = SiLU<FP32, Float>("CustomSiLU")
        val customGELU = GELU<FP32, Float>("CustomGELU")
        val customSigmoid = Sigmoid<FP32, Float>("CustomSigmoid")

        assertEquals("CustomSiLU", customSiLU.name)
        assertEquals("CustomGELU", customGELU.name)
        assertEquals("CustomSigmoid", customSigmoid.name)
    }

    @Test
    fun testActivations_DifferentShapes() {
        val siluActivation = SiLU<FP32, Float>()
        val geluActivation = GELU<FP32, Float>()
        val sigmoidActivation = Sigmoid<FP32, Float>()

        val shapes = listOf(
            Shape(1),
            Shape(10),
            Shape(2, 5),
            Shape(3, 4, 5),
            Shape(2, 3, 4, 5)
        )

        for (shape in shapes) {
            val input = createTensor(shape)

            val siluOutput = siluActivation.forward(input, ctx)
            val geluOutput = geluActivation.forward(input, ctx)
            val sigmoidOutput = sigmoidActivation.forward(input, ctx)

            assertEquals(shape, siluOutput.shape, "SiLU failed for shape $shape")
            assertEquals(shape, geluOutput.shape, "GELU failed for shape $shape")
            assertEquals(shape, sigmoidOutput.shape, "Sigmoid failed for shape $shape")
        }
    }
}