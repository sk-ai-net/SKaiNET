package sk.ainet.lang.nn

import sk.ainet.lang.nn.reflection.Summary
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SimpleModelDescribeTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun testMnistCnnModelDescribeAndSummary() {
        val model = definition<FP32, Float> {
            network {
                // Stage: "conv1"
                stage("conv1") {
                    conv2d(outChannels = 16, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu1") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }

                /*
                // Stage: "conv2"
                stage("conv2") {
                    conv2d(outChannels = 32, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu2") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }

                // Stage: "flatten"
                stage("flatten") {
                    flatten()
                }

                // Stage: "dense"
                stage("dense") {
                    dense(outputDimension = 128)
                    activation(id = "relu3") { tensor -> tensor.relu() }
                }

                // Stage: "output"
                stage("output") {
                    dense(outputDimension = 10)
                    activation(id = "softmax") { tensor -> tensor.softmax(dim = 1) }
                }

                 */
            }
        }
        print(model.describe(Shape(1,1,28,28), FP32::class))
    }


    @Test
    fun testSimpleModelDescribeAndSummary() {
        // Create a simple linear layer with proper weights: 10 inputs -> 5 outputs
        val inputShape = Shape(1, 10)  // batch size 1, 10 features
        val outputShape = Shape(5, 10) // weights: 5x10
        val biasShape = Shape(5)       // bias: 5

        val weights = createTensor(outputShape)
        val bias = createTensor(biasShape)

        val linear = Linear<FP32, Float>(
            inFeatures = 10,
            outFeatures = 5,
            name = "TestLinear",
            initWeights = weights,
            initBias = bias
        )

        // Test describe method
        val description = linear.describe(inputShape, FP32::class)
        assertNotNull(description, "Description should not be null")
        assertTrue(description.isNotEmpty(), "Description should not be empty")
        assertTrue(description.contains("Output Shape"), "Description should contain output shape information")
        assertTrue(description.contains("Param #"), "Description should contain parameter count information")

        println("[DEBUG_LOG] Model Description:")
        println(description)

        // Test summary method
        val summary = Summary<FP32, Float>()
        val nodes = summary.summary(linear, inputShape, FP32::class)

        assertNotNull(nodes, "Summary nodes should not be null")
        assertTrue(nodes.isNotEmpty(), "Summary should contain at least one node")

        println("[DEBUG_LOG] Model Summary nodes count: ${nodes.size}")
        nodes.forEachIndexed { index, node ->
            println("[DEBUG_LOG] Node $index: ${node.name}, Input: ${node.input}, Output: ${node.output}, Params: ${node.params}")
        }

        // Verify that we have exactly one linear layer
        assertTrue(nodes.size == 1, "Should have exactly one linear layer")
        assertTrue(nodes[0].name == "TestLinear", "Layer name should be TestLinear")

        // Verify total parameter count: 5*10 weights + 5 bias = 55 parameters
        val totalParams = nodes.sumOf { it.params }
        assertTrue(totalParams == 55L, "Total parameter count should be 55 (5*10 + 5)")
        println("[DEBUG_LOG] Total parameters: $totalParams")
    }
}