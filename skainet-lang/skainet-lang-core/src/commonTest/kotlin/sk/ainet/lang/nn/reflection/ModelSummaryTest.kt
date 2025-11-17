package sk.ainet.lang.nn.reflection

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelSummaryTest {
    
    private val dataFactory = DenseTensorDataFactory()
    private val ctx = DefaultNeuralNetworkExecutionContext()
    
    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }
    
    @Test
    fun testLinearLayerSummary() {
        // Create a simple linear layer: 10 inputs -> 5 outputs
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
        
        // Test the summary
        val summary = Summary<FP32, Float>(ctx)
        val nodes = summary.summary(linear, inputShape, FP32::class)
        
        assertEquals(1, nodes.size, "Should have exactly one node for Linear layer")
        
        val node = nodes[0]
        assertEquals("TestLinear", node.name)
        assertEquals(inputShape, node.input)
        assertEquals(Shape(intArrayOf(1, 5)), node.output) // Expected output: batch_size=1, features=5
        assertEquals(55L, node.params) // 5*10 weights + 5 bias = 55 parameters
    }
    
    @Test
    fun testMLPSummary() {
        // Create MLP: 10 -> 8 -> 4 -> 2
        val inputShape = Shape(intArrayOf(1, 10))
        
        // Create individual Linear layers
        val layer1 = Linear<FP32, Float>(
            inFeatures = 10,
            outFeatures = 8,
            name = "Layer1",
            initWeights = createTensor(Shape(intArrayOf(8, 10))),
            initBias = createTensor(Shape(intArrayOf(8)))
        )
        
        val layer2 = Linear<FP32, Float>(
            inFeatures = 8,
            outFeatures = 4,
            name = "Layer2", 
            initWeights = createTensor(Shape(intArrayOf(4, 8))),
            initBias = createTensor(Shape(intArrayOf(4)))
        )
        
        val layer3 = Linear<FP32, Float>(
            inFeatures = 4,
            outFeatures = 2,
            name = "Layer3",
            initWeights = createTensor(Shape(intArrayOf(2, 4))),
            initBias = createTensor(Shape(intArrayOf(2)))
        )
        
        val mlp = MLP<FP32, Float>(layer1, layer2, layer3, name = "TestMLP")
        
        // Test the summary
        val summary = Summary<FP32, Float>(ctx)
        val nodes = summary.summary(mlp, inputShape, FP32::class)
        
        // Should have 3 Linear layers (input->hidden1, hidden1->hidden2, hidden2->output)
        assertEquals(3, nodes.size, "MLP should have 3 linear layers")
        
        // Verify layer names and parameter counts
        val resultLayer1 = nodes[0] // 10 -> 8
        assertEquals("Layer1", resultLayer1.name)
        assertEquals(88L, resultLayer1.params) // 10*8 + 8 = 88
        
        val resultLayer2 = nodes[1] // 8 -> 4  
        assertEquals("Layer2", resultLayer2.name)
        assertEquals(36L, resultLayer2.params) // 8*4 + 4 = 36
        
        val resultLayer3 = nodes[2] // 4 -> 2
        assertEquals("Layer3", resultLayer3.name)
        assertEquals(10L, resultLayer3.params) // 4*2 + 2 = 10
    }
    
    @Test
    fun testDescribeExtensionFunction() {
        // Test the describe extension function
        val inputShape = Shape(intArrayOf(1, 5))
        val outputShape = Shape(intArrayOf(3, 5))
        val biasShape = Shape(intArrayOf(3))
        
        val weights = createTensor(outputShape)
        val bias = createTensor(biasShape)
        
        val linear = Linear<FP32, Float>(
            inFeatures = 5,
            outFeatures = 3,
            name = "TestLayer",
            initWeights = weights,
            initBias = bias
        )
        
        val description = linear.describe(inputShape, ctx,FP32::class, )
        
        // Verify the description contains expected elements
        assertTrue(description.contains("TestLayer"), "Description should contain layer name")
        assertTrue(description.contains("Output Shape"), "Description should contain output shape header")
        assertTrue(description.contains("Param #"), "Description should contain parameter count header")
        assertTrue(description.contains("18"), "Description should show parameter count: 5*3 + 3 = 18")
    }
    
    @Test
    fun testShapeInferenceCorrectness() {
        // Test that shape inference works correctly through VoidTensorOps
        val inputShape = Shape(intArrayOf(2, 4))  // batch_size=2, features=4
        val outputShape = Shape(intArrayOf(7, 4)) // weights: 7x4
        val biasShape = Shape(intArrayOf(7))      // bias: 7
        
        val weights = createTensor(outputShape)
        val bias = createTensor(biasShape)
        
        val linear = Linear<FP32, Float>(
            inFeatures = 4,
            outFeatures = 7,
            name = "ShapeTest",
            initWeights = weights,
            initBias = bias
        )
        
        val summary = Summary<FP32, Float>(ctx)
        val nodes = summary.summary(linear, inputShape, FP32::class)
        
        val node = nodes[0]
        assertEquals(inputShape, node.input)
        // Expected output: [batch_size=2, features=7] due to matmul([2,4], [4,7]) + bias[7] = [2,7]
        assertEquals(Shape(intArrayOf(2, 7)), node.output)
    }
}