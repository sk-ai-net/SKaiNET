package sk.ainet.lang.nn.topology

import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.activations.ReLU
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.*

class MLPTest {

    @Test
    fun testMLPInference() {

        // Layer 1: 1 input -> 16 outputs
        data {
            val weights1 = tensor<FP32, Float> {
                shape(16, 1) {  ones() }
            }

            val bias1 = tensor<FP32, Float> {
                shape(1, 16) {  zeros() }
            }

            // Layer 2: 16 inputs -> 16 outputs
            val weights2 = tensor<FP32, Float> {
                shape(16, 16) {  ones() }
            }
            val bias2 = tensor<FP32, Float> {
                shape(1, 16) {  zeros() }
            }

            // Layer 3: 16 inputs -> 1 output with default dense data factory
            val weights3 = tensor<FP32, Float> {
                shape(1, 16) {  ones() }
            }
            val bias3 = tensor<FP32, Float> {
                shape(1, 1) {  zeros() }
            }

            // Create the MLP with 2 hidden layers of 16 neurons each and 1 output
            val mlp = MLP(
                Linear(1, 16, "layer1", weights1, bias1),
                ReLU("relu1"),
                Linear(16, 16, "layer2", weights2, bias2),
                ReLU("relu2"),
                Linear(16, 1, "layer3", weights3, bias3)
            )

            // Create test input
            val input = tensor<FP32, Float> {
                shape(1, 1) {  full(2.0f) }
            }

            // Perform inference
            val output = mlp.forward(input)

            // Basic assertions to verify the test works
            assertNotNull(output)
            assertEquals(Shape(1, 1), output.shape)
        }
    }
}