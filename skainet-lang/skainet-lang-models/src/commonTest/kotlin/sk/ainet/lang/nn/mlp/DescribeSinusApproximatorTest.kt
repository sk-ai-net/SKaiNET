package sk.ainet.lang.nn.mlp

import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test

class DescribeSinusApproximatorTest {

    @Test
    fun testSinusApproximatorShapes() {
        println("[DEBUG_LOG] Starting SinusApproximator to ComputeGraph conversion test")

        // Create the SinusApproximator model
        val sinusModel = SinusApproximator()

        print(sinusModel.model<FP32, Float>().describe(Shape(16, 1), FP32::class))

    }
}