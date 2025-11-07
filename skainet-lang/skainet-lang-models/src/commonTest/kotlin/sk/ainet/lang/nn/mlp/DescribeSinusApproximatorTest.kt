package sk.ainet.lang.nn.mlp

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import sk.ainet.lang.nn.reflection.describe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test

class DescribeSinusApproximatorTest {

    @Test
    fun testSinusApproximatorShapes() {
        println("[DEBUG_LOG] Starting SinusApproximator to ComputeGraph conversion test")
        val ctx = DirectCpuExecutionContext()


        // Create the SinusApproximator model
        val sinusModel = SinusApproximator()

        print(sinusModel.model(ctx).describe(Shape(16, 1), FP32::class))

    }
}