package sk.ainet.lang.nn.normalization

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class BatchNormalizationJvmTest {

    @Test
    fun broadcast_stats_over_spatial_dims() {
        val exec = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val input = exec.fromFloatArray<FP32, Float>(
            Shape(1, 2, 2, 2),
            FP32::class,
            floatArrayOf(
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
            )
        )
        val bn = BatchNormalization<FP32, Float>(
            numFeatures = 2,
            eps = 1e-5,
            momentum = 1.0,
            affine = false,
            name = "bn_broadcast_jvm"
        )
        bn.train()
        val out = bn.forward(input, exec)
        assertEquals(input.shape, out.shape)
    }
}
