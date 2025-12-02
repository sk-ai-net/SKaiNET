package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.FP32

class DefaultCpuOpsUpsampleTest {
    private val ctx = DirectCpuExecutionContext()
    private val ops = ctx.ops

    @Test
    fun nearest_mode_repeats_values() {
        data(ctx) { _ ->
            val input = tensor<FP32, Float> {
                shape(1, 1, 2, 2) {
                    init { idx -> (1 + idx[2] * 2 + idx[3]).toFloat() }
                }
            }

            val upsampled = ops.upsample2d(
                input = input,
                scale = 2 to 2,
                mode = UpsampleMode.Nearest,
                alignCorners = false
            )

            assertEquals(Shape(1, 1, 4, 4), upsampled.shape)
            assertEquals(1f, upsampled.data[0, 0, 0, 0])
            assertEquals(2f, upsampled.data[0, 0, 0, 2])
            assertEquals(3f, upsampled.data[0, 0, 2, 0])
            assertEquals(4f, upsampled.data[0, 0, 3, 3])
        }
    }
}
