package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

class Conv2dVoidOpsShapeTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()

    private fun tensor(shape: Shape): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(dataFactory.zeros(shape, FP32::class), FP32::class)

    @Test
    fun conv2d_no_padding_stride1_kernel5_returns_expected_shape() {
        // Input: NCHW = (1, 3, 28, 28)
        val x = tensor(Shape(1, 3, 28, 28))
        // Weight: (out_channels=16, in_channels=3, kh=5, kw=5)
        val w = tensor(Shape(16, 3, 5, 5))

        val y = ops.conv2d(
            input = x,
            weight = w,
            bias = null,
            stride = 1 to 1,
            padding = 0 to 0,
            dilation = 1 to 1,
            groups = 1
        )

        // Expected: (1, 16, (28 - 5) + 1 = 24, 24)
        assertEquals(listOf(1, 16, 24, 24), y.shape.dimensions.toList())
    }

    @Test
    fun conv2d_same_padding_stride1_kernel5_preserves_spatial_dims() {
        val x = tensor(Shape(1, 3, 28, 28))
        val w = tensor(Shape(16, 3, 5, 5))

        val y = ops.conv2d(
            input = x,
            weight = w,
            bias = null,
            stride = 1 to 1,
            padding = 2 to 2, // SAME-like padding for 5x5
            dilation = 1 to 1,
            groups = 1
        )

        // Expected: (1, 16, 28, 28)
        assertEquals(listOf(1, 16, 28, 28), y.shape.dimensions.toList())
    }

    @Test
    fun conv2d_dilated_kernel_affects_effective_receptive_field() {
        val x = tensor(Shape(1, 3, 32, 32))
        // 3x3 kernel with dilation 2 → effective kernel = 1 + (3-1)*2 = 5
        val w = tensor(Shape(16, 3, 3, 3))

        val y = ops.conv2d(
            input = x,
            weight = w,
            bias = null,
            stride = 1 to 1,
            padding = 0 to 0,
            dilation = 2 to 2,
            groups = 1
        )

        // Expected H_out = ((32 - 1*2*(3-1) - 1) + 1) = 28; W_out = 28
        assertEquals(listOf(1, 16, 28, 28), y.shape.dimensions.toList())
    }
}
