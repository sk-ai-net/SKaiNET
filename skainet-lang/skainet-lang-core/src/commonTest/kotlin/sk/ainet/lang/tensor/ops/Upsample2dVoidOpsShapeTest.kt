package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

class Upsample2dVoidOpsShapeTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()

    private fun tensor(shape: Shape): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(dataFactory.zeros(shape, FP32::class), FP32::class)

    @Test
    fun upsample2d_scales_spatial_dims() {
        val x = tensor(Shape(1, 3, 10, 20))
        val y = ops.upsample2d(x, scale = 2 to 3, mode = UpsampleMode.Nearest, alignCorners = false)
        assertEquals(listOf(1, 3, 20, 60), y.shape.dimensions.toList())
    }

    @Test
    fun upsample2d_invalid_rank_throws() {
        val x = tensor(Shape(1, 3, 10))
        assertFailsWith<IllegalArgumentException> {
            ops.upsample2d(x, scale = 2 to 2, mode = UpsampleMode.Nearest, alignCorners = false)
        }
    }
}
