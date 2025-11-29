package sk.ainet.lang.tensor.ops

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals

class UnsqueezeViewExtraTests {
    private val ctx = DefaultDataExecutionContext()

    @Test
    fun unsqueeze_view_preserves_and_reflects_mutations() {
        // Base tensor 1x2x2 with bytes [1,2,3,4]
        val baseShape = Shape(1, 2, 2)
        val bytes = byteArrayOf(1, 2, 3, 4)
        val t = ctx.fromByteArray<Int8, Byte>(baseShape, Int8::class, bytes)

        // Unsqueeze at dim=1 -> shape [1,1,2,2]
        val u = t.ops.unsqueeze(t, 1)
        assertEquals(Shape(1, 1, 2, 2), u.shape)

        // Read values through the view
        assertEquals(1.toByte(), u.data[0, 0, 0, 0])
        assertEquals(4.toByte(), u.data[0, 0, 1, 1])

        // Mutate through the unsqueezed view and verify it reflects in the base tensor
        u.data[0, 0, 1, 0] = 99
        // This should map to base index [0,1,0]
        assertEquals(99.toByte(), t.data[0, 1, 0])

        // Another mutation
        u.data[0, 0, 0, 1] = (-7).toByte()
        assertEquals((-7).toByte(), t.data[0, 0, 1])
    }

    @Test
    fun unsqueeze_negative_dim_normalization() {
        val shape = Shape(2, 3)
        val t = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, ByteArray(6) { it.toByte() })

        val uEnd = t.ops.unsqueeze(t, -1)
        assertEquals(Shape(2, 3, 1), uEnd.shape)

        val uBegin = t.ops.unsqueeze(t, 0)
        assertEquals(Shape(1, 2, 3), uBegin.shape)
    }
}
