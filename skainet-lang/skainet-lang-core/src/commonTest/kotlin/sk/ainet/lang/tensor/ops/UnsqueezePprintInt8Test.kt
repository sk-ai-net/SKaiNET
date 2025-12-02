package sk.ainet.lang.tensor.ops

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.pprint
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UnsqueezePprintInt8Test {
    @Test
    fun unsqueeze_keeps_data_and_pprint_not_zeros() {
        val ctx = DefaultDataExecutionContext()
        // Create 1x2x2 tensor with bytes: [1,2,3,4]
        val shape = Shape(1, 2, 2)
        val bytes = byteArrayOf(1, 2, 3, 4)
        val t = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes)

        // Unsqueeze at dim=1 -> shape [1,1,2,2]
        val u = t.ops.unsqueeze(t, 1)
        assertEquals(4, u.volume)
        assertEquals(4, t.volume)
        // Check values via data view mapping
        assertEquals(1.toByte(), (u.data[0, 0, 0, 0]))
        assertEquals(2.toByte(), (u.data[0, 0, 0, 1]))
        assertEquals(3.toByte(), (u.data[0, 0, 1, 0]))
        assertEquals(4.toByte(), (u.data[0, 0, 1, 1]))

        // pprint for rank>2 falls back to toString(); ensure not all zeros by spot checking contains "0"
        val printed = u.pprint()
        // At least some of our non-zero byte values should appear in the print string
        val has1or2 = printed.contains("1") || printed.contains("2") || printed.contains("3") || printed.contains("4")
        assertTrue(has1or2, "pprint output should reflect non-zero contents: $printed")
    }
}
