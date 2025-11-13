package sk.ainet.lang.nn.hooks

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingParams
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Int32

class ForwardHooksTapeTest {

    @Test
    fun embedding_records_single_forward() {
        val tape = TapeRecorder()
        val ctx: ExecutionContext = DirectCpuExecutionContext(_hooks = tape)

        val params = EmbeddingParams(numEmbeddings = 10, embeddingDim = 4, paddingIdx = null)
        val emb = Embedding(ctx, FP32::class, params, name = "Emb")

        val indices = intArrayOf(1, 2, 3)
        val out = emb.forward(indices, ctx)
        assertEquals(Shape(3, 4), out.shape)

        val entries = tape.tape
        assertEquals(1, entries.size, "Exactly one module should be recorded")
        val e = entries[0]
        assertEquals("Emb", e.moduleName)
        assertNotNull(e.outputSpec)
        assertEquals(listOf(3, 4), e.outputSpec!!.shape)
    }
}
