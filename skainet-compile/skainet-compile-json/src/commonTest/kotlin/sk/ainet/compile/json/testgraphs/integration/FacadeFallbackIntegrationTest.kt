package sk.ainet.compile.json.testgraphs.integration

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.compile.json.exportModelToJson
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution

class FacadeFallbackIntegrationTest {

    private fun tensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = DenseTensorDataFactory().zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    private fun recOps(base: TensorOps = VoidTensorOps()): TensorOps = Execution.recordingOps(base)

    @Test
    fun export_facade_fallback_records_and_exports_json() {
        val ops = recOps()
        val a = tensor(Shape(1, 4))

        val export = exportModelToJson(
            model = Any(), // no direct adapter; will use fallback path
            forwardPass = {
                val r = ops.relu(a)
                @Suppress("UNUSED_VARIABLE")
                val s = ops.sigmoid(r)
            },
            label = "facade_fallback_chain"
        )

        assertEquals("facade_fallback_chain", export.label)
        assertEquals(1, export.graphs.size)
        val g = export.graphs.first()
        // Expect at least input + relu + sigmoid
        assertTrue(g.nodes.size >= 3)
        val labels = g.nodes.map { it.label }
        assertTrue(labels.contains("relu"))
        assertTrue(labels.contains("sigmoid"))

        // Determine deterministic structure: collect incoming edges count for sigmoid node
        val sigmoid = g.nodes.first { it.label == "sigmoid" }
        // incomingEdges shouldn't be null; assert at least one
        assertNotNull(sigmoid.incomingEdges)
        assertTrue(sigmoid.incomingEdges!!.size >= 1)
    }

    @Test
    fun conv2d_kernel_shape_inferred_from_weight_input() {
        val ops = recOps()
        val x = tensor(Shape(1, 3, 28, 28))
        val w = tensor(Shape(16, 3, 5, 5))

        val export = exportModelToJson(
            model = Any(),
            forwardPass = {
                @Suppress("UNUSED_VARIABLE")
                val y = ops.conv2d(
                    input = x,
                    weight = w,
                    bias = null,
                    stride = 1 to 1,
                    padding = 0 to 0,
                    dilation = 1 to 1,
                    groups = 1
                )
            },
            label = "facade_fallback_conv2d"
        )

        val g = export.graphs.first()
        val convNode = g.nodes.first { it.label == "conv2d" }
        // Find kernel_shape attribute; expected "(5, 5)"
        val kernelAttr = convNode.attrs.firstOrNull { it.key == "kernel_shape" }
        assertNotNull(kernelAttr, "Expected kernel_shape attribute mapped from weight input")
        assertEquals("(5, 5)", kernelAttr!!.value)
    }
}
