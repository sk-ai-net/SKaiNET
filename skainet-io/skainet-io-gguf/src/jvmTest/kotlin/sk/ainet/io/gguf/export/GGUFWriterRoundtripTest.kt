package sk.ainet.io.gguf.export

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.FP32
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGUFWriterRoundtripTest {

    private fun stubOp(name: String): Operation = object : Operation {
        override val name: String = name
        override val type: String = "stub"
        override val parameters: Map<String, Any> = emptyMap()
        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name)
    }

    @Test
    fun roundtrip_singleTensorAndGraphMetadata() {
        // Build a tiny graph: a -> b with one edge.
        val graph = DefaultComputeGraph()
        val a = graph.addNode(
            GraphNode(
                id = "n0",
                operation = stubOp("input"),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("t0", listOf(1, 2), "FP32"))
            )
        )
        val b = graph.addNode(
            GraphNode(
                id = "n1",
                operation = stubOp("relu"),
                inputs = listOf(TensorSpec("t0", listOf(1, 2), "FP32")),
                outputs = listOf(TensorSpec("t1", listOf(1, 2), "FP32"))
            )
        )
        graph.addEdge(
            GraphEdge(
                id = "e0",
                source = a,
                destination = b,
                sourceOutputIndex = 0,
                destinationInputIndex = 0,
                tensorSpec = a.outputs.first()
            )
        )

        // One FP32 weight tensor.
        val dataFactory = DenseTensorDataFactory()
        val weightData = dataFactory.fromFloatArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f), FP32)
        val weightTensor = VoidOpsTensor(weightData, FP32::class)
        val weights = mapOf("layer0.weight" to weightTensor)

        val request = exportGraphToGguf(graph, weights, label = "demo")
        val (report, bytes) = GGUFWriter.writeToByteArray(request)

        assertEquals(1, report.tensorCount)
        assertTrue(bytes.isNotEmpty())

        // Read back with GGUFReader to validate header and tensor directory.
        val reader = GGUFReader(bytes.inputStream().asSource().buffered(), loadTensorData = true)
        assertNotNull(reader.fields["GGUF.version"])
        val tensorCountField = reader.fields["skainet.tensor.count"]
        assertNotNull(tensorCountField)
        val countList = tensorCountField.parts[tensorCountField.data[0]] as List<*>
        val tensorCount = (countList.first() as Number).toInt()
        assertEquals(1, tensorCount)
        assertEquals(1, reader.tensors.size)
        val tensor = reader.tensors.first()
        assertEquals("layer0.weight", tensor.name)
        assertEquals(listOf(1u, 2u), tensor.shape)
    }
}
