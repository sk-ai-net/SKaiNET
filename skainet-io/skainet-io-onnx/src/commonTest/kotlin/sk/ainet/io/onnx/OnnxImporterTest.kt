package sk.ainet.io.onnx

import onnx.AttributeProto
import onnx.GraphProto
import onnx.ModelProto
import onnx.NodeProto
import onnx.TensorProto
import onnx.TensorShapeProto
import onnx.TypeProto
import onnx.ValueInfoProto
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

class OnnxImporterTest {

    @Test
    fun `imports conv graph with initializer and input`() {
        val inputInfo = valueInfo(
            name = "input",
            shape = tensorShape(1, 3, 224, 224)
        )
        val outputInfo = valueInfo(
            name = "output",
            shape = tensorShape(1, 64, 224, 224)
        )
        val weightTensor = TensorProto(
            name = "weight",
            dataType = TensorProto.DataType.FLOAT.value,
            dims = listOf(64L, 3L, 3L, 3L)
        )
        val convNode = NodeProto(
            input = listOf("input", "weight"),
            output = listOf("output"),
            name = "conv0",
            opType = "Conv",
            attribute = listOf(
                AttributeProto(name = "strides", ints = listOf(1L, 1L)),
                AttributeProto(name = "pads", ints = listOf(1L, 1L, 1L, 1L))
            )
        )
        val graph = GraphProto(
            node = listOf(convNode),
            initializer = listOf(weightTensor),
            input = listOf(inputInfo),
            output = listOf(outputInfo)
        )
        val model = ModelProto(graph = graph)

        val computeGraph = OnnxToComputeGraphImporter().import(model.toGraphView())

        assertEquals(3, computeGraph.nodes.size, "expected input, initializer, and conv nodes")
        assertEquals(2, computeGraph.edges.size, "input and weight should feed conv")

        val convGraphNode = computeGraph.nodes.first { it.id == "conv0" }
        assertIs<sk.ainet.lang.tensor.ops.Conv2dOperation<*, *>>(convGraphNode.operation)
        assertEquals(listOf("output"), convGraphNode.outputs.map { it.name })
        assertEquals(listOf("input", "weight"), convGraphNode.inputs.map { it.name })

        val initNode = computeGraph.nodes.first { it.id == "init_weight" }
        assertEquals("weight", initNode.outputs.single().name)

        val convIncoming = computeGraph.edges.filter { it.destination.id == "conv0" }
        val sourceIds = convIncoming.map { it.source.id }.toSet()
        assertEquals(setOf("input", "init_weight"), sourceIds)
    }

    private fun tensorShape(vararg dims: Long): TensorShapeProto = TensorShapeProto(
        dim = dims.map { TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.DimValue(it)) }
    )

    private fun valueInfo(name: String, shape: TensorShapeProto): ValueInfoProto = ValueInfoProto(
        name = name,
        type = TypeProto(
            value = TypeProto.Value.TensorType(
                TypeProto.Tensor(
                    elemType = TensorProto.DataType.FLOAT.value,
                    shape = shape
                )
            )
        )
    )
}
