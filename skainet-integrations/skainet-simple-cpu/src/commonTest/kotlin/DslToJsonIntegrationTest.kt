import sk.ainet.compile.json.exportGraphToJson
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration smoke tests to ensure we can:
 * 1) Instantiate a DSL model (from skainet-lang-models) with a real ExecutionContext
 * 2) Build/obtain a ComputeGraph (here we use a tiny synthetic graph as a stand-in for the compile step)
 * 3) Export the graph to the JSON model successfully
 */
class DslToJsonIntegrationTest {

    @Test
    fun dsl_model_instantiation_and_export_tiny_graph_succeeds() {
        // 1) Instantiate a DSL model using a real execution context (ensures DSL is wired and usable)
        val ctx = DirectCpuExecutionContext()
        val model = SinusApproximator().model(ctx)
        // Just verify we got a module object; we don't need to execute it here
        assertTrue(model != null, "Expected a non-null DSL model instance")

        // 2) Build a tiny synthetic compute graph (acting as compiled output)
        val graph = TinyGraphs.tinyAddReluGraph()

        // 3) Export to JSON and validate basic structure
        val export = exportGraphToJson(graph, label = "dsl_integration_add_relu")
        assertEquals("dsl_integration_add_relu", export.label)
        assertEquals(1, export.graphs.size)
        val g = export.graphs.first()
        assertEquals("main_graph", g.id)

        // Expect 4 nodes: input, bias, add, relu
        assertEquals(4, g.nodes.size)

        val labels = g.nodes.map { it.label }.toSet()
        // Operation.name in TensorOperations is lower-case (e.g., "input", "add", "relu")
        assertTrue(labels.contains("input"))
        assertTrue(labels.contains("add"))
        assertTrue(labels.contains("relu"))

        // Basic port metadata checks for presence of tensor_tag and tensor_shape
        val anyPorts = g.nodes.flatMap { it.outputsMetadata + it.inputsMetadata }
        assertTrue(anyPorts.isNotEmpty(), "Expected some port metadata present")
        val anyAttrs = anyPorts.flatMap { it.attrs }
        val keys = anyAttrs.map { it.key }.toSet()
        assertTrue(keys.contains("__tensor_tag"), "Port metadata should contain __tensor_tag")
        assertTrue(keys.contains("tensor_shape"), "Port metadata should contain tensor_shape")
    }
}