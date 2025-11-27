package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport
import sk.ainet.compile.json.testgraphs.TinyGraphs
import kotlin.test.Test
import kotlin.test.assertEquals

class GoldenJsonTest {

    private fun resourceText(path: String): String {
        val url = this::class.java.classLoader.getResource(path)
            ?: error("Resource not found: $path")
        return url.openStream().bufferedReader().use { it.readText() }.normalizeEol()
    }

    private fun String.normalizeEol(): String = this.replace("\r\n", "\n").replace('\r', '\n')

    @Test
    fun tiny_add_relu_matches_golden_and_round_trips() {
        // Arrange: build tiny test graph and export
        val graph = TinyGraphs.tinyAddReluGraph()
        val export = exportGraphToJson(graph, label = "tiny_add_relu")

        // Act: serialize with pretty printing for stable comparison
        val json = export.toJsonString(pretty = true).normalizeEol()

        // Load golden
        val golden = resourceText("golden/tiny_add_relu.json")

        // Assert: compare exact JSON text
        assertEquals(golden, json, "Exported JSON must match the golden fixture")

        // Round-trip: parse back and compare objects
        val parsed: SkJsonExport = exportJson(pretty = true).decodeFromString(json)
        assertEquals(export, parsed, "Parsed JSON should equal the original export object")
    }
}