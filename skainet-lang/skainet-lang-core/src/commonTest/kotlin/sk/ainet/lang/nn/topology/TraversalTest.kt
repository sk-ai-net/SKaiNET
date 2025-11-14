package sk.ainet.lang.nn.topology

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class TraversalTest {

    private val dataFactory = DenseTensorDataFactory()

    // Minimal ModuleNode impl for tests
    private data class FakeNode(
        override val id: String,
        override val name: String,
        override var path: String? = null,
        override val children: MutableList<ModuleNode> = mutableListOf(),
        override val params: List<ModuleParameter<*, *>> = emptyList()
    ) : ModuleNode

    private fun makeWeightParam(name: String): ModuleParameter<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(Shape(1), FP32::class)
        val tensor = VoidOpsTensor(data, FP32::class)
        return ModuleParameter.WeightParameter(name, tensor)
    }

    @Test
    fun walkDepthFirst_visitsAllNodes_inOrder() {
        // Build tree: root -> (a -> (a1, a2), b)
        val a1 = FakeNode(id = "a1", name = "A1")
        val a2 = FakeNode(id = "a2", name = "A2")
        val a = FakeNode(id = "a", name = "A", children = mutableListOf(a1, a2))
        val b = FakeNode(id = "b", name = "B")
        val root = FakeNode(id = "root", name = "Root", children = mutableListOf(a, b))

        val visited = mutableListOf<String>()
        root.walkDepthFirst { node -> visited += node.name }

        assertEquals(listOf("Root", "A", "A1", "A2", "B"), visited)
    }

    @Test
    fun collectParams_aggregatesFromSubtree() {
        // Create params at different levels
        val pRoot = listOf(makeWeightParam("root.weight"))
        val pA = listOf(makeWeightParam("a.w1"), makeWeightParam("a.w2"))
        val pA1 = listOf(makeWeightParam("a1.w"))
        val pB = emptyList<ModuleParameter<*, *>>()

        val a1 = FakeNode(id = "a1", name = "A1", params = pA1)
        val a2 = FakeNode(id = "a2", name = "A2")
        val a = FakeNode(id = "a", name = "A", children = mutableListOf(a1, a2), params = pA)
        val b = FakeNode(id = "b", name = "B", params = pB)
        val root = FakeNode(id = "root", name = "Root", children = mutableListOf(a, b), params = pRoot)

        val all = root.collectParams()
        val names = all.map { it.name }.toSet()

        assertEquals(4, all.size) // 1 + 2 + 1 + 0
        assertTrue("root.weight" in names)
        assertTrue("a.w1" in names)
        assertTrue("a.w2" in names)
        assertTrue("a1.w" in names)
    }

    @Test
    fun bindPaths_setsHierarchicalPaths_andFindWorks() {
        val a1 = FakeNode(id = "x-1", name = "leaf")
        val a2 = FakeNode(id = "x-2", name = "leaf2")
        val a = FakeNode(id = "x-a", name = "branch", children = mutableListOf(a1, a2))
        val root = FakeNode(id = "x-root", name = "root", children = mutableListOf(a))

        bindPaths(root, base = "model")

        // Verify paths
        assertEquals("model", root.path)
        // child path uses child name if available
        val branch = root.children.first()
        assertEquals("model/branch", branch.path)
        assertEquals("model/branch/leaf", branch.children[0].path)
        assertEquals("model/branch/leaf2", branch.children[1].path)

        // findById should locate by id
        assertNotNull(root.findById("x-2"))
        // findByPath should locate by bound path
        assertNotNull(root.findByPath("model/branch/leaf2"))
    }
}
