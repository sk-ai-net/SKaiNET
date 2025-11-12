package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.MaxPool2d
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class MaxPool2dDslTest {

    private fun flattenModules(module: sk.ainet.lang.nn.Module<FP32, Float>): List<sk.ainet.lang.nn.Module<FP32, Float>> {
        val result = mutableListOf<sk.ainet.lang.nn.Module<FP32, Float>>()
        fun walk(m: sk.ainet.lang.nn.Module<FP32, Float>) {
            result += m
            m.modules.forEach { walk(it) }
        }
        walk(module)
        return result
    }

    @Test
    fun testMaxPool2dBodyDslBuildsModuleWithConfiguredProperties() {
        val model = definition<FP32, Float> {
            network {
                // Use body-style DSL for MaxPool2d
                maxPool2d("pool1") {
                    kernelSize(3)
                    stride(2)
                    padding(1)
                }
            }
        }

        assertNotNull(model)

        // Collect all nested modules and find MaxPool2d
        val allModules = flattenModules(model)
        val pool = allModules.firstOrNull { it is MaxPool2d<*, *> } as? MaxPool2d<FP32, Float>
        assertNotNull(pool, "Expected to find a MaxPool2d module in the built model")

        // Verify configured properties propagated from the DSL body
        assertEquals(3 to 3, pool.kernelSize, "kernelSize should be set via helper setter")
        assertEquals(2 to 2, pool.stride, "stride should be set via helper setter")
        assertEquals(1 to 1, pool.padding, "padding should be set via helper setter")

        // Verify name comes from the provided id
        assertEquals("pool1", pool.name)
    }
}
