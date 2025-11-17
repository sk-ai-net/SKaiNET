package sk.ainet.io

import junit.framework.TestCase.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSource
import kotlinx.io.buffered

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.io.mapper.NamesBasedValuesModelMapper
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.json.JsonParametersLoader
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32

class JsonParametersLoaderTest {

    private fun Module<FP32, Float>.infer(ctx: DirectCpuExecutionContext, x: Float): Float {
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(x))
        val out = this.forward(input, ctx)
        // If you have a proper accessor/iterator use that; otherwise adjust per your Tensor API
        return out.data[0] // Replace with the correct element access for your Tensor
    }

    @Test
    fun `test csv load with mapper by names`() = runBlocking {
        val model: Module<FP32, Float> = definition {
            network<FP32, Float> {
                input(1)
                dense(16) {
                    weights {
                        zeros()
                    }
                    bias {
                        zeros()
                    }
                }
                activation { tensor -> tensor.relu() }
                dense(16) {
                    weights {
                        zeros()
                    }
                    bias {
                        zeros()
                    }
                }
                activation { tensor -> tensor.relu() }
                dense(1)
            }
        }

        val isr = javaClass.getResourceAsStream("/sinus-approximator.json")
        checkNotNull(isr) { "sinus-approximator.json not found on classpath" }
        isr.use { inputStream ->
            val source = inputStream.asSource().buffered()
            val loader = JsonParametersLoader { source }
            val mapper = NamesBasedValuesModelMapper<FP32, Float>()
            val ctx = DirectCpuExecutionContext()

            loader.load<FP32, Float>(ctx) { name, tensor ->
                mapper.mapToModel(model, mapOf(name to tensor))
            }

            // Example: verify we indeed updated some parameters via ModuleParameters
            val paramCount = countParams(model)
            assertTrue(paramCount > 0, "Model should have parameters mapped")

            // Functional check (adjust tolerance and indexing as per your API)
            val approx = model.infer(ctx, (Math.PI / 2.0).toFloat())
            assertTrue(kotlin.math.abs(approx - 1.0f) < 0.05f)
        }
    }

    private fun countParams(model: Module<FP32, Float>): Int {
        var count = 0
        fun visit(m: Module<FP32, Float>) {
            val mp = (m as? sk.ainet.lang.nn.topology.ModuleParameters<FP32, Float>)
            if (mp != null) count += mp.params.size
            m.modules.forEach { visit(it as Module<FP32, Float>) }
        }
        visit(model)
        return count
    }
}
