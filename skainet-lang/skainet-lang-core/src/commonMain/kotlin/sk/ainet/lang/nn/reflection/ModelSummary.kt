package sk.ainet.lang.nn.reflection

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType
import sk.ainet.lang.utils.table.table


public data class NodeSummary(val name: String, val input: Shape, val output: Shape, val params: Long)

public class Summary<T : DType, V>(private val ctx: ExecutionContext) {

    public val nodes: MutableList<NodeSummary> = mutableListOf<NodeSummary>()
    private val dataFactory = DenseTensorDataFactory()

    private fun countParameters(module: Module<T, V>): Long {
        var params = 0L

        if (module is ModuleParameters<*, *>) {
            module.params.forEach { param ->
                params += param.value.shape.volume
            }
        }

        return params
    }

    private fun nodeSummary(module: Module<T, V>, input: Shape, output: Shape): NodeSummary {
        val params = countParameters(module)

        return NodeSummary(
            module.name,
            input,
            output,
            params
        )
    }


    private fun traverseModules(
        module: Module<T, V>,
        currentInput: Shape,
        dtypeClass: kotlin.reflect.KClass<T>
    ): List<NodeSummary> {
        val result = mutableListOf<NodeSummary>()
        var currentShape = currentInput

        // For leaf modules (modules with parameters OR modules that transform shapes), create a summary
        if ((module is ModuleParameters<*, *> && module.params.isNotEmpty()) || module.modules.isEmpty()) {
            try {
                // Create a dummy input tensor using VoidOpsTensor for shape inference
                val inputTensorData = dataFactory.zeros<T, V>(currentShape, dtypeClass)
                val inputTensor: Tensor<T, V> = VoidOpsTensor(inputTensorData, dtypeClass)

                // Use actual forward pass to get proper output shape
                val outputTensor = module.forward(inputTensor, ctx)
                val outputShape = outputTensor.shape

                // Only add to summary if module has parameters (to avoid cluttering with activation/flatten layers)
                if (module is ModuleParameters<*, *> && module.params.isNotEmpty()) {
                    val summary = nodeSummary(module, currentShape, outputShape)
                    result.add(summary)
                }

                // Update current shape for next module in sequence (important for all modules)
                currentShape = outputShape
            } catch (e: Exception) {
                // Enhanced error reporting with layer details
                val errorMessage = buildString {
                    appendLine("Error in layer '${module.name}' during model summary generation:")
                    appendLine("  Layer type: ${module::class.simpleName}")
                    appendLine("  Input shape: $currentShape")
                    appendLine("  Expected tensor creation with dtype: ${dtypeClass.simpleName}")
                    if (module is ModuleParameters<*, *> && module.params.isNotEmpty()) {
                        appendLine("  Parameters:")
                        module.params.forEachIndexed { idx, param ->
                            appendLine("    ${idx + 1}. ${param.name}: ${param.value.shape}")
                        }
                    }
                    appendLine("  Original error: ${e::class.simpleName}: ${e.message}")
                }
                throw IllegalStateException(errorMessage, e)
            }
        }

        // Recursively traverse nested modules with proper shape propagation
        module.modules.forEach { subModule ->
            try {
                val subResults = traverseModules(subModule, currentShape, dtypeClass)
                result.addAll(subResults)

                // If this submodule produced results, update currentShape to the last output shape
                if (subResults.isNotEmpty()) {
                    currentShape = subResults.last().output
                }
            } catch (e: Exception) {
                // Re-throw with context about which submodule caused the issue
                if (e is IllegalStateException && e.message?.contains("Error in layer") == true) {
                    // Already has detailed error info, just re-throw
                    throw e
                } else {
                    // Add context about the parent module and current traversal state
                    val errorMessage = buildString {
                        appendLine("Error in submodule of '${module.name}' during model summary generation:")
                        appendLine("  Parent module: ${module.name} (${module::class.simpleName})")
                        appendLine("  Current shape before submodule: $currentShape")
                        appendLine("  Submodule being processed: ${subModule.name} (${subModule::class.simpleName})")
                        appendLine("  Original error: ${e::class.simpleName}: ${e.message}")
                    }
                    throw IllegalStateException(errorMessage, e)
                }
            }
        }

        return result
    }

    public fun summary(
        model: Module<T, V>,
        input: Shape,
        dtypeClass: kotlin.reflect.KClass<T>,
        batch_size: Int = -1
    ): List<NodeSummary> {
        nodes.clear()
        val summaries = traverseModules(model, input, dtypeClass)
        nodes.addAll(summaries)
        return summaries
    }

    public fun printSummary(nodes: List<NodeSummary>): String =
        table {
            cellStyle {
                border = true
            }
            header {
                row {
                    cell("Layer (type)")
                    cell("Output Shape")
                    cell("Param #")
                }
            }
            nodes.forEach { node ->
                row {
                    cell(node.name)
                    cell(node.output.toString())
                    cell(node.params)
                }
            }
        }.toString()
}

public fun <T : DType, V> Module<T, V>.describe(
    input: Shape,
    ctx: ExecutionContext,
    dtypeClass: kotlin.reflect.KClass<T>,
    batch_size: Int = -1
): String {
    val summary = Summary<T, V>(ctx)
    val nodes = summary.summary(this, input, dtypeClass, batch_size)
    return summary.printSummary(nodes)
}