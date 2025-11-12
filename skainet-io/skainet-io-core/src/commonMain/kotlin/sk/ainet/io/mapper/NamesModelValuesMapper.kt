package sk.ainet.io.mapper

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


internal fun defaultNamesMatcher(moduleParamFullName: String, wandbKey: String): Boolean {
    val a = moduleParamFullName
    val b = wandbKey
    if (a == b) return true
    // Case-insensitive compare as a relaxed path
    if (a.equals(b, ignoreCase = true)) return true
    // Fallback: allow matching by suffix in case loaders prepend scopes (e.g., prefixes)
    if (b.endsWith(".$a")) return true
    if (b.endsWith(a)) return true
    return false
}

class NamesBasedValuesModelMapper<T : DType, V>(
    private val matcher: (moduleParamFullName: String, wandbKey: String) -> Boolean = ::defaultNamesMatcher
) : ModelValuesMapper<T, V> {

    override fun mapToModel(model: Module<T, V>, wAndB: Map<String, Tensor<T, V>>) {
        // Track already-used keys to avoid assigning the same tensor to multiple params
        val used = mutableSetOf<String>()
        traverseAndMap(model, wAndB, used)
    }

    // Recursively traverse the module tree.
    private fun traverseAndMap(module: Module<T, V>, wandb: Map<String, Tensor<T, V>>, used: MutableSet<String>) {
        if (module is ModuleParameters<*, *>) {
            @Suppress("UNCHECKED_CAST")
            mapModuleParameters(module as ModuleParameters<T, V>, wandb, used)
        }
        module.modules.forEach { child ->
            @Suppress("UNCHECKED_CAST")
            traverseAndMap(child as Module<T, V>, wandb, used)
        }
    }

    // For a module implementing ModuleParameters, match and update its parameters.
    private fun mapModuleParameters(
        moduleParameters: ModuleParameters<T, V>,
        wandb: Map<String, Tensor<T, V>>,
        used: MutableSet<String>
    ) {
        moduleParameters.params.forEach { param ->
            // Param names are already fully-qualified (e.g., "linear-1.weight").
            val fullName = param.name
            // Use the injected matcher function to find a matching wandb key.
            val matchingEntry = wandb.entries.firstOrNull { (key, _) ->
                key !in used && matcher(fullName, key)
            }
            if (matchingEntry != null) {
                used += matchingEntry.key
                param.value = matchingEntry.value
            }
        }
    }
}
