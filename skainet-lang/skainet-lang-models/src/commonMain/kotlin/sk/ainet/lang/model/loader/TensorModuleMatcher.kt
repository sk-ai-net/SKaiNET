package sk.ainet.lang.model.loader

import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.FP32

public interface TensorModuleMatcher {
    public fun applyTensorsToModel(reader: Any, model: Module<FP32, Float>)
}

public object TensorModuleMatcherFactory {
    public fun createMatcher(modelType: String): TensorModuleMatcher = object : TensorModuleMatcher {
        override fun applyTensorsToModel(reader: Any, model: Module<FP32, Float>) {
            // no-op placeholder
        }
    }
}
