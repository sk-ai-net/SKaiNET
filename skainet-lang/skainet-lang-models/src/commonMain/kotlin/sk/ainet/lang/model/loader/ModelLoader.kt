package sk.ainet.lang.model.loader

import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.FP32

/**
 * Utility functions for loading model weights and parameters.
 */

/**
 * Loads model weights into a neural network.
 * This is currently a no-op placeholder to keep the module compiling in environments
 * where GGUF loading is not available. It preserves API compatibility.
 *
 * @param model The neural network to apply the weights to.
 * @param modelParamsSource The source containing the model parameters (placeholder type Any).
 * @param modelType The type of model, which determines the matching strategy.
 */
public fun loadModelWeights(model: Module<FP32, Float>, modelParamsSource: Any, modelType: String = "") {
    // No-op placeholder. Implementations for GGUF loading live in skainet-io module.
}
