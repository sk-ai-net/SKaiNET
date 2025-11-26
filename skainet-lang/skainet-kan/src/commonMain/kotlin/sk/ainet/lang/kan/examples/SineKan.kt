package sk.ainet.lang.kan.examples

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.kan.kanLayer
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.types.FP32

/**
 * Minimal KAN-based approximator for y = sin(x) on [0, pi/2].
 *
 * Uses a single KAN layer with a small grid and identity activation; basis/mixing
 * tensors are learnable via the usual parameter updates.
 */
public fun sineKan(executionContext: ExecutionContext): Module<FP32, Float> = definition {
    sequential<FP32, Float>(executionContext) {
        input(1, "input")
        kanLayer(outputDim = 1, gridSize = 16, degree = 3, id = "sine-kan") {
            baseActivation = { it } // identity to let spline/mixing do the work
        }
    }
}
