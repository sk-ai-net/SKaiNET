package sk.ainet.lang.kan.examples

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.kan.kanLayer
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.types.FP32

/**
 * A tiny, fixed KAN-based approximator for y = sin(x) on the interval [0, π/2].
 * It consists of a single KAN layer whose basis, mixer weights and bias were
 * exported from the accompanying PyTorch script in `piekan/train.py`, and are
 * embedded here as constants. At runtime, it builds a Module with those values
 * frozen, serving as a lightweight, dependency-free example of loading
 * pretrained KAN parameters in Kotlin.
 *
 * What it is:
 * - A minimal, ready-to-use example demonstrating how to instantiate a KAN layer
 *   with predetermined basis/weights/bias to approximate sin(x) on a narrow
 *   domain.
 * - Deterministic and non-trainable in this form; it does not modify parameters
 *   during execution.
 *
 * What it is not:
 * - Not a general-purpose sine model outside [0, π/2]; quality may degrade
 *   beyond the trained range.
 * - Not a full KAN stack or training pipeline; it omits optimizers, loss, and
 *   data handling, focusing solely on inference with fixed parameters.
 */
public object SineKanPretrained {

    public val basisValues: FloatArray = floatArrayOf(
        -0.61100733f,
        -0.54657453f,
        0.46215805f,
        0.43084604f,
        0.09301915f,
        -0.35296658f,
        0.47887635f,
        0.7482730f,
        -0.60586500f,
        0.63107467f,
        0.95884424f,
        0.82173777f,
        0.76084960f,
        -1.0838321f,
        -0.83811396f,
        1.101861f
    )
    public val mixingValues: FloatArray = floatArrayOf(
        0.65286535f,
        0.53845954f,
        -0.41289636f,
        -0.2080267f,
        0.08796077f,
        -0.2876495f,
        0.3955012f,
        0.36192146f,
        -0.56926537f,
        0.65100110f,
        0.4879738f,
        0.6272730f,
        0.72681934f,
        -0.5352414f,
        -0.7117360f,
        0.54629606f
    )
    public val biasValues: FloatArray = floatArrayOf(0.3989265f)

    /**
     * Build the pretrained KAN module with fixed weights/basis/bias.
     */
    public fun create(executionContext: ExecutionContext): Module<FP32, Float> = definition {
        sequential<FP32, Float> {
            input(1, "input")
            kanLayer(outputDim = 1, gridSize = 16, degree = 3, id = "sine-kan-pretrained") {
                baseActivation = { it } // identity; spline+mixer encode the fit
                weights { _ -> fromArray(mixingValues) }
                basis { _ -> fromArray(basisValues) }
                bias { _ -> fromArray(biasValues) }
            }
        }
    }

    /**
     * Produce a minimal Graphviz/DOT visualization of this tiny pretrained KAN.
     * This does not depend on any external Graphviz library — it just returns
     * a DOT string that you can render with the `dot` tool or any Graphviz viewer.
     *
     * The graph shows a single input feeding into the fixed KAN layer node,
     * and annotates that node with concise parameter summaries (counts).
     *
     * Example usage:
     * - Save to file: println(SineKanPretrained.toGraphvizDot()) > sine_kan.dot
     * - Render PNG:   dot -Tpng sine_kan.dot -o sine_kan.png
     */
    public fun toGraphvizDot(rankdir: String = "LR"): String {
        require(rankdir == "LR" || rankdir == "TB") { "rankdir must be either 'LR' or 'TB'" }

        val inputId = "input"
        val kanId = "sine_kan_pretrained"

        val basisCount = basisValues.size
        val weightCount = mixingValues.size
        val biasCount = biasValues.size

        val label = buildString {
            append("KAN Layer | id: sine-kan-pretrained\\n")
            append("gridSize: 16, degree: 3\\n")
            append("basis: ").append(basisCount)
            append(", weights: ").append(weightCount)
            append(", bias: ").append(biasCount)
        }

        return buildString {
            appendLine("digraph {")
            appendLine("    rankdir=$rankdir;")
            appendLine("    $inputId [label=\"input | x\", shape=record, style=filled, fillcolor=lightblue];")
            appendLine("    $kanId [label=\"$label\", shape=record];")
            appendLine("    $inputId -> $kanId;")
            appendLine("}")
        }
    }
}
