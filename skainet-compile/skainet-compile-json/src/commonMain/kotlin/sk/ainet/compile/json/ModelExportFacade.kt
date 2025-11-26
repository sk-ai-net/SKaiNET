package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph

/**
 * Facade to export a high-level model or a prebuilt ComputeGraph into the JSON export model.
 *
 * Capabilities:
 * - If [model] is already a ComputeGraph, delegates to [exportGraphToJson].
 * - Otherwise, prefer using the overload that accepts a [forwardPass] lambda to run a single
 *   forward execution under a recording tape; the tape is then converted to a ComputeGraph and exported.
 */
public fun <T : Any> exportModelToJson(model: T, label: String = model::class.simpleName ?: "model"): SkJsonExport {
    return when (model) {
        is ComputeGraph -> exportGraphToJson(model, label)
        else -> {
            error(
                buildString {
                    // Avoid KClass.qualifiedName which is not supported on Kotlin/JS
                    val typeName = model::class.simpleName ?: "unknown"
                    appendLine("exportModelToJson(model) does not have a direct adapter for type: $typeName.")
                    appendLine("If you have a ComputeGraph already, call exportGraphToJson(graph, label) instead.")
                    appendLine(
                        "Otherwise, call exportModelToJson(model, forwardPass = { /* run a single forward pass here */ }, label) " +
                            "so the facade can record execution to a tape, build a ComputeGraph, and export JSON."
                    )
                }
            )
        }
    }
}

/**
 * Fallback overload for models without a direct adapter.
 *
 * Provide a [forwardPass] that executes exactly one forward run of the model with example inputs.
 * The facade will:
 * 1) Start a recording tape
 * 2) Execute [forwardPass]
 * 3) Convert the tape to a [ComputeGraph]
 * 4) Delegate to [exportGraphToJson]
 */
public fun <T : Any> exportModelToJson(
    model: T,
    forwardPass: () -> Unit,
    label: String = model::class.simpleName ?: "model"
): SkJsonExport {
    // If caller passed a graph as model, keep the fast path
    if (model is ComputeGraph) return exportGraphToJson(model, label)

    // Record a single forward pass under the new graph/tape execution context
    val ctx = DefaultGraphExecutionContext.tape()
    val (tape, _) = ctx.record {
        forwardPass()
    }
    // Prefer DefaultExecutionTape.toComputeGraph() which builds a real graph from traces/ops.
    val graph = when (tape) {
        is DefaultExecutionTape -> tape.toComputeGraph()
        // Fallback to the generic extension (may be a stub in some builds)
        else -> tape?.toComputeGraph() ?: sk.ainet.lang.graph.DefaultComputeGraph()
    }
    return exportGraphToJson(graph, label)
}

// --- Adapter registry (optional, API-only hook) ---
