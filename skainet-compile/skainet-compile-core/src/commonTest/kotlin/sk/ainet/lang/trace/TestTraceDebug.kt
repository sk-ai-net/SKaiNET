package sk.ainet.lang.trace

/**
 * Test-only trace debug utilities. Lightweight and multiplatform-friendly.
 *
 * Usage in tests:
 *   TraceDebug.enable()
 *   val sink = wrapWithDebug(originalSink)
 *   // run code that emits traces; traces will be printed while enabled
 */
object TraceDebug {
    // Not using @Volatile to keep common test compatibility; simple flag is enough for tests
    private var enabled: Boolean = false

    fun enable() { enabled = true }
    fun disable() { enabled = false }
    fun isEnabled(): Boolean = enabled
}

/**
 * Wrap an existing sink to conditionally print traces when TraceDebug is enabled.
 * Safe for hot paths in tests: minimal overhead when disabled.
 */
fun wrapWithDebug(delegate: OpSink): OpSink = object : OpSink {
    override fun onOpExecuted(trace: OpTrace) {
        if (TraceDebug.isEnabled()) {
            // Keep a simple, deterministic print format
            println("[TRACE_DEBUG] opType=${trace.opType} inputs=${trace.inputs} outputs=${trace.outputs} attrs=${trace.attributes}")
        }
        delegate.onOpExecuted(trace)
    }
}
