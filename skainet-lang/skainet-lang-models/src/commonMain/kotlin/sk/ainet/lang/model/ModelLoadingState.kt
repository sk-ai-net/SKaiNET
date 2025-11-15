package sk.ainet.lang.model

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import sk.ainet.context.ExecutionContext


public sealed interface ModelLoadPhase {
    public data object Idle : ModelLoadPhase
    public data class Downloading(val bytesReceived: Long?, val totalBytes: Long?) : ModelLoadPhase
    public data class Unpacking(val progress: Float?) : ModelLoadPhase // 0..1
    public data class Initializing(val step: String?, val progress: Float?) : ModelLoadPhase
    public data object Ready : ModelLoadPhase
}

// Convenience enum-like state with percentage when applicable
public sealed interface ModelLoadState {
    public val phase: ModelLoadPhase
    public val message: String?
    public val progress: Float? // 0..1 when known

    public data class InProgress(
        override val phase: ModelLoadPhase,
        override val message: String? = null,
        override val progress: Float? = null
    ) : ModelLoadState

    public data class Succeeded(
        override val message: String? = null
    ) : ModelLoadState {
        override val phase: ModelLoadPhase = ModelLoadPhase.Ready
        override val progress: Float? = 1f
    }

    public data class Failed(
        val error: Throwable,
        override val message: String? = error.message
    ) : ModelLoadState {
        override val phase: ModelLoadPhase = ModelLoadPhase.Idle
        override val progress: Float? = null
    }
}

public interface ModelLoader {
    /** Emits granular states from start to fully ready (or error). */
    public fun load(executionContext: ExecutionContext, inBackground: Boolean = true): Flow<ModelLoadState>
}

public interface ModelHandle {
    public val status: StateFlow<ModelLoadState>
    public val executionContext: ExecutionContext
    public val scope: kotlinx.coroutines.CoroutineScope

    /** Returns true when weights and runtime are fully initialized and executable. */
    public val isReady: Boolean
}