package sk.ainet.lang.model


public sealed class ModelExecutionResult<out V> {
    public data object Processing : ModelExecutionResult<Nothing>()
    public data class Progress(
        val current: Int,
        val total: Int,
        val message: String? = null
    ) : ModelExecutionResult<Nothing>()

    public data class Done<out V>(val result: V) : ModelExecutionResult<V>()
    public data class Error(
        val message: String,
        val cause: Throwable? = null
    ) : ModelExecutionResult<Nothing>()
}
