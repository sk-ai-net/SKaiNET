package sk.ainet.lang.model

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType

public interface Model<T : DType, V, I, O> {

    public val coroutineContext: CoroutineDispatcher
        get() = Dispatchers.Default

    // output
    public fun create(executionContext: ExecutionContext): Module<T, V>

    /**
     * Implement this in your models.
     * Call [reportProgress] whenever you want to emit a progress update.
     */
    public suspend fun calculate(
        module: Module<T, V>,
        inputValue: I,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): O

    /**
     * Default implementation that wraps [calculate] into a Flow and emits
     * Processing / Progress / Done / Error.
     */
    public fun fit(
        module: Module<T, V>,
        inputValue: I,
        executionContext: ExecutionContext
    ): Flow<ModelExecutionResult<O>> = flow {
        emit(ModelExecutionResult.Processing)

        try {
            val result: O = withContext(coroutineContext) {
                // heavy work off the main thread
                calculate(
                    module = module,
                    inputValue = inputValue,
                    executionContext = executionContext,
                    reportProgress = { current, total, message ->
                        // This lambda is suspended, so we can emit from here
                        emit(
                            ModelExecutionResult.Progress(
                                current = current,
                                total = total,
                                message = message
                            )
                        )
                    }
                )
            }

            emit(ModelExecutionResult.Done(result))
        } catch (t: Throwable) {
            emit(
                ModelExecutionResult.Error(
                    message = t.message ?: "Unknown error",
                    cause = t
                )
            )
        }
    }

    public fun modelCard(): ModelCard
}