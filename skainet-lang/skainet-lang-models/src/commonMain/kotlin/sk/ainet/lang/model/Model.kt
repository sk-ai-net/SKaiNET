package sk.ainet.lang.model

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType

public interface Model<T : DType, V> {
    // output
    public fun model(executionContext: ExecutionContext): Module<T, V>
    public fun modelCard(): ModelCard
}