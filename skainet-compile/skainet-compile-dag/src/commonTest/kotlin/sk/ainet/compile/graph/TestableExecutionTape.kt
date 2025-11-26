package sk.ainet.sk.ainet.compile.graph

import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.tape.RecordedOperation

// Use a subclass to add operations directly by specs (no real tensors required)
class TestableDefaultExecutionTape : DefaultExecutionTape() {
    fun addBySpec(op: Operation, inputs: List<TensorSpec>, outputs: List<TensorSpec>) {
        _operations.add(
            RecordedOperation(
                operation = op,
                inputs = inputs,
                outputs = outputs,
                timestamp = _operationCounter++
            )
        )
    }
}