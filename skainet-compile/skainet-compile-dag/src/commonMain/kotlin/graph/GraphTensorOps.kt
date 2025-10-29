package sk.ainet.lang.tensor.ops

import sk.ainet.context.GraphExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.graph.*

/**
 * Graph-aware implementation of TensorOps that records operations as graph nodes.
 * This implementation extends the base TensorOps interface with graph node creation methods.
 */
public class GraphTensorOps<V>(
    private val baseOps: TensorOps<V>,
    private val graph: ComputeGraph,
    private val executionContext: GraphExecutionContext<V>
) : TensorOps<V> {

    private var nodeCounter = 0L
    
    private fun generateNodeId(operationName: String): String {
        return "${operationName}_${++nodeCounter}"
    }
    
    private fun createTensorSpec(tensor: Tensor<*, V>, name: String): TensorSpec {
        return TensorSpec(
            name = name,
            shape = tensor.shape.dimensions.toList(),
            dtype = tensor.dtype.simpleName ?: "Unknown",
            requiresGrad = false, // TODO: Add gradient support
            metadata = emptyMap()
        )
    }

    private fun <T : DType> ensureInputNode(tensor: Tensor<T, V>, inputName: String): GraphNode {
        // Check if a node for this tensor already exists
        val existingNode = graph.nodes.find { 
            it.operation.type == "input" && it.outputs.any { spec -> spec.name == inputName }
        }
        
        if (existingNode != null) {
            return existingNode
        }
        
        // Create new input node
        val inputOperation = InputOperation<T, V>()
        val inputNodeId = generateNodeId("input")
        val tensorSpec = createTensorSpec(tensor, inputName)
        
        val inputNode = GraphNode(
            id = inputNodeId,
            operation = inputOperation,
            inputs = emptyList(),
            outputs = listOf(tensorSpec)
        )
        
        graph.addNode(inputNode)
        return inputNode
    }

    // Basic mathematical operations
    override fun <T : DType> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.add(a, b)
        
        if (executionContext.isRecording) {
            // Ensure input nodes exist for both tensors
            val inputNodeA = ensureInputNode(a, "tensor_a")
            val inputNodeB = ensureInputNode(b, "tensor_b")
            
            // Create the addition operation node
            val operation = AddOperation<T, V>()
            val nodeId = generateNodeId("add")
            val inputs = listOf(
                createTensorSpec(a, "input_0"),
                createTensorSpec(b, "input_1")
            )
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val addNode = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(addNode)
            
            // Add edges from input nodes to the addition node
            graph.addEdge(GraphEdge("edge_a_to_add", inputNodeA, addNode, 0, 0, inputNodeA.outputs.first()))
            graph.addEdge(GraphEdge("edge_b_to_add", inputNodeB, addNode, 0, 1, inputNodeB.outputs.first()))
        }
        
        return result
    }

    override fun <T : DType> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.subtract(a, b)
        
        if (executionContext.isRecording) {
            val operation = SubtractOperation<T, V>()
            val nodeId = generateNodeId("subtract")
            val inputs = listOf(
                createTensorSpec(a, "input_0"),
                createTensorSpec(b, "input_1")
            )
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.multiply(a, b)
        
        if (executionContext.isRecording) {
            val operation = MultiplyOperation<T, V>()
            val nodeId = generateNodeId("multiply")
            val inputs = listOf(
                createTensorSpec(a, "input_0"),
                createTensorSpec(b, "input_1")
            )
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.divide(a, b)
        
        if (executionContext.isRecording) {
            val operation = DivideOperation<T, V>()
            val nodeId = generateNodeId("divide")
            val inputs = listOf(
                createTensorSpec(a, "input_0"),
                createTensorSpec(b, "input_1")
            )
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Linear algebra operations
    override fun <T : DType> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.matmul(a, b)
        
        if (executionContext.isRecording) {
            val operation = MatmulOperation<T, V>()
            val nodeId = generateNodeId("matmul")
            val inputs = listOf(
                createTensorSpec(a, "input_0"),
                createTensorSpec(b, "input_1")
            )
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.transpose(tensor)
        
        if (executionContext.isRecording) {
            val operation = TransposeOperation<T, V>()
            val nodeId = generateNodeId("transpose")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Convolutional operations
    override fun <T : DType> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        val result = baseOps.conv2d(input, weight, bias, stride, padding, dilation, groups)
        
        if (executionContext.isRecording) {
            val parameters = mapOf(
                "stride" to stride,
                "padding" to padding,
                "dilation" to dilation,
                "groups" to groups
            )
            val operation = Conv2dOperation<T, V>(parameters)
            val nodeId = generateNodeId("conv2d")
            
            val inputs = mutableListOf(
                createTensorSpec(input, "input"),
                createTensorSpec(weight, "weight")
            )
            if (bias != null) {
                inputs.add(createTensorSpec(bias, "bias"))
            }
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Pooling operations
    override fun <T : DType> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        val result = baseOps.maxPool2d(input, kernelSize, stride, padding)
        
        if (executionContext.isRecording) {
            val parameters = mapOf(
                "kernelSize" to kernelSize,
                "stride" to stride,
                "padding" to padding
            )
            val operation = MaxPool2dOperation<T, V>(parameters)
            val nodeId = generateNodeId("maxPool2d")
            val inputs = listOf(createTensorSpec(input, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Shape operations
    override fun <T : DType> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        val result = baseOps.reshape(tensor, newShape)
        
        if (executionContext.isRecording) {
            val parameters = mapOf("newShape" to newShape.dimensions)
            val operation = ReshapeOperation<T, V>(parameters)
            val nodeId = generateNodeId("reshape")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        val result = baseOps.flatten(tensor, startDim, endDim)
        
        if (executionContext.isRecording) {
            val parameters = mapOf(
                "startDim" to startDim,
                "endDim" to endDim
            )
            val operation = FlattenOperation<T, V>(parameters)
            val nodeId = generateNodeId("flatten")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Activation functions
    override fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.relu(tensor)
        
        if (executionContext.isRecording) {
            val operation = ReluOperation<T, V>()
            val nodeId = generateNodeId("relu")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val result = baseOps.softmax(tensor, dim)
        
        if (executionContext.isRecording) {
            val parameters = mapOf("dim" to dim)
            val operation = SoftmaxOperation<T, V>(parameters)
            val nodeId = generateNodeId("softmax")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.sigmoid(tensor)
        
        if (executionContext.isRecording) {
            val operation = SigmoidOperation<T, V>()
            val nodeId = generateNodeId("sigmoid")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Enhanced shape operations with graph recording
    override fun <T : DType> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val result = baseOps.squeeze(tensor, dim)
        
        if (executionContext.isRecording) {
            val parameters = mapOf<String, Any>("dim" to (dim ?: -1))
            val operation = SqueezeOperation<T, V>(parameters)
            val nodeId = generateNodeId("squeeze")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    override fun <T : DType> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val result = baseOps.unsqueeze(tensor, dim)
        
        if (executionContext.isRecording) {
            val parameters = mapOf("dim" to dim)
            val operation = UnsqueezeOperation<T, V>(parameters)
            val nodeId = generateNodeId("unsqueeze")
            val inputs = listOf(createTensorSpec(tensor, "input_0"))
            val outputs = listOf(createTensorSpec(result, "output_0"))
            
            val node = GraphNode(nodeId, operation, inputs, outputs)
            graph.addNode(node)
        }
        
        return result
    }

    // Delegate remaining operations to base implementation
    override fun <T : DType> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> = baseOps.concat(tensors, dim)
    override fun <T : DType> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> = baseOps.split(tensor, splitSize, dim)
    override fun <T : DType> silu(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.silu(tensor)
    override fun <T : DType> gelu(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.gelu(tensor)
    override fun <T : DType> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.sum(tensor, dim)
    override fun <T : DType> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.mean(tensor, dim)
    override fun <T : DType> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.variance(tensor, dim)
    override fun <T : DType> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.sqrt(tensor)
    override fun <TFrom : DType, TTo : DType> convert(tensor: Tensor<TFrom, V>, targetType: TTo): Tensor<TTo, V> = baseOps.convert(tensor, targetType)
}