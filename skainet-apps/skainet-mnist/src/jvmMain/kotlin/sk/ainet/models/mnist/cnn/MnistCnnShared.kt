package sk.ainet.models.mnist.cnn

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.dnn.cnn.MnistCnn
import sk.ainet.lang.model.loader.loadModelWeights
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

class MnistCnnShared(
    private val scope: CoroutineScope,
    private val execContext: ExecutionContext,
    private val resourcePath: String = "/models/mnist/mnist_cnn.gguf"
) {
    private val stateMutable = MutableStateFlow<MnistCnnSharedState>(MnistCnnSharedState.Idle)
    val state: StateFlow<MnistCnnSharedState> = stateMutable.asStateFlow()

    private val mutex = Mutex()
    private var module: Module<FP32, Float>? = null

    suspend fun load() = mutex.withLock {
        if (module != null && stateMutable.value is MnistCnnSharedState.Ready) return@withLock
        stateMutable.value = MnistCnnSharedState.Loading(0, 100, "initializing")

        val modelDef = MnistCnn()
        val built: Module<FP32, Float> = modelDef.create(execContext)

        // Load GGUF weights from resources
        val paramsBytes = readResource(resourcePath)
        loadModelWeights(
            model = built,
            modelParamsSource = paramsBytes,
            modelType = "mnist-cnn-gguf"
        )

        module = built
        stateMutable.value = MnistCnnSharedState.Ready(built)
    }

    suspend fun run(input: Tensor<FP32, Float>): Tensor<FP32, Float> = mutex.withLock {
        val m = module ?: error("Model not loaded")
        stateMutable.value = MnistCnnSharedState.Running(0, 1, "forward")
        val out = m.forward(input, execContext)
        stateMutable.value = MnistCnnSharedState.Ready(m)
        out
    }

    suspend fun unload() = mutex.withLock {
        module = null
        stateMutable.value = MnistCnnSharedState.Unloaded
    }

    private fun readResource(path: String): ByteArray {
        val stream = this::class.java.getResourceAsStream(path)
            ?: error("Resource not found: $path")
        return stream.use { it.readAllBytes() }
    }
}
