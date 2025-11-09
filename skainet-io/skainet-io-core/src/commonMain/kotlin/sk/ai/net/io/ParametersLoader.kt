package sk.ai.net.io

import sk.ai.net.Tensor

interface ParametersLoader {
    suspend fun load(onTensorLoaded: (String, Tensor) -> Unit)
}