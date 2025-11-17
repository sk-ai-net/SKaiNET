package sk.ainet.io.json

import kotlinx.serialization.Serializable

@Serializable
internal data class Tensor(
    val shape: List<Int>,
    val values: List<Double>
)

@Serializable
internal data class Parameter(
    val unique_parameter_name: String,
    val tensor: Tensor
)