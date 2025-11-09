package sk.ai.net.io.csv

import kotlinx.serialization.Serializable

@Serializable
data class Tensor(
    val shape: List<Int>,
    val values: List<Double>
)

@Serializable
data class Parameter(
    val unique_parameter_name: String,
    val tensor: Tensor
)