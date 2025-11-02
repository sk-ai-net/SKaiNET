package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertFailsWith

class DoubleTensorsTest {

    data class WeightAndBias<V>(val name: String, val weights: Tensor<FP32, V>, val bias: Tensor<FP32, V>? = null)

    @Test
    fun testTensorDSLSyntaxWithDoubles() {
        assertFailsWith<ClassCastException> {
            data {
                val layer1Weigths = doubleArrayOf(
                    -0.5437579154968262,
                    -0.40618014335632324,
                    -0.04907243698835373,
                    -0.2054896354675293,
                    0.7046658992767334,
                    0.12716591358184814,
                    -0.6933680772781372,
                    -0.6911409497261047,
                    0.7351927757263184,
                    0.8775343298912048,
                    0.03899011388421059,
                    -0.5247653126716614,
                    -0.907718300819397,
                    0.7729036808013916,
                    -0.584505021572113,
                    0.8824521899223328
                )

                val layer1Bias = doubleArrayOf(
                    -0.6030652523040771,
                    0.8545130491256714,
                    0.9554063677787781,
                    -0.014425158500671387,
                    0.4734141230583191,
                    -0.8752269744873047,
                    0.9116081595420837,
                    0.29334962368011475,
                    -0.03179183229804039,
                    -0.4028461277484894,
                    0.6525490880012512,
                    0.6051297783851624,
                    -0.40821588039398193,
                    -0.6744815111160278,
                    0.39602163434028625,
                    0.2196938693523407
                )

                val layer1WandB = WeightAndBias<Double>(
                    "layer1WandB", tensor<FP32, Double> {
                        shape(1, 16) {
                            init { indices ->
                                val col = indices[1]
                                layer1Weigths[col]
                            }
                        }
                    },
                    tensor<FP32, Double> {
                        shape(16) {
                            init { indices ->
                                val col = indices[0]
                                layer1Bias[col]
                            }
                        }
                    })
            }
        }
    }
}