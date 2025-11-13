package sk.ainet.lang.nn.compute

import sk.ainet.context.data
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP16
import kotlin.test.Test
import kotlin.test.assertEquals

class Rgb2GrayScaleMultiplyTest {

    private fun makeVoidInput1x1(rgb: Triple<Float, Float, Float>): Tensor<FP16, Float> {
        val (r, g, b) = rgb
        return data<FP16, Float> {
            tensor<FP16, Float> {
                shape(1, 3, 1, 1) {
                    fromArray(floatArrayOf(r, g, b))
                }
            }
        }
    }

    private fun makeVoidInput1x2(
        r: Pair<Float, Float>,
        g: Pair<Float, Float>,
        b: Pair<Float, Float>
    ): Tensor<FP16, Float> {
        // NCHW layout with W=2: [R0, R1, G0, G1, B0, B1]
        val (r0, r1) = r
        val (g0, g1) = g
        val (b0, b1) = b
        return data<FP16, Float> {
            tensor<FP16, Float> {
                shape(1, 3, 1, 2) {
                    fromArray(
                        floatArrayOf(
                            r0, r1,
                            g0, g1,
                            b0, b1
                        )
                    )
                }
            }
        }
    }

    @Test
    fun testOutputShapeIsN1HW() {
        val ctx = DefaultNeuralNetworkExecutionContext()
        val model = Rgb2GrayScaleMatMul(DefaultNeuralNetworkExecutionContext())
        val module = model.model(ctx)

        val input = makeVoidInput1x2(
            r = 1.0f to 0.0f,
            g = 0.0f to 1.0f,
            b = 0.5f to 0.25f
        )

        val output = module.forward(input, ctx)
        assertEquals(Shape(1, 1, 1, 2), output.shape, "Output shape must be (N,1,H,W)")
        assertEquals(FP16::class, output.dtype)

    }

    @Test
    fun testZeroInputProducesZeroOutputShapeAndValues() {
        val ctx = DefaultNeuralNetworkExecutionContext()
        val model = Rgb2GrayScaleMatMul(DefaultNeuralNetworkExecutionContext())
        val module = model.model(ctx)

        val input = makeVoidInput1x1(Triple(0f, 0f, 0f))
        val output = module.forward(input, ctx)

        // Output shape should be (1,1,1,1) and value should be 0 with VoidOps
        assertEquals(Shape(1, 1, 1, 1), output.shape)
        val y = output.data[0, 0, 0, 0]
        assertEquals(0f, y)
    }
}
