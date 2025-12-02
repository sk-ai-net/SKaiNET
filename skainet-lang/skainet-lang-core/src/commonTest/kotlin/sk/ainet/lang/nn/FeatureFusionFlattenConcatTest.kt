package sk.ainet.lang.nn

import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Example-oriented unit test that demonstrates how to fuse a 2D feature map (from a conv stack)
 * with an auxiliary 1D vector input by: Flatten(spatial) then Concat(features).
 *
 * Scenario:
 * - We start with a batched image-like tensor of shape (N, C, H, W).
 * - After some conv/pool layers, you typically still have (N, C', H', W').
 * - Flatten along spatial dimensions to get (N, C' * H' * W').
 * - Have an auxiliary vector V of shape (N, D).
 * - Concatenate along the last dimension to get (N, C' * H' * W' + D).
 *
 * This test uses VoidTensorOps which does "shape-only" ops, suitable for illustrating
 * how to wire operations together without requiring numeric kernels.
 */
class FeatureFusionFlattenConcatTest {

    private val ops = VoidTensorOps()
    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun makeTensor(shape: Shape, fill: Float): Tensor<FP32, Float> =
        tensor(ctx, FP32::class) {
            tensor {
                shape(shape) { full(fill) }
            }
        }

    @Test
    fun flattenConvOutputAndConcatWithVector_lastDim() {
        // Suppose a conv stack produced a feature map of shape (N=2, C=3, H=4, W=5)
        val batch = 2
        val convC = 3
        val convH = 4
        val convW = 5
        val convOut = makeTensor(Shape(batch, convC, convH, convW), fill = 1.0f)

        // Flatten keep batch via DSL network
        val flattenModel = sequential<FP32, Float> {
            flatten { /* defaults startDim=1, endDim=-1 */ }
        }
        val flat = flattenModel.forward(convOut, ctx)
        assertEquals(Shape(batch, convC * convH * convW), flat.shape)

        // Auxiliary vector V with D features per sample: shape (N, D)
        val d = 7
        val vector = makeTensor(Shape(batch, d), fill = 2.0f)

        // Concat along last dimension (feature axis): dim = -1
        val fused = ops.concat(listOf(flat, vector), dim = -1)

        // Expected output shape: (N, C*H*W + D)
        val expected = Shape(batch, convC * convH * convW + d)
        assertEquals(expected, fused.shape)

        // Sanity: dtype preserved and ordering is [flat | vector]
        assertTrue(fused.dtype == FP32::class)
    }

    @Test
    fun concatWithBroadcastableBatch_singleSample() {
        // Single-sample case also common: conv output (1, C, H, W) + vector (1, D)
        val conv = makeTensor(Shape(1, 8, 2, 2), fill = 0.5f) // (1, 8, 2, 2) -> flatten -> (1, 32)
        val flattenModel = sequential<FP32, Float> { flatten { } }
        val flat = flattenModel.forward(conv, ctx)
        assertEquals(Shape(1, 8 * 2 * 2), flat.shape)

        val vec = makeTensor(Shape(1, 10), fill = 3.14f)
        val fused = ops.concat(listOf(flat, vec), dim = -1)
        assertEquals(Shape(1, 32 + 10), fused.shape)
    }
}
