package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Explicit unit tests for checking shape operations functionality.
 * This test class focuses specifically on validating tensor shape transformations
 * and ensuring proper shape calculations for all shape-related operations.
 */
class ShapeOperationsTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()

    // Helper function to create test tensors
    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    // ============== RESHAPE OPERATION TESTS ==============

    @Test
    fun testReshape_BasicShapeTransformation() {
        val tensor = createTensor(Shape(2, 3, 4))  // Volume = 24

        val result = ops.reshape(tensor, Shape(6, 4))  // Same volume

        assertEquals(Shape(6, 4), result.shape)
        assertEquals(24, result.volume)  // Volume preserved
    }

    @Test
    fun testReshape_ToSingleDimension() {
        val tensor = createTensor(Shape(2, 3, 4))

        val result = ops.reshape(tensor, Shape(24))  // Flatten to 1D

        assertEquals(Shape(24), result.shape)
        assertEquals(1, result.rank)
    }

    @Test
    fun testReshape_ToHigherDimensions() {
        val tensor = createTensor(Shape(24))

        val result = ops.reshape(tensor, Shape(2, 3, 2, 2))  // 1D to 4D

        assertEquals(Shape(2, 3, 2, 2), result.shape)
        assertEquals(4, result.rank)
    }

    @Test
    fun testReshape_VolumeMismatch_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))  // Volume = 24

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.reshape(tensor, Shape(5, 6))  // Volume = 30 (mismatch)
        }
        assertTrue(exception.message?.contains("volume mismatch") == true)
    }

    // ============== FLATTEN OPERATION TESTS ==============

    @Test
    fun testFlatten_DefaultParameters() {
        val tensor = createTensor(Shape(2, 3, 4, 5))

        val result = ops.flatten(tensor)  // Default: startDim=0, endDim=-1

        assertEquals(Shape(120), result.shape)  // All dimensions flattened
        assertEquals(1, result.rank)
    }

    @Test
    fun testFlatten_PartialDimensions() {
        val tensor = createTensor(Shape(2, 3, 4, 5))

        val result = ops.flatten(tensor, startDim = 1, endDim = 2)  // Flatten dims 1 and 2

        assertEquals(Shape(2, 12, 5), result.shape)  // 3*4=12
        assertEquals(3, result.rank)
    }

    @Test
    fun testFlatten_SingleDimension() {
        val tensor = createTensor(Shape(2, 3, 4))

        val result = ops.flatten(tensor, startDim = 1, endDim = 1)  // Just dim 1

        assertEquals(Shape(2, 3, 4), result.shape)  // No change, single dim
    }

    @Test
    fun testFlatten_NegativeIndices() {
        val tensor = createTensor(Shape(2, 3, 4, 5))

        val result = ops.flatten(tensor, startDim = -2, endDim = -1)  // Last two dims

        assertEquals(Shape(2, 3, 20), result.shape)  // 4*5=20
    }

    @Test
    fun testFlatten_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, startDim = 5, endDim = 6)
        }
        assertTrue(exception.message?.contains("out of bounds") == true)
    }

    // ============== CONCAT OPERATION TESTS ==============

    @Test
    fun testConcat_MultipleTensors() {
        val tensor1 = createTensor(Shape(2, 3))
        val tensor2 = createTensor(Shape(2, 3))
        val tensor3 = createTensor(Shape(2, 3))
        val tensors = listOf(tensor1, tensor2, tensor3)

        val result = ops.concat(tensors, dim = 0)

        assertEquals(Shape(6, 3), result.shape)  // 2+2+2=6 in dim 0
    }

    @Test
    fun testConcat_DifferentSizesInConcatDimension() {
        val tensor1 = createTensor(Shape(2, 5))
        val tensor2 = createTensor(Shape(2, 3))
        val tensor3 = createTensor(Shape(2, 7))
        val tensors = listOf(tensor1, tensor2, tensor3)

        val result = ops.concat(tensors, dim = 1)

        assertEquals(Shape(2, 15), result.shape)  // 5+3+7=15 in dim 1
    }

    @Test
    fun testConcat_HighDimensionalTensors() {
        val tensor1 = createTensor(Shape(1, 2, 3, 4))
        val tensor2 = createTensor(Shape(1, 2, 3, 4))
        val tensors = listOf(tensor1, tensor2)

        val result = ops.concat(tensors, dim = 2)

        assertEquals(Shape(1, 2, 6, 4), result.shape)  // 3+3=6 in dim 2
    }

    @Test
    fun testConcat_ShapeMismatch_ThrowsException() {
        val tensor1 = createTensor(Shape(2, 3, 4))
        val tensor2 = createTensor(Shape(2, 5, 4))  // Different in dim 1
        val tensors = listOf(tensor1, tensor2)

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.concat(tensors, dim = 0)  // Concat on dim 0, but dim 1 differs
        }
        assertTrue(exception.message?.contains("same shape except in the concatenation dimension") == true)
    }

    // ============== SPLIT OPERATION TESTS ==============

    @Test
    fun testSplit_EqualChunks() {
        val tensor = createTensor(Shape(6, 4))

        val result = ops.split(tensor, splitSize = 2, dim = 0)

        assertEquals(3, result.size)  // 6/2 = 3 chunks
        result.forEach { chunk ->
            assertEquals(Shape(2, 4), chunk.shape)
        }
    }

    @Test
    fun testSplit_HighDimensionalTensor() {
        val tensor = createTensor(Shape(2, 3, 12, 5))

        val result = ops.split(tensor, splitSize = 3, dim = 2)

        assertEquals(4, result.size)  // 12/3 = 4 chunks
        result.forEach { chunk ->
            assertEquals(Shape(2, 3, 3, 5), chunk.shape)
        }
    }

    @Test
    fun testSplit_SingleChunk() {
        val tensor = createTensor(Shape(4, 6))

        val result = ops.split(tensor, splitSize = 6, dim = 1)

        assertEquals(1, result.size)  // 6/6 = 1 chunk
        assertEquals(Shape(4, 6), result[0].shape)
    }

    @Test
    fun testSplit_NonDivisibleSize_ThrowsException() {
        val tensor = createTensor(Shape(2, 7))

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.split(tensor, splitSize = 3, dim = 1)  // 7 not divisible by 3
        }
        assertTrue(exception.message?.contains("not divisible") == true)
    }

    // ============== SQUEEZE OPERATION TESTS ==============

    @Test
    fun testSqueeze_RemoveAllSingletonDimensions() {
        val tensor = createTensor(Shape(1, 4, 1, 3, 1))

        val result = ops.squeeze(tensor, dim = null)

        assertEquals(Shape(4, 3), result.shape)  // Only non-1 dimensions remain
    }

    @Test
    fun testSqueeze_RemoveSpecificDimension() {
        val tensor = createTensor(Shape(2, 1, 3, 4))

        val result = ops.squeeze(tensor, dim = 1)

        assertEquals(Shape(2, 3, 4), result.shape)  // Dimension 1 removed
    }

    @Test
    fun testSqueeze_AllSingletonDimensions() {
        val tensor = createTensor(Shape(1, 1, 1))

        val result = ops.squeeze(tensor, dim = null)

        assertEquals(Shape(1), result.shape)  // Becomes scalar tensor
    }

    @Test
    fun testSqueeze_NoSingletonDimensions() {
        val tensor = createTensor(Shape(2, 3, 4))

        val result = ops.squeeze(tensor, dim = null)

        assertEquals(Shape(2, 3, 4), result.shape)  // No change
    }

    @Test
    fun testSqueeze_NonSingletonDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.squeeze(tensor, dim = 1)  // Dimension 1 has size 3, not 1
        }
        assertTrue(exception.message?.contains("Only dimensions of size 1 can be squeezed") == true)
    }

    // ============== UNSQUEEZE OPERATION TESTS ==============

    @Test
    fun testUnsqueeze_AddDimensionAtBeginning() {
        val tensor = createTensor(Shape(3, 4))

        val result = ops.unsqueeze(tensor, dim = 0)

        assertEquals(Shape(1, 3, 4), result.shape)
        assertEquals(3, result.rank)
    }

    @Test
    fun testUnsqueeze_AddDimensionInMiddle() {
        val tensor = createTensor(Shape(2, 4))

        val result = ops.unsqueeze(tensor, dim = 1)

        assertEquals(Shape(2, 1, 4), result.shape)
    }

    @Test
    fun testUnsqueeze_AddDimensionAtEnd() {
        val tensor = createTensor(Shape(2, 3))

        val result = ops.unsqueeze(tensor, dim = 2)

        assertEquals(Shape(2, 3, 1), result.shape)
    }

    @Test
    fun testUnsqueeze_NegativeIndex() {
        val tensor = createTensor(Shape(2, 3))

        val result = ops.unsqueeze(tensor, dim = -1)

        assertEquals(Shape(2, 3, 1), result.shape)  // Add at end
    }

    @Test
    fun testUnsqueeze_ScalarTensor() {
        val tensor = createTensor(Shape(1))  // Scalar

        val result = ops.unsqueeze(tensor, dim = 0)

        assertEquals(Shape(1, 1), result.shape)
        assertEquals(2, result.rank)
    }

    @Test
    fun testUnsqueeze_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.unsqueeze(tensor, dim = 5)  // Out of bounds for new rank 3
        }
        assertTrue(exception.message?.contains("out of bounds") == true)
    }

    // ============== TRANSPOSE OPERATION TESTS ==============

    @Test
    fun testTranspose_2DMatrix() {
        val tensor = createTensor(Shape(3, 4))

        val result = ops.transpose(tensor)

        assertEquals(Shape(4, 3), result.shape)  // Dimensions swapped
    }

    @Test
    fun testTranspose_HighDimensionalTensor() {
        val tensor = createTensor(Shape(2, 3, 4, 5))

        val result = ops.transpose(tensor)

        assertEquals(Shape(2, 3, 5, 4), result.shape)  // Last two dims swapped
    }

    @Test
    fun testTranspose_SquareMatrix() {
        val tensor = createTensor(Shape(5, 5))

        val result = ops.transpose(tensor)

        assertEquals(Shape(5, 5), result.shape)  // Shape unchanged for square
    }

    @Test
    fun testTranspose_1DTensor_ThrowsException() {
        val tensor = createTensor(Shape(5))

        val exception = assertFailsWith<IllegalArgumentException> {
            ops.transpose(tensor)  // 1D tensor cannot be transposed
        }
        assertTrue(exception.message?.contains("at least 2 dimensions") == true)
    }

    // ============== COMPLEX SHAPE OPERATION COMBINATIONS ==============

    @Test
    fun testComplexShapeOperations_CombinedOperations() {
        // Test chaining multiple shape operations
        val original = createTensor(Shape(2, 3, 4))

        // Unsqueeze -> Concat -> Split -> Squeeze
        val unsqueezed = ops.unsqueeze(original, dim = 0)  // (1, 2, 3, 4)
        assertEquals(Shape(1, 2, 3, 4), unsqueezed.shape)

        val concat = ops.concat(listOf(unsqueezed, unsqueezed), dim = 0)  // (2, 2, 3, 4)
        assertEquals(Shape(2, 2, 3, 4), concat.shape)

        val split = ops.split(concat, splitSize = 1, dim = 0)  // 2 tensors of (1, 2, 3, 4)
        assertEquals(2, split.size)
        split.forEach {
            assertEquals(Shape(1, 2, 3, 4), it.shape)
        }

        val squeezed = ops.squeeze(split[0], dim = 0)  // (2, 3, 4)
        assertEquals(Shape(2, 3, 4), squeezed.shape)
        assertEquals(original.shape, squeezed.shape)  // Back to original shape
    }

    @Test
    fun testShapeOperations_VolumeConservation() {
        val original = createTensor(Shape(2, 3, 4))
        val originalVolume = original.volume

        // Operations that should preserve volume
        val reshaped = ops.reshape(original, Shape(6, 4))
        assertEquals(originalVolume, reshaped.volume)

        val flattened = ops.flatten(original)
        assertEquals(originalVolume, flattened.volume)

        val transposed = ops.transpose(original)
        assertEquals(originalVolume, transposed.volume)

        val unsqueezed = ops.unsqueeze(original, dim = 1)
        assertEquals(originalVolume, unsqueezed.volume)

        val squeezed = ops.squeeze(unsqueezed, dim = 1)
        assertEquals(originalVolume, squeezed.volume)
    }

    @Test
    fun testShapeOperations_EdgeCasesWithEmptyDimensions() {
        // Test with tensor containing dimension of size 0
        val emptyTensor = createTensor(Shape(2, 0, 3))
        assertEquals(0, emptyTensor.volume)

        // Operations should handle empty dimensions correctly
        val reshaped = ops.reshape(emptyTensor, Shape(0))
        assertEquals(Shape(0), reshaped.shape)

        val flattened = ops.flatten(emptyTensor)
        assertEquals(Shape(0), flattened.shape)
    }
}