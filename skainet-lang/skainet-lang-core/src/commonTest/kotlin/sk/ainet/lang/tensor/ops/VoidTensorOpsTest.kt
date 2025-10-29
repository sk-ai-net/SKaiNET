package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class VoidTensorOpsTest {
    
    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()
    
    // Helper function to create test tensors
    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun testAdd_SameShape_Success() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(2, 3))
        
        val result = ops.add(a, b)
        
        assertEquals(Shape(2, 3), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testAdd_DifferentShape_ThrowsException() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(3, 2))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.add(a, b)
        }
        assertEquals(exception.message?.contains("Shape mismatch for addition"), true)
    }
    
    @Test
    fun testSubtract_SameShape_Success() {
        val a = createTensor(Shape(4, 1))
        val b = createTensor(Shape(4, 1))
        
        val result = ops.subtract(a, b)
        
        assertEquals(Shape(4, 1), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMultiply_ScalarTensors_Success() {
        val a = createTensor(Shape(1))
        val b = createTensor(Shape(1))
        
        val result = ops.multiply(a, b)
        
        assertEquals(Shape(1), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testDivide_DifferentShape_ThrowsException() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(3, 2))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.divide(a, b)
        }
        assertEquals(exception.message?.contains("Shape mismatch for division"), true)
    }
    
    @Test
    fun testMatmul_2D_Success() {
        val a = createTensor(Shape(3, 4))  // 3x4
        val b = createTensor(Shape(4, 5))  // 4x5
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(3, 5), result.shape)  // 3x5
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_BatchedMatrices_Success() {
        val a = createTensor(Shape(2, 3, 4))  // batch=2, 3x4 matrices
        val b = createTensor(Shape(2, 4, 5))  // batch=2, 4x5 matrices
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(2, 3, 5), result.shape)  // batch=2, 3x5 matrices
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_IncompatibleInnerDimensions_ThrowsException() {
        val a = createTensor(Shape(3, 4))  // 3x4
        val b = createTensor(Shape(5, 2))  // 5x2 (incompatible: 4 != 5)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertEquals(exception.message?.contains("inner dimensions must match"), true)
    }
    
    @Test
    fun testMatmul_1D_1D_DotProduct() {
        val a = createTensor(Shape(4))  // (k,)
        val b = createTensor(Shape(4))  // (k,)
        val result = ops.matmul(a, b)
        // Result is scalar (0-D)
        assertEquals(0, result.rank)
        assertEquals(Shape(), result.shape)
    }

    // Additional comprehensive matmul tests covering all specification cases
    
    @Test
    fun testMatmul_VectorTimesBatchedMatrices_Broadcast() {
        // (k,) @ (b, k, n) -> (b, n)
        val a = createTensor(Shape(4))        // k=4
        val b = createTensor(Shape(3, 4, 2))  // b=3, k=4, n=2
        val result = ops.matmul(a, b)
        assertEquals(Shape(3, 2), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_1D_2D_RowVectorTimesMatrix() {
        // (k,) @ (k, n) -> (n)
        val a = createTensor(Shape(4))      // (k,)
        val b = createTensor(Shape(4, 2))   // (k, n)
        val result = ops.matmul(a, b)
        assertEquals(Shape(2), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_2D_1D_MatrixTimesColumnVector() {
        // (..., m, k) @ (k,) -> (..., m)
        val a = createTensor(Shape(3, 4))   // (m, k)
        val b = createTensor(Shape(4))      // (k,)
        val result = ops.matmul(a, b)
        assertEquals(Shape(3), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testTranspose_2D_Success() {
        val tensor = createTensor(Shape(3, 4))
        
        val result = ops.transpose(tensor)
        
        assertEquals(Shape(4, 3), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testTranspose_3D_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.transpose(tensor)
        
        assertEquals(Shape(2, 4, 3), result.shape)  // Last two dimensions swapped
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testTranspose_1D_ThrowsException() {
        val tensor = createTensor(Shape(5))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.transpose(tensor)
        }
        assertEquals(exception.message?.contains("at least 2 dimensions"), true)
    }
    
    @Test
    fun testReshape_SameVolume_Success() {
        val tensor = createTensor(Shape(2, 3, 4))  // Volume = 24
        val newShape = Shape(6, 4)  // Volume = 24
        
        val result = ops.reshape(tensor, newShape)
        
        assertEquals(newShape, result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testReshape_DifferentVolume_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))  // Volume = 6
        val newShape = Shape(3, 3)  // Volume = 9
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.reshape(tensor, newShape)
        }
        assertEquals(exception.message?.contains("volume mismatch"), true)
    }
    
    @Test
    fun testFlatten_MiddleDimensions_Success() {
        val tensor = createTensor(Shape(2, 3, 4, 5))  // 4D tensor
        
        val result = ops.flatten(tensor, 1, 2)  // Flatten dimensions 1 and 2
        
        assertEquals(Shape(2, 12, 5), result.shape)  // 2, (3*4), 5
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testFlatten_AllDimensions_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.flatten(tensor, 0, -1)  // Flatten all dimensions (negative indexing)
        
        assertEquals(Shape(24), result.shape)  // 2*3*4 = 24
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testFlatten_InvalidDimensions_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 5, 6)  // Out of bounds
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testFlatten_StartGreaterThanEnd_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 2, 1)  // start > end
        }
        assertEquals(exception.message?.contains("must be <="), true)
    }
    
    @Test
    fun testActivationFunctions_PreserveShape() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val reluResult = ops.relu(tensor)
        val sigmoidResult = ops.sigmoid(tensor)
        val siluResult = ops.silu(tensor)
        val geluResult = ops.gelu(tensor)
        
        assertEquals(tensor.shape, reluResult.shape)
        assertEquals(tensor.shape, sigmoidResult.shape)
        assertEquals(tensor.shape, siluResult.shape)
        assertEquals(tensor.shape, geluResult.shape)
        assertEquals(FP32::class, reluResult.dtype)
        assertEquals(FP32::class, sigmoidResult.dtype)
        assertEquals(FP32::class, siluResult.dtype)
        assertEquals(FP32::class, geluResult.dtype)
    }
    
    @Test
    fun testSoftmax_ValidDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.softmax(tensor, 1)  // Apply softmax along dimension 1
        
        assertEquals(tensor.shape, result.shape)  // Softmax preserves shape
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSoftmax_NegativeDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.softmax(tensor, -1)  // Last dimension
        
        assertEquals(tensor.shape, result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSoftmax_InvalidDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.softmax(tensor, 5)  // Out of bounds
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testSum_AllDimensions_Scalar() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, null)  // Sum all dimensions
        
        assertEquals(Shape(1), result.shape)  // Scalar result
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_SpecificDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, 1)  // Sum along dimension 1
        
        assertEquals(Shape(2, 4), result.shape)  // Remove dimension 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_LastDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, -1)  // Sum along last dimension
        
        assertEquals(Shape(2, 3), result.shape)  // Remove last dimension
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_SingleDimensionTensor_Scalar() {
        val tensor = createTensor(Shape(5))
        
        val result = ops.sum(tensor, 0)  // Sum the only dimension
        
        assertEquals(Shape(1), result.shape)  // Result is scalar
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMean_SpecificDimension_Success() {
        val tensor = createTensor(Shape(3, 4, 5))
        
        val result = ops.mean(tensor, 2)  // Mean along dimension 2
        
        assertEquals(Shape(3, 4), result.shape)  // Remove dimension 2
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testReduction_InvalidDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.sum(tensor, 5)  // Out of bounds
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testVariance_AllDimensions_Scalar() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.variance(tensor, null)  // Variance over all dimensions
        
        assertEquals(Shape(1), result.shape)  // Scalar result
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testVariance_SpecificDimension_Success() {
        val tensor = createTensor(Shape(3, 4, 5))
        
        val result = ops.variance(tensor, 1)  // Variance along dimension 1
        
        assertEquals(Shape(3, 5), result.shape)  // Remove dimension 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testVariance_LastDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.variance(tensor, -1)  // Variance along last dimension
        
        assertEquals(Shape(2, 3), result.shape)  // Remove last dimension
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMathematicalFunctions_PreserveShape() {
        val tensor = createTensor(Shape(3, 4, 2))
        
        val sqrtResult = ops.sqrt(tensor)
        
        assertEquals(tensor.shape, sqrtResult.shape)  // sqrt preserves shape
        assertEquals(FP32::class, sqrtResult.dtype)
    }
    
    @Test
    fun testConvert_PreservesShape() {
        val floatTensor = createTensor(Shape(2, 3))
        
        val result = ops.convert(floatTensor, Int32)
        
        assertEquals(Shape(2, 3), result.shape)
        assertEquals(Int32::class, result.dtype)
    }
    
    @Test
    fun testEdgeCases_EmptyTensor() {
        val emptyTensor = createTensor(Shape(0))
        
        // Operations should work with empty tensors
        val reluResult = ops.relu(emptyTensor)
        assertEquals(Shape(0), reluResult.shape)
        
        val sumResult = ops.sum(emptyTensor, null)
        assertEquals(Shape(1), sumResult.shape)
    }
    
    @Test
    fun testEdgeCases_SingleElementTensor() {
        val scalarTensor = createTensor(Shape(1))
        
        // Test element-wise operations with scalars
        val addResult = ops.add(scalarTensor, scalarTensor)
        assertEquals(Shape(1), addResult.shape)
        
        val sumResult = ops.sum(scalarTensor, null)
        assertEquals(Shape(1), sumResult.shape)
    }
    
    @Test
    fun testMatmul_BatchDimensionBroadcasting() {
        val a = createTensor(Shape(1, 2, 3))  // Batch=1, 2x3
        val b = createTensor(Shape(4, 3, 5))  // Batch=4, 3x5
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(4, 2, 5), result.shape)  // Broadcast batch dimension
    }
    
    @Test
    fun testMatmul_IncompatibleBatchDimensions_ThrowsException() {
        val a = createTensor(Shape(2, 3, 4))  // Batch=2
        val b = createTensor(Shape(3, 4, 5))  // Batch=3 (incompatible)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertEquals(exception.message?.contains("batch dimension mismatch"), true)
    }

    @Test
    fun testMatmul_ComplexBroadcasting_MultipleBatchDimensions() {
        val a = createTensor(Shape(2, 1, 3, 4))  // Batch=(2, 1), 3×4 matrices
        val b = createTensor(Shape(1, 5, 4, 6))  // Batch=(1, 5), 4×6 matrices
        
        val result = ops.matmul(a, b)
        
        // Batch dimensions broadcast: (2, 1) × (1, 5) -> (2, 5)
        // Matrix dimensions: 3×4 × 4×6 -> 3×6
        assertEquals(Shape(2, 5, 3, 6), result.shape)
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_SingleBatchDimensionBroadcasting() {
        val a = createTensor(Shape(2, 3))    // No batch, 2×3 matrix
        val b = createTensor(Shape(5, 3, 4)) // Batch=5, 3×4 matrices
        
        val result = ops.matmul(a, b)
        
        // Should broadcast to (5, 2, 4)
        assertEquals(Shape(5, 2, 4), result.shape)
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_BatchedMatricesTimesVector_Broadcast() {
        // (b, m, k) @ (k,) -> (b, m)
        val a = createTensor(Shape(3, 2, 4))  // b=3, m=2, k=4
        val b = createTensor(Shape(4))        // k=4
        val result = ops.matmul(a, b)
        assertEquals(Shape(3, 2), result.shape)
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_HighDimensionalBroadcasting() {
        val a = createTensor(Shape(2, 1, 1, 3, 4))  // Complex batch dimensions
        val b = createTensor(Shape(1, 3, 5, 4, 2))  // Different batch dimensions
        
        val result = ops.matmul(a, b)
        
        // Batch broadcast: (2, 1, 1) × (1, 3, 5) -> (2, 3, 5)
        // Matrix: 3×4 × 4×2 -> 3×2
        assertEquals(Shape(2, 3, 5, 3, 2), result.shape)
    }

    @Test
    fun testMatmul_EdgeCase_MinimumMatrixSize() {
        val a = createTensor(Shape(1, 1))  // 1×1 matrix
        val b = createTensor(Shape(1, 1))  // 1×1 matrix
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(1, 1), result.shape)  // 1×1 result
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_EdgeCase_VeryLargeInnerDimension() {
        val a = createTensor(Shape(2, 1000))  // 2×1000
        val b = createTensor(Shape(1000, 3))  // 1000×3
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(2, 3), result.shape)  // 2×3 result
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_ShapeValidation_ZeroDimension() {
        val a = createTensor(Shape(0, 4))  // Empty matrix
        val b = createTensor(Shape(4, 3))
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(0, 3), result.shape)  // Empty result
        assertEquals(FP32::class, result.dtype)
    }

    @Test
    fun testMatmul_ShapeValidation_InnerDimensionMismatch_DetailedError() {
        val a = createTensor(Shape(3, 7))  // 3×7
        val b = createTensor(Shape(5, 2))  // 5×2 (7 != 5)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertEquals(exception.message?.contains("7 vs 5"), true)
        assertEquals(exception.message?.contains("inner dimensions must match"), true)
    }

    @Test
    fun testMatmul_BatchedMatrices_DifferentRanks() {
        val a = createTensor(Shape(1, 3, 4))     // 3D tensor with batch=1
        val b = createTensor(Shape(2, 4, 5))     // 3D tensor with batch=2
        
        val result = ops.matmul(a, b)
        
        // Batch dimensions should broadcast: 1 × 2 -> 2
        // Matrix dimensions: 3×4 × 4×5 -> 3×5
        assertEquals(Shape(2, 3, 5), result.shape)
    }

    @Test
    fun testMatmul_BatchedMatrices_IncompatibleBatchBroadcast() {
        val a = createTensor(Shape(3, 2, 4))  // Batch=3
        val b = createTensor(Shape(5, 4, 6))  // Batch=5 (can't broadcast 3 and 5)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertEquals(exception.message?.contains("batch dimension mismatch"), true)
        assertEquals(exception.message?.contains("3 vs 5"), true)
    }
    
    @Test
    fun testLargeTensorShapes() {
        // Test with larger tensor shapes
        val largeTensor = createTensor(Shape(100, 200))
        
        val transposeResult = ops.transpose(largeTensor)
        assertEquals(Shape(200, 100), transposeResult.shape)
        
        val reshapeResult = ops.reshape(largeTensor, Shape(20, 1000))
        assertEquals(Shape(20, 1000), reshapeResult.shape)
    }

    // Tests for new YOLO shape operations
    
    @Test
    fun testConcat_SameDimensions_Success() {
        val tensor1 = createTensor(Shape(2, 3, 4))
        val tensor2 = createTensor(Shape(2, 3, 4))
        val tensors = listOf(tensor1, tensor2)
        
        val result = ops.concat(tensors, 0)  // Concatenate along dimension 0
        
        assertEquals(Shape(4, 3, 4), result.shape)  // First dimension doubled
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testConcat_DifferentSizesInConcatDim_Success() {
        val tensor1 = createTensor(Shape(2, 3, 4))
        val tensor2 = createTensor(Shape(2, 5, 4))  // Different size in dim 1
        val tensors = listOf(tensor1, tensor2)
        
        val result = ops.concat(tensors, 1)  // Concatenate along dimension 1
        
        assertEquals(Shape(2, 8, 4), result.shape)  // 3 + 5 = 8 in dimension 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testConcat_NegativeDimension_Success() {
        val tensor1 = createTensor(Shape(2, 3))
        val tensor2 = createTensor(Shape(2, 3))
        val tensors = listOf(tensor1, tensor2)
        
        val result = ops.concat(tensors, -1)  // Concatenate along last dimension
        
        assertEquals(Shape(2, 6), result.shape)  // Last dimension doubled
    }
    
    @Test
    fun testConcat_EmptyList_ThrowsException() {
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.concat(emptyList<VoidOpsTensor<FP32, Float>>(), 0)
        }
        assertEquals(exception.message?.contains("empty list"), true)
    }
    
    @Test
    fun testConcat_IncompatibleShapes_ThrowsException() {
        val tensor1 = createTensor(Shape(2, 3, 4))
        val tensor2 = createTensor(Shape(2, 5, 6))  // Different in non-concat dimension
        val tensors = listOf(tensor1, tensor2)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.concat(tensors, 0)  // Try to concat along dim 0
        }
        assertEquals(exception.message?.contains("same shape except in the concatenation dimension"), true)
    }
    
    @Test
    fun testConcat_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))
        val tensors = listOf(tensor)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.concat(tensors, 5)  // Out of bounds dimension
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testSplit_EvenSplit_Success() {
        val tensor = createTensor(Shape(4, 6, 8))
        
        val result = ops.split(tensor, 2, 1)  // Split dimension 1 into chunks of size 2
        
        assertEquals(3, result.size)  // 6 / 2 = 3 chunks
        result.forEach { chunk ->
            assertEquals(Shape(4, 2, 8), chunk.shape)
            assertEquals(FP32::class, chunk.dtype)
        }
    }
    
    @Test
    fun testSplit_LastDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 12))
        
        val result = ops.split(tensor, 3, -1)  // Split last dimension
        
        assertEquals(4, result.size)  // 12 / 3 = 4 chunks
        result.forEach { chunk ->
            assertEquals(Shape(2, 3, 3), chunk.shape)
        }
    }
    
    @Test
    fun testSplit_NotDivisible_ThrowsException() {
        val tensor = createTensor(Shape(2, 5, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.split(tensor, 3, 1)  // 5 is not divisible by 3
        }
        assertEquals(exception.message?.contains("not divisible"), true)
    }
    
    @Test
    fun testSplit_InvalidSplitSize_ThrowsException() {
        val tensor = createTensor(Shape(2, 4, 6))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.split(tensor, 0, 1)  // Split size must be positive
        }
        assertEquals(exception.message?.contains("must be positive"), true)
    }
    
    @Test
    fun testSplit_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.split(tensor, 1, 5)  // Out of bounds dimension
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testSqueeze_AllOnes_Success() {
        val tensor = createTensor(Shape(2, 1, 3, 1, 4))
        
        val result = ops.squeeze(tensor, null)  // Remove all dimensions of size 1
        
        assertEquals(Shape(2, 3, 4), result.shape)  // Only dimensions > 1 remain
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSqueeze_SpecificDimension_Success() {
        val tensor = createTensor(Shape(2, 1, 3, 4))
        
        val result = ops.squeeze(tensor, 1)  // Remove dimension 1 (size 1)
        
        assertEquals(Shape(2, 3, 4), result.shape)  // Dimension 1 removed
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSqueeze_NegativeDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 1))
        
        val result = ops.squeeze(tensor, -1)  // Remove last dimension (size 1)
        
        assertEquals(Shape(2, 3), result.shape)
    }
    
    @Test
    fun testSqueeze_AllOnesScalar_Success() {
        val tensor = createTensor(Shape(1, 1, 1))
        
        val result = ops.squeeze(tensor, null)  // Remove all dimensions
        
        assertEquals(Shape(1), result.shape)  // Becomes scalar
    }
    
    @Test
    fun testSqueeze_NonOneDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.squeeze(tensor, 1)  // Dimension 1 has size 3, not 1
        }
        assertEquals(exception.message?.contains("Only dimensions of size 1 can be squeezed"), true)
    }
    
    @Test
    fun testSqueeze_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 1, 3))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.squeeze(tensor, 5)  // Out of bounds dimension
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
    
    @Test
    fun testUnsqueeze_PositiveDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.unsqueeze(tensor, 1)  // Add dimension at position 1
        
        assertEquals(Shape(2, 1, 3, 4), result.shape)  // New dimension of size 1 at position 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testUnsqueeze_NegativeDimension_Success() {
        val tensor = createTensor(Shape(2, 3))
        
        val result = ops.unsqueeze(tensor, -1)  // Add dimension at end
        
        assertEquals(Shape(2, 3, 1), result.shape)  // New dimension added at end
    }
    
    @Test
    fun testUnsqueeze_AtBeginning_Success() {
        val tensor = createTensor(Shape(3, 4))
        
        val result = ops.unsqueeze(tensor, 0)  // Add dimension at beginning
        
        assertEquals(Shape(1, 3, 4), result.shape)
    }
    
    @Test
    fun testUnsqueeze_Scalar_Success() {
        val tensor = createTensor(Shape(1))  // Scalar tensor
        
        val result = ops.unsqueeze(tensor, 0)  // Add dimension at beginning
        
        assertEquals(Shape(1, 1), result.shape)
    }
    
    @Test
    fun testUnsqueeze_OutOfBounds_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.unsqueeze(tensor, 5)  // Out of bounds for new rank (3)
        }
        assertEquals(exception.message?.contains("out of bounds"), true)
    }
}