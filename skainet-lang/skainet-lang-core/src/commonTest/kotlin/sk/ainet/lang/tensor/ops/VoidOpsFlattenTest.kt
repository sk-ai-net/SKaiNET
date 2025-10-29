package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Dedicated unit tests for flatten implementation in VoidOps.
 * Important focus on resulting shapes verification as requested.
 */
class VoidOpsFlattenTest {
    
    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()
    
    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun testFlatten_1D_AlreadyFlat_PreservesShape() {
        val tensor = createTensor(Shape(10))
        
        val result = ops.flatten(tensor, 0, 0)
        
        assertEquals(Shape(10), result.shape, "1D tensor flattening should preserve shape")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 1D flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_2D_FirstDimension_ResultShape() {
        val tensor = createTensor(Shape(3, 4))
        
        val result = ops.flatten(tensor, 0, 0) // Flatten only first dimension (no-op for 2D)
        
        assertEquals(Shape(3, 4), result.shape, "Flattening single dimension should preserve that dimension")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 2D first dim flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_2D_AllDimensions_ResultShape() {
        val tensor = createTensor(Shape(3, 4))
        
        val result = ops.flatten(tensor, 0, 1) // Flatten all dimensions
        
        assertEquals(Shape(12), result.shape, "2D tensor flattened should have volume as single dimension")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 2D all dims flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_3D_KeepBatchDimension_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4)) // Batch, Height, Width
        
        val result = ops.flatten(tensor, 1, 2) // Flatten spatial dimensions, keep batch
        
        assertEquals(Shape(2, 12), result.shape, "Should keep batch dim and flatten spatial: (2, 3*4)")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 3D batch preserve flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_3D_MiddleDimension_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.flatten(tensor, 1, 1) // Flatten only middle dimension
        
        assertEquals(Shape(2, 3, 4), result.shape, "Flattening single dimension should preserve original shape")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 3D middle dim flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_4D_ConvolutionalLayout_ResultShape() {
        val tensor = createTensor(Shape(1, 32, 28, 28)) // Batch, Channels, Height, Width
        
        val result = ops.flatten(tensor, 1, 3) // Flatten C*H*W, keep batch
        
        assertEquals(Shape(1, 25088), result.shape, "CNN flatten should keep batch: (1, 32*28*28)")
        assertEquals(25088, 32 * 28 * 28, "Verify calculation: 32*28*28")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 4D CNN flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_4D_PartialFlatten_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4, 5))
        
        val result = ops.flatten(tensor, 1, 2) // Flatten middle two dimensions
        
        assertEquals(Shape(2, 12, 5), result.shape, "Partial flatten: (2, 3*4, 5)")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 4D partial flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_5D_ComplexShape_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4, 5, 6))
        
        val result = ops.flatten(tensor, 2, 4) // Flatten last three dimensions
        
        assertEquals(Shape(2, 3, 120), result.shape, "Complex flatten: (2, 3, 4*5*6)")
        assertEquals(120, 4 * 5 * 6, "Verify calculation: 4*5*6")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] 5D complex flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_NegativeIndices_StartDim_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4, 5))
        
        val result = ops.flatten(tensor, -2, -1) // Flatten last two dimensions
        
        assertEquals(Shape(2, 3, 20), result.shape, "Negative indices flatten: (2, 3, 4*5)")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] Negative indices flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_NegativeIndices_AllDimensions_ResultShape() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.flatten(tensor, 0, -1) // From start to end (all dims)
        
        assertEquals(Shape(24), result.shape, "All dimensions flatten with negative end: volume 24")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] All dims with negative end: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_LargeDimensions_ResultShape() {
        val tensor = createTensor(Shape(10, 20, 30, 40))
        
        val result = ops.flatten(tensor, 1, 3) // Flatten last three dimensions
        
        assertEquals(Shape(10, 24000), result.shape, "Large flatten: (10, 20*30*40)")
        assertEquals(24000, 20 * 30 * 40, "Verify large calculation: 20*30*40")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] Large dims flatten: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_SingleDimensionTensor_ResultShape() {
        val tensor = createTensor(Shape(1))
        
        val result = ops.flatten(tensor, 0, 0)
        
        assertEquals(Shape(1), result.shape, "Single dimension tensor should remain unchanged")
        assertEquals(FP32::class, result.dtype)
        println("[DEBUG_LOG] Single dim tensor: ${tensor.shape} -> ${result.shape}")
    }

    @Test
    fun testFlatten_IdentityOperations_ResultShapes() {
        // Test various identity operations that shouldn't change shape
        val shapes = listOf(
            Shape(5),
            Shape(3, 4),
            Shape(2, 3, 4),
            Shape(1, 2, 3, 4)
        )
        
        shapes.forEach { shape ->
            val tensor = createTensor(shape)
            for (i in 0 until shape.rank) {
                val result = ops.flatten(tensor, i, i) // Flatten single dimension
                assertEquals(shape, result.shape, "Identity flatten on dim $i should preserve shape for $shape")
                println("[DEBUG_LOG] Identity flatten dim $i: $shape -> ${result.shape}")
            }
        }
    }

    @Test
    fun testFlatten_DifferentDataTypes_PreserveDtype() {
        val intDataFactory = DenseTensorDataFactory()
        val intOps = VoidTensorOps()
        val intData = intDataFactory.zeros<Int32, Int>(Shape(2, 3, 4), Int32::class)
        val intTensor = VoidOpsTensor(intData, Int32::class)
        
        val result = intOps.flatten(intTensor, 1, 2)
        
        assertEquals(Shape(2, 12), result.shape, "Int tensor flatten should work correctly")
        assertEquals(Int32::class, result.dtype, "Data type should be preserved")
        println("[DEBUG_LOG] Int32 tensor flatten: ${intTensor.shape} -> ${result.shape}")
    }

    // Error cases - testing bounds and validation

    @Test
    fun testFlatten_StartDimOutOfBounds_Positive_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4)) // rank = 3
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 5, 6) // Both out of bounds
        }
        assertEquals(exception.message?.contains("Start dimension 5 is out of bounds"), true)
        println("[DEBUG_LOG] Caught expected exception: ${exception.message}")
    }

    @Test
    fun testFlatten_EndDimOutOfBounds_Positive_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4)) // rank = 3
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 1, 5) // End dim out of bounds
        }
        assertTrue(exception.message?.contains("End dimension 5 is out of bounds") == true)
        println("[DEBUG_LOG] Caught expected exception: ${exception.message}")
    }

    @Test
    fun testFlatten_StartDimOutOfBounds_Negative_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4)) // rank = 3
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, -5, -1) // Start dim too negative
        }
        assertTrue(exception.message?.contains("Start dimension -5 is out of bounds") == true)
        println("[DEBUG_LOG] Caught expected exception: ${exception.message}")
    }

    @Test
    fun testFlatten_EndDimOutOfBounds_Negative_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4)) // rank = 3
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 0, -5) // End dim too negative
        }
        assertTrue(exception.message?.contains("End dimension -5 is out of bounds") == true)
        println("[DEBUG_LOG] Caught expected exception: ${exception.message}")
    }

    @Test
    fun testFlatten_StartGreaterThanEnd_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4, 5))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 3, 1) // start > end
        }
        assertTrue(exception.message?.contains("Start dimension 3 must be <= end dimension 1") == true)
        println("[DEBUG_LOG] Caught expected exception: ${exception.message}")
    }

    @Test
    fun testFlatten_BoundaryConditions_ResultShapes() {
        val tensor = createTensor(Shape(2, 3, 4, 5))
        
        // Test all valid boundary combinations
        val result1 = ops.flatten(tensor, 0, 3) // All dimensions
        assertEquals(Shape(120), result1.shape, "All dims: 2*3*4*5")
        
        val result2 = ops.flatten(tensor, 0, 0) // First dimension only
        assertEquals(Shape(2, 3, 4, 5), result2.shape, "First dim only")
        
        val result3 = ops.flatten(tensor, 3, 3) // Last dimension only
        assertEquals(Shape(2, 3, 4, 5), result3.shape, "Last dim only")
        
        val result4 = ops.flatten(tensor, -1, -1) // Last dimension with negative index
        assertEquals(Shape(2, 3, 4, 5), result4.shape, "Last dim negative index")
        
        println("[DEBUG_LOG] All boundary tests passed")
    }

    @Test
    fun testFlatten_VolumeConservation() {
        // Ensure that flattening always preserves tensor volume
        val testShapes = listOf(
            Shape(6),
            Shape(2, 3),
            Shape(2, 3, 4),
            Shape(1, 2, 3, 4),
            Shape(2, 1, 3, 1, 4)
        )
        
        testShapes.forEach { shape ->
            val tensor = createTensor(shape)
            val originalVolume = shape.volume
            
            // Test flattening all dimensions
            val flatResult = ops.flatten(tensor, 0, shape.rank - 1)
            val flatVolume = flatResult.shape.volume
            
            assertEquals(originalVolume, flatVolume, "Volume must be conserved for shape $shape")
            println("[DEBUG_LOG] Volume conservation verified: $shape volume=$originalVolume -> ${flatResult.shape} volume=$flatVolume")
        }
    }
}