package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Test for the new modern DSL syntax
 */
class TensorDSLTest {

    @Test
    fun testTensorDSLSyntaxWithOnes() {
        data {

            val vector = tensor<FP32, Float> {
                shape(5) { 
                    ones()
                }
            }

            assertNotNull(vector)
            assertEquals(Shape(5), vector.shape)
            // All elements should be 1.0f
            for (i in 0 until vector.volume) {
                assertEquals(1.0f, vector.data[i])
            }
        }

    }

    @Test
    fun testTensorDSLSyntaxWithZeros() {
        data {

            val matrix = tensor<FP32, Float> {
                shape(3, 4) { 
                    zeros()
                }
            }

            assertNotNull(matrix)
            assertEquals(Shape(3, 4), matrix.shape)
            // All elements should be 0.0f
            for (r in 0 until 3) {
                for (c in 0 until 4) {
                    assertEquals(0.0f, matrix.data[r, c])
                }
            }
        }
    }

    @Test
    fun testTensorDSLSyntaxWithFull() {
        data {

            val vector = tensor<FP32, Float> {
                shape(5) { 
                    full(2.5f)
                }
            }

            assertNotNull(vector)
            assertEquals(Shape(5), vector.shape)
            // All elements should be 2.5f
            for (i in 0 until 5) {
                assertEquals(2.5f, vector.data[i])
            }
        }
    }

    @Test
    fun testTensorDSLSyntaxWithRandn() {
        data {
            val matrix = tensor<FP32, Float> {
                shape(2, 3) { 
                    randN(mean = 0.0f, std = 1.0f)
                }
            }

            assertNotNull(matrix)
            assertEquals(Shape(2, 3), matrix.shape)
        }
    }

    @Test
    fun testTensorDSLSyntaxWithUniform() {
        data {
            val vector = tensor<FP32, Float> {
                shape(10) { 
                    uniform(min = -1.0f, max = 1.0f)
                }
            }
            assertNotNull(vector)
            assertEquals(Shape(10), vector.shape)
        }
    }

    @Test
    fun testTensorDSLSyntaxWithCustomInit() {
        data {
            val vector = tensor<FP32, Float> {
                shape(5) { 
                    init { indices -> indices[0].toFloat() }
                }
            }
            assertNotNull(vector)
            assertEquals(Shape(5), vector.shape)
            // Check that values equal to their index
            for (i in 0 until 5) {
                assertEquals(i.toFloat(), vector.data[i])
            }
        }
    }
}