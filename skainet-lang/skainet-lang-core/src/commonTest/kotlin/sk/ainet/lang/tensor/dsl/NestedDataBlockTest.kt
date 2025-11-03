package sk.ainet.lang.tensor.dsl

import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NestedDataBlockTest {

    @Test
    fun testNestedDataBlock() {
        val tensor = data<FP32, Float> {
            // create vector
            val vectorFP32 = tensor<FP32, Float> {
                shape(3, 3, 3) {
                    //fromArray(1.0f,2.0f,3.0f,4.0f,5.0f,6.0f)
                    zeros()
                }
            }

            val nestedVector = data<Int8, Byte> {
                vector<Int8, Byte>(10) {
                    ones()
                }
            }
            assertEquals(1, nestedVector.data[9])
            assertEquals(10, nestedVector.volume)
            assertEquals(1, nestedVector.rank)

            assertTrue(nestedVector is Tensor<Int8, Byte>)
            vectorFP32
        }
        assertEquals(0.0f, tensor.data[0, 0, 0])
        assertEquals(27, tensor.volume)
        assertEquals(3, tensor.rank)
    }

    @Test
    fun testsingleDataBlockReturningSafeTypedTensor() {
        val result = data<FP32, Float> {
            // create vector
            val vectorFP32 = tensor {
                shape(3, 2) {
                    //fromArray(1,2,3,4,5,6)
                    zeros()
                }
            }

            val nestedVector = data<Int8, Int> {
                vector(10) {
                    ones()
                }
            }
            assertTrue(nestedVector is Tensor<Int8, Int>)
            vectorFP32
        }
        assertTrue(result is Tensor<FP32, Float>)
    }
}