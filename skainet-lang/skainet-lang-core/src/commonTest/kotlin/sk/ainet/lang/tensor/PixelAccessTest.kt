package sk.ainet.lang.tensor

import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.*
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Comprehensive test suite for pixel-by-pixel tensor access functionality.
 * Demonstrates how to access every pixel separately in 4D tensors (BCHW format).
 */
class PixelAccessTest {

    @Test
    fun testPixelByPixelAccess() {
        println("[DEBUG_LOG] Testing pixel-by-pixel access for 4D tensor")

        data {
            // Create a 4D tensor in BCHW format: B=2, C=3, H=4, W=4
            val tensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Deterministic pattern for validation: b*1000 + c*100 + h*10 + w
                        (indices[0] * 1000 + indices[1] * 100 + indices[2] * 10 + indices[3]).toFloat()
                    }
                }
            }

            var pixelCount = 0
            for (b in 0 until 2) {
                for (c in 0 until 3) {
                    for (h in 0 until 4) {
                        for (w in 0 until 4) {
                            val actual = tensor.data[b, c, h, w]
                            val expected = (b * 1000 + c * 100 + h * 10 + w).toFloat()
                            assertEquals(expected, actual, 0.001f, "Mismatch at [b=$b,c=$c,h=$h,w=$w]")
                            pixelCount++
                        }
                    }
                }
            }

            // Ensure we actually visited every pixel: 2*3*4*4 = 96
            assertEquals(2 * 3 * 4 * 4, pixelCount, "Visited pixel count mismatch")
        }

        println("[DEBUG_LOG] Successfully accessed and verified all ${2 * 3 * 4 * 4} pixels")
    }

    @Test
    fun testSpecificPixelPatterns() {
        println("[DEBUG_LOG] Testing specific pixel access patterns")

        data {

            // Create a 4D tensor with a different pattern
            val tensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Pattern: sum of all indices
                        (indices[0] + indices[1] + indices[2] + indices[3]).toFloat()
                    }
                }
            }

            // Test corner pixels
            assertEquals(0.0f, tensor.data[0, 0, 0, 0]) // Top-left corner of first batch/channel
            assertEquals(
                9.0f,
                tensor.data[1, 2, 3, 3]
            ) // Bottom-right corner of second batch, third channel (1+2+3+3=9)

            // Test center pixels
            assertEquals(4.0f, tensor.data[0, 1, 1, 2]) // Center-ish pixel
            assertEquals(6.0f, tensor.data[1, 1, 2, 2]) // Another center pixel (1+1+2+2=6)

            // Test edge pixels
            assertEquals(3.0f, tensor.data[0, 0, 0, 3]) // Right edge, first row
            assertEquals(6.0f, tensor.data[0, 0, 3, 3]) // Bottom-right corner, first batch/channel

            println("[DEBUG_LOG] All specific pixel patterns verified successfully")
        }

    }

    @Test
    fun testFirstBatchFirstChannelPixels() {
        println("[DEBUG_LOG] Testing comprehensive access to first batch, first channel")

        data {
            val imageTensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Initialize with meaningful pattern: batch*1000 + channel*100 + height*10 + width
                        (indices[0] * 1000 + indices[1] * 100 + indices[2] * 10 + indices[3]).toFloat()
                    }
                }
            }

            println("[DEBUG_LOG] Sample values from tensor[0,0,:,:] (first batch, first channel):")
            val pixelValues = mutableListOf<Float>()

            for (h in 0 until 4) {
                val rowValues = mutableListOf<Float>()
                for (w in 0 until 4) {
                    val value = imageTensor.data[0, 0, h, w]
                    rowValues.add(value)
                    pixelValues.add(value)
                    print("${value.toInt()}\t")
                }
                println()

                // Verify the row values match expected pattern
                for ((w, value) in rowValues.withIndex()) {
                    val expectedValue = (h * 10 + w).toFloat()
                    assertEquals(
                        expectedValue, value, 0.001f,
                        "Row $h, Col $w: expected $expectedValue, got $value"
                    )
                }
            }

            // Verify we captured all 16 pixels for the first batch/channel
            assertEquals(16, pixelValues.size)

            // Verify specific expected values
            assertEquals(0.0f, pixelValues[0])   // [0,0]
            assertEquals(33.0f, pixelValues[15]) // [3,3] = 3*10 + 3 = 33
            assertEquals(13.0f, pixelValues[7])  // [1,3] = 1*10 + 3 = 13 (index 7 is row 1, col 3)

            println("[DEBUG_LOG] First batch, first channel pixels verified successfully")
        }
    }

    @Test
    fun testAllChannelsFromFirstBatch() {
        println("[DEBUG_LOG] Testing access to all channels from first batch")

        data {
            val imageTensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Initialize with meaningful pattern: batch*1000 + channel*100 + height*10 + width
                        (indices[0] * 1000 + indices[1] * 100 + indices[2] * 10 + indices[3]).toFloat()
                    }
                }
            }

            // Test accessing all channels for the first batch, first pixel [0,0]
            for (channel in 0 until 3) {
                val value = imageTensor.data[0, channel, 0, 0]
                val expectedValue = (channel * 100).toFloat() // batch=0, h=0, w=0
                assertEquals(
                    expectedValue, value, 0.001f,
                    "Channel $channel pixel [0,0]: expected $expectedValue, got $value"
                )

                println("[DEBUG_LOG] Channel $channel, pixel [0,0] = $value")
            }

            // Test accessing all channels for a middle pixel [2,2]
            for (channel in 0 until 3) {
                val value = imageTensor.data[0, channel, 2, 2]
                val expectedValue = (channel * 100 + 2 * 10 + 2).toFloat() // batch=0, h=2, w=2
                assertEquals(
                    expectedValue, value, 0.001f,
                    "Channel $channel pixel [2,2]: expected $expectedValue, got $value"
                )

                println("[DEBUG_LOG] Channel $channel, pixel [2,2] = $value")
            }

            println("[DEBUG_LOG] All channels from first batch verified successfully")
        }
    }

    @Test
    fun testBatchSeparation() {
        println("[DEBUG_LOG] Testing pixel access across different batches")
        data {


            val imageTensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Initialize with meaningful pattern: batch*1000 + channel*100 + height*10 + width
                        (indices[0] * 1000 + indices[1] * 100 + indices[2] * 10 + indices[3]).toFloat()
                    }
                }
            }

            // Compare same pixel location across different batches
            val pixel00Batch0 = imageTensor.data[0, 0, 0, 0] // Should be 0
            val pixel00Batch1 = imageTensor.data[1, 0, 0, 0] // Should be 1000

            assertEquals(0.0f, pixel00Batch0, 0.001f)
            assertEquals(1000.0f, pixel00Batch1, 0.001f)

            // Compare same pixel location, same channel, different batches
            val pixel22Ch1Batch0 = imageTensor.data[0, 1, 2, 2] // 0*1000 + 1*100 + 2*10 + 2 = 122
            val pixel22Ch1Batch1 = imageTensor.data[1, 1, 2, 2] // 1*1000 + 1*100 + 2*10 + 2 = 1122

            assertEquals(122.0f, pixel22Ch1Batch0, 0.001f)
            assertEquals(1122.0f, pixel22Ch1Batch1, 0.001f)

            println("[DEBUG_LOG] Batch separation verified: batch 0 pixel [1,2,2] = $pixel22Ch1Batch0, batch 1 pixel [1,2,2] = $pixel22Ch1Batch1")

            // Verify the difference is exactly 1000 (the batch multiplier)
            val batchDifference = pixel22Ch1Batch1 - pixel22Ch1Batch0
            assertEquals(1000.0f, batchDifference, 0.001f)

            println("[DEBUG_LOG] Batch separation test completed successfully")
        }
    }

    @Test
    fun testEdgeAndCornerPixels() {
        println("[DEBUG_LOG] Testing edge and corner pixel access")
        data {


            val imageTensor = tensor<FP32, Float> {
                shape(2, 3, 4, 4) { _ ->
                    init { indices ->
                        // Simple pattern: just the sum of indices
                        (indices[0] + indices[1] + indices[2] + indices[3]).toFloat()
                    }
                }
            }

            // Test all four corners of first batch, first channel
            val topLeft = imageTensor.data[0, 0, 0, 0]     // 0+0+0+0 = 0
            val topRight = imageTensor.data[0, 0, 0, 3]    // 0+0+0+3 = 3
            val bottomLeft = imageTensor.data[0, 0, 3, 0]  // 0+0+3+0 = 3
            val bottomRight = imageTensor.data[0, 0, 3, 3] // 0+0+3+3 = 6

            assertEquals(0.0f, topLeft)
            assertEquals(3.0f, topRight)
            assertEquals(3.0f, bottomLeft)
            assertEquals(6.0f, bottomRight)

            // Test edge pixels (middle of each edge)
            val topEdge = imageTensor.data[0, 0, 0, 2]    // 0+0+0+2 = 2
            val bottomEdge = imageTensor.data[0, 0, 3, 2] // 0+0+3+2 = 5
            val leftEdge = imageTensor.data[0, 0, 2, 0]   // 0+0+2+0 = 2
            val rightEdge = imageTensor.data[0, 0, 2, 3]  // 0+0+2+3 = 5

            assertEquals(2.0f, topEdge)
            assertEquals(5.0f, bottomEdge)
            assertEquals(2.0f, leftEdge)
            assertEquals(5.0f, rightEdge)

            println("[DEBUG_LOG] All corner and edge pixels verified successfully")
            println("[DEBUG_LOG] Corners: TL=$topLeft, TR=$topRight, BL=$bottomLeft, BR=$bottomRight")
            println("[DEBUG_LOG] Edges: T=$topEdge, B=$bottomEdge, L=$leftEdge, R=$rightEdge")
        }
    }
}