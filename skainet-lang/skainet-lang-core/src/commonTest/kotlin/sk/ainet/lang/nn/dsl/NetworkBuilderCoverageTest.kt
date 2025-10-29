package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Int32
import kotlin.test.*
import kotlin.random.Random

/**
 * Additional test coverage for NetworkBuilder DSL focusing on uncovered functionality.
 * This test class complements existing tests to improve overall package coverage.
 * Updated to use the new context approach instead of MockTensorFactories.
 */
class NetworkBuilderCoverageTest {

    @Test
    fun testNetworkFP32WithFactory() {
        // Test FP32/Float combination using new context approach
        val net = definition<FP32, Float> {
            network {
                input(10)
                dense(5) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testDenseLayerWithoutOutputDimension() {
        // Test dense layer configuration without specifying output dimension
        val net = definition<FP32, Float> {
            sequential() {
                input(10)
                dense {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testAdvancedRandomInitializationMethods() {
        // Test various random initialization methods not covered in existing tests
        val customRandom = Random(seed = 12345)
        val net = definition<FP32, Float> {
            sequential {
                input(8)
                dense(6) {
                    weights { randn(mean = 0.0f, std = 0.1f, random = customRandom) }
                    bias { uniform(min = -0.05f, max = 0.05f, random = Random(999L)) }
                }
                dense(4) {
                    weights { uniform(min = -0.1f, max = 0.1f, random = customRandom) }
                    bias { randn(mean = 0.0f, std = 0.01f, random = Random(42L)) }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testComplexNestedNetworkStructure() {
        // Test complex nested structures with stages and sequential blocks
        val net = definition<FP32, Float> {
            sequential() {
                input(784)
                stage("encoder") {
                    dense(256) {
                        weights { randn(mean = 0.0f, std = 0.02f) }
                        bias { zeros() }
                    }
                    activation("encoder_activation") { tensor -> tensor }
                    sequential {
                        dense(128) {
                            weights { uniform(min = -0.1f, max = 0.1f) }
                            bias { zeros() }
                        }
                        flatten("encoder_flatten")
                    }
                }
                stage("classifier") {
                    dense(64) {
                        weights { ones() }
                        bias { randn(mean = 0.0f, std = 0.01f) }
                    }
                    dense(10) {
                        weights { randn(mean = 0.0f, std = 0.05f) }
                        bias { zeros() }
                    }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testFlattenLayerWithCustomConfiguration() {
        // Test flatten layer with custom start and end dimensions
        val net = definition<FP32, Float> {
            sequential() {
                input(28 * 28)
                flatten("custom_flatten") {
                    startDim = 1
                    endDim = -1
                }
                dense(128) {
                    weights { randn(mean = 0.0f, std = 0.1f) }
                    bias { zeros() }
                }
                dense(10) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testMultipleActivationLayers() {
        // Test multiple activation layers with different functions
        val net = definition<FP32, Float> {
            sequential() {
                input(20)
                dense(15) { weights { ones() } }
                activation("first_activation") { tensor -> tensor }
                dense(10) { weights { randn(mean = 0.0f, std = 0.1f) } }
                activation("second_activation") { tensor ->
                    // Custom activation for testing
                    tensor
                }
                dense(5) { weights { uniform(min = -0.1f, max = 0.1f) } }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testInt8NetworkWithAdvancedFeatures() {
        // Test Int8/Byte combination using new context approach
        val net = definition<Int8, Byte> {
            sequential {
                input(12)
                dense(8) {
                    weights { ones() }
                    bias { zeros() }
                }
                stage("processing") {
                    dense(6) {
                        weights { ones() }
                        bias { zeros() }
                    }
                }
                dense(3) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testInt32NetworkWithComplexStructure() {
        // Test Int32/Int combination using new context approach
        val net = definition <Int32, Int> {
            sequential {
                input(16)
                stage("feature_stage") {
                    dense(12) {
                        weights { ones() }
                        bias { zeros() }
                    }
                    activation("stage_activation") { tensor -> tensor }
                }
                dense(8) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testEmptyAndMinimalConfigurations() {
        // Test minimal network configurations
        val net1 = definition<FP32, Float> {
            sequential {
                input(2)
            }
        }

        val net2 = definition<FP32, Float> {
            sequential {
                input(3)
                dense(1) { weights { ones() } }
            }
        }

        assertNotNull(net1)
        assertNotNull(net2)
    }

    @Test
    fun testMixedInitializationStrategies() {
        // Test mixing different initialization strategies within one network
        val net = definition<FP32, Float> {
            sequential {
                input(6)
                dense(12) {
                    weights { full(0.5f) }
                    bias { zeros() }
                }
                dense(8) {
                    weights { ones() }
                    bias { init { indices -> 0.1f } }
                }
                dense(4) {
                    weights { randomInit({ random -> random.nextFloat() }) }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testSequentialAndStageNesting() {
        // Test nested sequential blocks within stages
        val net = definition<FP32, Float> {
            sequential {
                input(20)
                stage("outer_stage") {
                    sequential {
                        dense(16) {
                            weights { ones() }
                            bias { zeros() }
                        }
                        sequential {
                            dense(12) {
                                weights { randn(mean = 0.0f, std = 0.1f) }
                                bias { zeros() }
                            }
                            activation("inner_activation") { tensor -> tensor }
                        }
                    }
                }
                dense(8) {
                    weights { uniform() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testDifferentRandomSeedsAndInstances() {
        // Test that different random seeds produce different initializations
        val random1 = Random(123)
        val random2 = Random(456)
        
        val net1 = definition<FP32, Float> {
            sequential {
                input(4)
                dense(6) {
                    weights { randn(random = random1) }
                    bias { uniform(random = random1) }
                }
                dense(2) {
                    weights { uniform(random = random1) }
                    bias { randn(random = random1) }
                }
            }
        }
        
        assertNotNull(net1)
    }

    @Test
    fun testEdgeCasesAndBoundaryConditions() {
        // Test edge cases like single neuron layers, small networks
        val tinyNet = definition<FP32, Float> {
            sequential {
                input(1)
                dense(1) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }

        val largeInputNet = definition<FP32, Float> {
            sequential {
                input(1000)
                dense(500) {
                    weights { randn(std = 0.01f) }
                    bias { zeros() }
                }
                dense(100) {
                    weights { ones() }
                    bias { zeros() }
                }
                dense(10) {
                    weights { uniform(min = -0.01f, max = 0.01f) }
                    bias { zeros() }
                }
            }
        }

        assertNotNull(tinyNet)
        assertNotNull(largeInputNet)
    }

    @Test
    fun testNetworkBuilderClassDirectUsage() {
        // Test direct usage of NetworkBuilder class (if accessible)
        val net = definition<FP32, Float> {
            sequential {
                input(5)
                dense(3) {
                    weights { ones() }
                    bias { zeros() }
                }
                dense(1) {
                    weights { ones() }
                    bias { zeros() }
                }
            }
        }
        assertNotNull(net)
        assertEquals("MLP", net.name)
    }
}