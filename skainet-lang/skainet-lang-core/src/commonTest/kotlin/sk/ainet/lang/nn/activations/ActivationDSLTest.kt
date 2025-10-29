package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.tensor.sigmoid
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ActivationDSLTest {

    @Test
    fun testSiLU_DSLIntegration() {
        val model = definition<FP32, Float> {
            network {
                input(10)
                dense(5) {
                    activation = { tensor -> with(tensor) { silu() } }
                }
            }
        }
        
        assertNotNull(model)
        assertTrue(model.modules.isNotEmpty())
    }

    @Test
    fun testGELU_DSLIntegration() {
        val model = definition<FP32, Float> {
            network {
                input(10)
                dense(5) {
                    activation = { tensor -> with(tensor) { gelu() } }
                }
            }
        }
        
        assertNotNull(model)
        assertTrue(model.modules.isNotEmpty())
    }

    @Test
    fun testSigmoid_DSLIntegration() {
        val model = definition<FP32, Float> {
            network {
                input(10)
                dense(5) {
                    activation = { tensor -> with(tensor) { sigmoid() } }
                }
            }
        }
        
        assertNotNull(model)
        assertTrue(model.modules.isNotEmpty())
    }

    @Test
    fun testMultipleNewActivations_DSLIntegration() {
        val model = definition<FP32, Float> {
            network {
                input(20)
                
                // First layer with SiLU (YOLO8 style)
                dense(16) {
                    activation = { tensor -> with(tensor) { silu() } }
                }
                
                // Second layer with GELU (modern alternative)
                dense(8) {
                    activation = { tensor -> with(tensor) { gelu() } }
                }
                
                // Output layer with Sigmoid (detection outputs)
                dense(1) {
                    activation = { tensor -> with(tensor) { sigmoid() } }
                }
            }
        }
        
        assertNotNull(model)
        assertTrue(model.modules.isNotEmpty())
    }

    @Test
    fun testYOLO8StyleActivationChain() {
        // Simulate YOLO8-style network using SiLU as primary activation
        val yoloModel = definition<FP32, Float> {
            network {
                input(512)
                
                dense(256) {
                    activation = { tensor -> with(tensor) { silu() } }
                }
                
                dense(128) {
                    activation = { tensor -> with(tensor) { silu() } }
                }
                
                dense(64) {
                    activation = { tensor -> with(tensor) { silu() } }
                }
                
                // Final detection layer with sigmoid
                dense(10) {
                    activation = { tensor -> with(tensor) { sigmoid() } }
                }
            }
        }
        
        assertNotNull(yoloModel)
        assertTrue(yoloModel.modules.size >= 4) // Input + 4 dense layers
    }
}