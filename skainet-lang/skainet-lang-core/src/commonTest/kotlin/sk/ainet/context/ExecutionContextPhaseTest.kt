package sk.ainet.context

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext

class ExecutionContextPhaseTest {

    @Test
    fun default_is_eval() {
        val ctx = DefaultNeuralNetworkExecutionContext()
        assertEquals(Phase.EVAL, ctx.phase)
        assertFalse(ctx.inTraining, "inTraining should be false by default")
    }

    @Test
    fun train_scope_sets_training_true() {
        val base = DefaultNeuralNetworkExecutionContext()
        var sawTraining = false
        train(base) { ctx ->
            sawTraining = ctx.inTraining
        }
        assertTrue(sawTraining, "train scope should set inTraining to true")
    }

    @Test
    fun eval_scope_sets_training_false() {
        val base = DefaultNeuralNetworkExecutionContext(phase = Phase.TRAIN)
        var sawTraining = true
        eval(base) { ctx ->
            sawTraining = ctx.inTraining
        }
        assertFalse(sawTraining, "eval scope should set inTraining to false")
    }
}
