package sk.ainet.apps.onnx.detect

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import onnx.TensorProto
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ApplyWeightsTest {
    private class DummyModule(
        private val param: ModuleParameter<FP32, Float>
    ) : Module<FP32, Float>(), ModuleParameters<FP32, Float> {
        override val name: String = "dummy"
        override val modules: List<Module<FP32, Float>> = emptyList()
        override val params: List<ModuleParameter<FP32, Float>> = listOf(param)
        override fun forward(input: sk.ainet.lang.tensor.Tensor<FP32, Float>, ctx: sk.ainet.context.ExecutionContext) = input
    }

    @Test
    fun throws_when_not_all_params_mapped() {
        val ctx = DirectCpuExecutionContext()
        val weight = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1),
            dtype = FP32::class,
            data = floatArrayOf(1f)
        )
        val module = DummyModule(ModuleParameter.WeightParameter("w", weight))
        val mapping = applyWeights(module, emptyList())
        assertEquals(0, mapping.mapped)
        assertFailsWith<IllegalArgumentException> {
            validateAllParametersMapped(mapping, skipped = emptyList())
        }
    }

    @Test
    fun maps_when_shapes_match() {
        val ctx = DirectCpuExecutionContext()
        val paramTensor = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1),
            dtype = FP32::class,
            data = floatArrayOf(0f)
        )
        val incoming = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1),
            dtype = FP32::class,
            data = floatArrayOf(2f)
        )
        val module = DummyModule(ModuleParameter.WeightParameter("w", paramTensor))
        val init = InitTensor(
            name = "w",
            isBias = false,
            shape = listOf(1),
            tensor = incoming
        )
        val mapping = applyWeights(module, listOf(init))
        validateAllParametersMapped(mapping, skipped = emptyList())
        assertEquals(2f, module.params[0].value.data[0])
    }

    @Test
    fun decode_initializer_defaults_missing_data_to_zero() {
        val ctx = DirectCpuExecutionContext()
        val tensor = TensorProto(
            name = "zero_const",
            dataType = TensorProto.DataType.FLOAT.value,
            dims = listOf(2L)
        )
        val init = decodeInitializer(tensor, ctx).getOrThrow()
        assertEquals(listOf(2), init.shape)
        val data = init.tensor.data
        assertEquals(0f, data[0])
        assertEquals(0f, data[1])
    }

    @Test
    fun error_message_lists_missing_and_skipped_details() {
        val mapping = MappingResult(
            mapped = 1,
            total = 3,
            missingParams = listOf("conv.weight shape=[2, 2]", "conv.bias shape=[2]"),
            unusedInitializers = listOf("unused_init shape=[1]")
        )
        val skipped = listOf("bad_init: no data for dtype=1")

        val ex = assertFailsWith<IllegalArgumentException> {
            validateAllParametersMapped(mapping, skipped)
        }

        val msg = ex.message ?: error("Expected message")
        assertTrue(msg.contains("Only mapped 1/3"))
        assertTrue(msg.contains("Missing params (2"))
        assertTrue(msg.contains("conv.weight"))
        assertTrue(msg.contains("Unused initializers (1"))
        assertTrue(msg.contains("bad_init"))
    }
}
