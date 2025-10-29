package sk.ainet.lang.nn.yolo

import sk.ainet.lang.nn.Model
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

public class Yolo8() : Model {
    public override fun <T : DType, V> model(): Module<FP32, Float> = model
    override fun modelCard(): String {
        TODO("Not yet implemented")
    }


    private val model = definition<FP32, Float> {
        network {
            input(1)  // Single input for x value
        }
    }
}