package sk.ainet.lang.model.dnn.yolo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.types.FP32

public class Yolo8() : Model<FP32, Float> {

    private val model = definition<FP32, Float> {
        network {
            input(1)  // Single input for x value
        }
    }

    public override fun model(executionContext: ExecutionContext): Module<FP32, Float> = model

    public override fun modelCard(): ModelCard {
        TODO()
    }
}