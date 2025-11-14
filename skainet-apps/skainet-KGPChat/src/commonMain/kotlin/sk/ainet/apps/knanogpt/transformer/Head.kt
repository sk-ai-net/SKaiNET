package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.*
import sk.ainet.lang.tensor.dsl.*
import sk.ainet.lang.types.FP16
import kotlin.math.pow


class Head(
    private val config: TransformerConfig,
    private val dropout: Double,
    private val headNumber: Int
) : Module<FP16, Float>() {

    private val _modules = mutableListOf<Module<FP16, Float>>()

    init {
        // Defer parameter tensor creation until we have an ExecutionContext at forward-time
        // We only capture config values here
    }

    private fun ensureInitialized(ctx: ExecutionContext, config: TransformerConfig, dropout: Double) {
        if (_modules.isNotEmpty()) return
        with(config) {
            data(ctx) {
                val initWeights = tensor<FP16, Float> {
                    shape(head_size, n_embd) { zeros() }
                }
                val initBias = tensor<FP16, Float> {
                    shape(head_size) { zeros() }
                }
                _modules += listOf(
                    Linear(
                        n_embd, head_size, "key",
                        initWeights = initWeights,
                        initBias = initBias
                    ),
                    Linear(
                        n_embd, head_size, "query",
                        initWeights = initWeights,
                        initBias = initBias
                    ),
                    Linear(
                        n_embd, head_size, "value",
                        initWeights = initWeights,
                        initBias = initBias
                    ),
                    Dropout(dropout.toFloat())
                )
            }
        }
    }

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            ensureInitialized(ctx, config, dropout)
            val (B, T, C) = input.shape.dimensions
            val k = modules[0].forward(input, ctx) // key
            val q = modules[1].forward(input, ctx) // query
            val v = modules[2].forward(input, ctx) // value

            val scale = k.shape.dimensions.last().toDouble().pow(-0.5)
            val out = data<FP16, Float>(ctx) {
                val wei = q.matmul(k.t()) * tensor<FP16, Float> { shape(1) { full(scale.toFloat()) } }

                // Build causal mask using tril: -Inf above diagonal, 0 on/under diagonal (broadcasted over batch)
                val onesTT = tensor<FP16, Float> { shape(T, T) { ones() } }
                val lower = onesTT.tril() // 1.0 on/under diagonal, 0.0 above
                val negInfMask = (onesTT - lower) // 1 above diagonal, 0 on/under
                    .let { it * tensor<FP16, Float> { shape(1) { full(Float.NEGATIVE_INFINITY) } } }
                val weiMasked = wei + negInfMask
                val weiSoftmax = weiMasked.softmax(-1)
                // Apply attention dropout after softmax
                val dropped = (modules[3] as Dropout<FP16, Float>).forward(weiSoftmax, ctx)
                dropped.matmul(v)
            }
            out
        }

    override val name: String
        get() = "Head-$headNumber"
    override val modules: List<Module<FP16, Float>>
        get() = _modules
}