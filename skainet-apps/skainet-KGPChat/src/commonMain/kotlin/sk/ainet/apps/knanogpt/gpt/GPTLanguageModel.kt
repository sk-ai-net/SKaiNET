package sk.ainet.apps.knanogpt.gpt

import sk.ainet.apps.knanogpt.transformer.Block
import sk.ainet.apps.knanogpt.transformer.TransformerConfig
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.types.FP16


class GPTLanguageModel(config: TransformerConfig, override val name: String) : Module<FP16, Float>() {
    private val lm_head: Linear<FP16,Float>
    private val ln_f: LayerNorm
    private val blocks: List<Block>
    private val position_embedding_table: Embedding<FP16, Float>
    private val token_embedding_table: Embedding<FP16, Float>

    init {
        // each token directly reads off the logits for the next token from a lookup table
        with(config) {
            token_embedding_table = Embedding(vocab_size, n_embd)
            position_embedding_table = Embedding(block_size, n_embd)
            blocks = List(n_layer) {
                Block(config)
            }

            ln_f = LayerNorm(n_embd)
            lm_head = Linear(n_embd, vocab_size)
        }
        initWeights()
    }

    private fun initWeights() {
        modules.forEach { module ->
            if (module is Linear) {
                val weights = module.params.by("W")
                weights.value = normalInit(weights.value.shape, 0.0, 0.02)
                val bias = module.params.by("B")
                bias.value = weights.value.shape.zeros()
            }
            if (module is Embedding) {
                val weights = module.params.by("W")
                weights.value = normalInit(weights.value.shape, 0.0, 0.02)
            }
        }
    }


    override val params: List<NamedParameter>
        get() = modules.map { module -> module.params }.flatten()
    override val modules: List<Module>
        get() = listOf(token_embedding_table, position_embedding_table, ln_f, lm_head) + blocks

    override fun forward(input: Tensor): Tensor {
        return token_embedding_table(input) // # (B,T,C)
            .let { it + position_embedding_table(it) } // # (B,T,C)
            .let { blocks.fold(it) { acc, block -> block(acc) } } // # (B,T,C)
            .let { ln_f(it) } // # (B,T,C)
            .let { lm_head(it) } // # (B,T,V)
    }
}