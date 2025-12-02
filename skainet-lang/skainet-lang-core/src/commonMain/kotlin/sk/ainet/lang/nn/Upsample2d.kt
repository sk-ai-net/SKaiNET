package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType

/**
 * 2D upsampling layer for increasing spatial resolution.
 *
 * Supports nearest-neighbor mode (typical for YOLO necks). Bilinear can be wired
 * once backends add support.
 */
public class Upsample2d<T : DType, V>(
    public val scale: Pair<Int, Int> = 2 to 2,
    public val mode: UpsampleMode = UpsampleMode.Nearest,
    public val alignCorners: Boolean = false,
    override val name: String = "Upsample2d",
) : Module<T, V>() {

    init {
        require(scale.first > 0 && scale.second > 0) { "Upsample2d scale factors must be positive" }
        require(mode == UpsampleMode.Nearest || !alignCorners) {
            "alignCorners applies only to bilinear mode"
        }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            input.ops.upsample2d(
                input = input,
                scale = scale,
                mode = mode,
                alignCorners = alignCorners
            )
        }
}
