package sk.ainet.io.onnx

import kotlinx.io.Source
import kotlinx.io.readByteArray
import onnx.ModelProto
import pbandk.Message
import pbandk.decodeFromByteArray

/**
 * Minimal pbandk-backed ONNX loader.
 *
 * The parser is injected so we can plug the generated ONNX ModelProto decoder
 * once it is available. For now, the API keeps the raw bytes to enable
 * debugging or alternative decoding strategies.
 */
public class OnnxLoader<M : Message>(
    private val readBytes: suspend () -> ByteArray,
    private val decode: (ByteArray) -> M
) {
    public suspend fun load(): OnnxLoadedModel<M> {
        val bytes = readBytes()
        val proto = decode(bytes)
        return OnnxLoadedModel(proto, bytes)
    }

    public companion object {
        /**
         * Convenience factory for sources provided by kotlinx-io.
         */
        public fun <M : Message> fromSource(
            sourceProvider: suspend () -> Source,
            decode: (ByteArray) -> M
        ): OnnxLoader<M> = OnnxLoader(
            readBytes = {
                sourceProvider().use { source -> source.readByteArray() }
            },
            decode = decode
        )

        /**
         * Convenience factory for ONNX ModelProto using the generated pbandk decoder.
         */
        public fun fromModelSource(
            sourceProvider: suspend () -> Source
        ): OnnxLoader<ModelProto> = fromSource(sourceProvider) { bytes ->
            ModelProto.decodeFromByteArray(bytes)
        }
    }
}

/**
 * Holder for the parsed ONNX model alongside its serialized form.
 */
public data class OnnxLoadedModel<M : Message>(
    val proto: M,
    val rawBytes: ByteArray
)
