package sk.ai.net.gguf

/**
 * This is a kotlin gguf reader related logic interpreted from python code "gguf-py/gguf/quants.py"
 * of github repo "https://github.com/ggerganov/llama.cpp"
 */

//TODO convert the rest of file from quants.py

fun quantShapeToByteShape(shape: List<ULong>, quantType: GGMLQuantizationType): List<ULong> {
    val (blockSize, typeSize) = GGML_QUANT_SIZES[quantType]!!
    if (shape.last().toInt() % blockSize != 0) {
        throw IllegalArgumentException(
            "Quantized tensor row size (${shape.last()}) is not a multiple of ${quantType.name} block size ($blockSize)"
        )
    }

    val newShape = shape.dropLast(1) + (shape.last() / blockSize.toULong() * typeSize.toULong())
    return newShape
}
