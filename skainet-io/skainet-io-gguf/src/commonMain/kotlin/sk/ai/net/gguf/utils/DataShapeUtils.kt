package sk.ai.net.gguf.utils

//TODO add support for multi dimensions
fun <T> List<T>.reshape(rows: Int, cols: Int): List<List<T>> {
    if (this.size != rows * cols) throw IllegalArgumentException("Invalid dimensions for reshape")

    return List(rows) { rowIndex ->
        this.subList(rowIndex * cols, (rowIndex + 1) * cols)
    }
}
