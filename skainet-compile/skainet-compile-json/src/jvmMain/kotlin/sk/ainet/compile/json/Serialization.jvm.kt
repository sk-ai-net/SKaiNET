package sk.ainet.compile.json

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.compile.json.model.SkJsonExport
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    val json = export.toJsonString(pretty)
    return withContext(Dispatchers.IO) {
        val p: Path = Paths.get(path)
        Files.createDirectories(p.parent ?: p.toAbsolutePath().parent)
        Files.writeString(p, json)
    }
}
