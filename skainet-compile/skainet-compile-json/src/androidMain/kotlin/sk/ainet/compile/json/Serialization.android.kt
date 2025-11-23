package sk.ainet.compile.json

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.compile.json.model.SkJsonExport
import java.io.File

public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    val json = export.toJsonString(pretty)
    return withContext(Dispatchers.IO) {
        val file = File(path)
        file.parentFile?.mkdirs()
        file.writeText(json)
    }
}
