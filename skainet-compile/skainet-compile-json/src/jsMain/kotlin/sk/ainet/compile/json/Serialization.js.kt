package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport

public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    throw NotImplementedError("writeExportToFile is not supported on JS/WASM environments yet. Use toJsonString() instead.")
}
