package sk.ainet.compile.json

import sk.ainet.compile.json.model.SkJsonExport

public actual suspend fun writeExportToFile(export: SkJsonExport, path: String, pretty: Boolean) {
    throw NotImplementedError("writeExportToFile is not implemented for macOS in this module. Use toJsonString() and handle file I/O in the host app.")
}
