package sk.ainet.compile.json

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sk.ainet.compile.json.model.SkJsonExport

// Shared Json configuration for exports
internal fun exportJson(pretty: Boolean): Json = Json {
    prettyPrint = pretty
    encodeDefaults = false
}

/**
 * Serialize this export to a JSON string.
 */
public fun SkJsonExport.toJsonString(pretty: Boolean = true): String {
    return exportJson(pretty).encodeToString(this)
}

/**
 * Write JSON export to a file at the given path.
 * Expect/actual is used for multiplatform file I/O.
 */
public expect suspend fun writeExportToFile(
    export: SkJsonExport,
    path: String,
    pretty: Boolean = true
)
