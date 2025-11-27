package sk.ainet.compile.json

import kotlinx.serialization.json.Json
import sk.ainet.compile.json.model.SkJsonExport

// Shared Json configuration for exports
internal fun exportJson(pretty: Boolean): Json = Json {
    prettyPrint = pretty
    // Match golden fixtures which use 2-space indentation
    if (pretty) prettyPrintIndent = "  "
    encodeDefaults = false
}

/**
 * Serialize this export to a JSON string.
 */
public fun SkJsonExport.toJsonString(pretty: Boolean = true): String {
    val s = exportJson(pretty).encodeToString(this)
    // Golden fixtures are checked in as text files that end with a newline.
    // Ensure our pretty output matches that convention to make string
    // comparison stable across platforms/editors.
    return if (pretty && !s.endsWith("\n")) s + "\n" else s
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
