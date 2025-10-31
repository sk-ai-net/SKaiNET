import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.tasks.*
import sk.ainet.buildlogic.json.SimpleJsonValidator

@CacheableTask
abstract class SchemaValidationTask : DefaultTask() {
    @get:InputDirectory
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val searchDirectory: DirectoryProperty

    init {
        searchDirectory.convention(project.rootProject.layout.projectDirectory)
    }

    @TaskAction
    fun validate() {
        val buildDir =
            (if (searchDirectory.isPresent) searchDirectory.get() else project.rootProject.layout.projectDirectory).asFile
        val schemaStream = this::class.java.classLoader.getResourceAsStream("schemas/operator-doc-schema-v1.json")
            ?: throw IllegalStateException("Cannot find schema resource: schemas/operator-doc-schema-v1.json")
        val validator = SimpleJsonValidator()

        if (!buildDir.exists()) {
            throw RuntimeException("Build directory does not exist: ${buildDir.absolutePath}")
        }

        val operatorJsonFiles = buildDir.walkTopDown()
            .filter { it.isFile && it.name == "operators.json" }
            .toList()

        if (operatorJsonFiles.isEmpty()) {
            logger.lifecycle("No operators.json files found under: ${buildDir.absolutePath}. Skipping schema validation.")
            return
        }

        var total = 0
        var valid = 0
        val errors = mutableListOf<String>()

        operatorJsonFiles.forEach { file ->
            total++
            val result = validator.validate(schemaStream, file)
            if (result.valid) {
                valid++
                logger.lifecycle("✓ VALID: ${file.relativeTo(buildDir)}")
            } else {
                logger.error("✗ INVALID: ${file.relativeTo(buildDir)}")
                result.errors.forEach { err ->
                    val msg = "  - $file: $err"
                    errors.add("${file.relativeTo(buildDir)}: $err")
                    logger.error(msg)
                }
            }
        }

        logger.lifecycle("============================================")
        logger.lifecycle("Schema Validation Summary")
        logger.lifecycle("Total files: $total  Valid: $valid  Invalid: ${total - valid}")

        if (errors.isNotEmpty()) {
            throw RuntimeException("Schema validation failed for ${errors.size} issue(s). See log for details.")
        }
    }
}
