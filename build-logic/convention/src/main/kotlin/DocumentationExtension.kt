import models.DocumentationFormat
import org.gradle.api.Project
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property

open class DocumentationExtension(project: Project) {
    val inputFile: RegularFileProperty = project.objects.fileProperty()
    val outputDirectory: DirectoryProperty = project.objects.directoryProperty()
    val templateDirectory: DirectoryProperty = project.objects.directoryProperty()
    val format: Property<DocumentationFormat> = project.objects.property(DocumentationFormat::class.java)
        .convention(DocumentationFormat.ASCIIDOC)
    val includeBackendStatus: Property<Boolean> = project.objects.property(Boolean::class.java)
        .convention(true)
    val generateIndex: Property<Boolean> = project.objects.property(Boolean::class.java)
        .convention(true)
}