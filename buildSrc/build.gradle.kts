plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.2.21"
    kotlin("plugin.serialization") version "2.2.21"
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
    implementation("org.asciidoctor:asciidoctorj:3.0.0")
    // JSON schema validation dependencies for SchemaValidationTask
    implementation("com.networknt:json-schema-validator:2.0.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.20.0")
    implementation(gradleApi())
}

kotlin {
    jvmToolchain(17)
}