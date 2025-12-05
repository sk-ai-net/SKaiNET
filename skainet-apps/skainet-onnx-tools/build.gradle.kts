plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    application
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    implementation(libs.kotlinx.serialization.json)
    implementation(project(":skainet-io:skainet-io-onnx"))
    implementation(libs.pbandk.runtime)

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("sk.ainet.apps.onnx.tools.WeightExportKt")
}

tasks.test {
    useJUnitPlatform()
}
