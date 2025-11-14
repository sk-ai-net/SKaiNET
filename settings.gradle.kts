pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}




rootProject.name = "SKaiNET"

includeBuild("build-logic")

// ====== LANG
include("skainet-lang:skainet-lang-core")
include("skainet-lang:skainet-lang-models")
include("skainet-lang:skainet-lang-ksp-annotations")
include("skainet-lang:skainet-lang-ksp-processor")


// ====== COMPILE
include("skainet-compile:skainet-compile-core")

// ====== BACKENDS
include("skainet-backends:skainet-backend-cpu")

// ====== BENCHMARKS
include("skainet-backends:benchmarks:jvm-cpu-jmh")

// ====== DATA
include("skainet-data:skainet-data-api")
include("skainet-data:skainet-data-simple")

// ====== IO
include("skainet-io:skainet-io-core")
include("skainet-io:skainet-io-gguf")


// ====== APPS
//include("skainet-apps:skainet-KGPChat")

