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
include("skainet-lang:skainet-lang-core")
include("skainet-lang:skainet-lang-models")
include("skainet-lang:skainet-lang-ksp-annotations")
include("skainet-lang:skainet-lang-ksp-processor")
include("skainet-compile:skainet-compile-core")
//include("skainet-compile:skainet-compile-dag")
//include("skainet-backends:skainet-backends-core")
include("skainet-backends:skainet-backend-cpu")
