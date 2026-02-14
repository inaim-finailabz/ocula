allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Force all subprojects (plugins) Kotlin JVM target to match their Java sourceCompatibility.
// Prevents "Inconsistent JVM-target compatibility" errors where Kotlin defaults to JVM 21
// while each plugin's Java compileOptions stay at 1.8.
gradle.projectsEvaluated {
    subprojects {
        // Read the Java sourceCompatibility set by each plugin's own build.gradle
        val javaVersion = extensions.findByType<com.android.build.gradle.BaseExtension>()
            ?.compileOptions?.sourceCompatibility ?: JavaVersion.VERSION_17
        val kotlinTarget = when {
            javaVersion <= JavaVersion.VERSION_1_8 -> org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_1_8
            javaVersion == JavaVersion.VERSION_11 -> org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
            else -> org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
        }
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
            compilerOptions {
                jvmTarget.set(kotlinTarget)
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
