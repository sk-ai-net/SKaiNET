# Repository Guidelines

## Project Structure & Module Organization
- SKaiNET is a Kotlin Multiplatform Gradle build (see `settings.gradle.kts`) with shared plugin logic in `build-logic/`.
- `skainet-lang/` contains the DSL, operator metadata, and the KSP processor; implementations sit in `src/commonMain`, tests in `src/commonTest`, and generated operator docs in `skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json`.
- `skainet-backends/skainet-backend-cpu` implements runtime kernels, `skainet-compile/*` handles graph/HLO/ONNX transforms, `skainet-data/*` provides loaders such as MNIST, and docs plus tooling live in `docs/` and `tools/docgen/`.

## Build, Test, and Development Commands
- `./gradlew clean assemble allTests` mirrors the CI job defined in `.github/workflows/build.yml`; run it before raising a pull request.
- `./gradlew test` or `./gradlew :module:allTests` gives faster feedback for a specific module or target.
- `./gradlew generateDocs validateOperatorSchema` regenerates the reflective AsciiDoc pages and checks the emitted JSON against the schema workflow.
- `./gradlew koverHtmlReport` (or `./gradlew check`) produces coverage reports under `build/reports/kover`.
- `./gradlew apiCheck` blocks unintended ABI drifts; call `apiDump` only when you purposefully evolve a public API.

## Coding Style & Naming Conventions
- Use the Kotlin formatter with 4-space indents and keep files under 120 columns; group related declarations instead of scattering helpers.
- Modules such as `skainet-lang-core` enable `explicitApi`, so spell out visibility and document DSL entry points with concise KDoc for Dokka.
- Favor `UpperCamelCase` types, `lowerCamelCase` members, reserve `snake_case` for fixtures, and suffix backend kernels with `*Kernel`; keep builders pure and gate experiments with `@OptIn`.

## Testing Guidelines
- Locate specs next to code (`skainet-lang/*/src/commonTest`, platform specifics under `src/jvmTest`); name files `FeatureNameTest.kt` and methods `operation_expectedBehavior`.
- Rely on `kotlin-test` and `kotlinx-coroutines-test`; share tensor fixtures through helpers rather than inline literals to stay deterministic.
- Run `./gradlew allTests` before pushing so CI’s build, documentation, and schema-validation workflows reproduce your results, and keep Kover’s reports stable by covering both success and failure paths.

## Commit & Pull Request Guidelines
- Follow GitFlow naming (`feature/<issue>-short-title`, `release/*`, `hotfix/*`) as detailed in `GITFLOW.adoc`.
- Write imperative commit subjects and add trailers such as `Related-To: #123`, matching the current log.
- PRs should summarize the change, list affected modules, mention any doc or API updates, link issues, and confirm the commands above were run locally with evidence (logs, screenshots for DSL/docs).

## Security & Configuration Tips
- Keep credentials in `local.properties` or `~/.gradle/gradle.properties` and never commit data copied from `.github/ci-gradle.properties`.
- Use JDK 17 (see `.java-version`) and avoid manual edits under `docs/modules/operators/_generated_/`; rerun the Gradle doc pipeline instead.
