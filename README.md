[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)

# SKaiNET

**SKaiNET** is an open-source deep learning framework written in Kotlin, designed with developers in mind to enable the creation modern AI powered applications with ease.

## Quick example: training/eval phases with context wrappers

SKaiNET provides small helper scopes to override the execution phase for a given call without mutating your base context:

- train(ctx) { ... } — runs the block with phase = TRAIN
- eval(ctx) { ... } — runs the block with phase = EVAL

This is useful for modules that behave differently in training vs evaluation (e.g., Dropout, BatchNorm).

Kotlin

import sk.ainet.context.train
import sk.ainet.context.eval
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
// import your model class that defines forward(x, ctx)

val base = DefaultNeuralNetworkExecutionContext() // default phase is EVAL
val x = /* prepare your input tensor */
val model = /* construct your model */

// Force TRAIN phase for this call only
val yTrain = train(base) { ctx ->
    model.forward(x, ctx)
}

// Force EVAL phase for this call only
val yEval = eval(base) { ctx ->
    model.forward(x, ctx)
}

// You can still call with the base context directly (uses its phase)
val yBase = model.forward(x, base)

## Development Practices

This project follows established development practices for maintaining code quality and release management:

* **Branching Model**: We use [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) as our branching strategy for managing feature development, releases, and hotfixes.
* **Versioning**: We follow [Semantic Versioning (SemVer)](https://semver.org/) for all releases, ensuring predictable version numbering based on the nature of changes.

## Reflective Documentation (short overview)

SKaiNET includes a reflective documentation system that keeps docs in sync with the code. During the build, a KSP processor extracts operator metadata (signatures, parameters, backend availability, implementation status) into a JSON file. A small DocGen tool then converts this JSON into AsciiDoc fragments and pages.

- Source of truth (generated): skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json
- Generated docs output: docs/modules/operators/_generated_/
- Asciidoctor site output: build/docs/asciidoc/ (if you run an Asciidoctor task locally)

### Quick start: generate reflective docs

Use any of the following Gradle tasks from the project root:

1) Full pipeline (recommended)
   ./gradlew generateDocs
   - Runs KSP to produce operators.json (if needed)
   - Generates AsciiDoc files under docs/modules/operators/_generated_
   - Optionally, you can run an Asciidoctor task to build an HTML site locally (output under build/docs/asciidoc)

2) Operators documentation only
   ./gradlew generateOperatorDocs
   - Depends on KSP; runs the built-in generateDocs task and then Asciidoctor

Open the generated AsciiDoc sources in docs/modules/operators/_generated_ with your preferred AsciiDoc viewer. If you build an HTML site locally with Asciidoctor, open build/docs/asciidoc.

### Documentation tooling

We now use the Gradle plugin (buildSrc) only. The former skainet-lang-export-ops module has been removed. All everyday workflows are covered by:
- generateDocs — converts KSP JSON to AsciiDoc
- validateOperatorSchema — validates generated operators.json against the JSON schema

Run from the project root, for example:
- ./gradlew generateDocs
- ./gradlew validateOperatorSchema

---

## Development Practices

This project follows established development practices for maintaining code quality and release management:

* Branching Model: We use GitFlow as our branching strategy for managing feature development, releases, and hotfixes.
* Versioning: We follow Semantic Versioning (SemVer) for all releases, ensuring predictable version numbering based on the nature of changes.
