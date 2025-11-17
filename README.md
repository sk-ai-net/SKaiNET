[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)

# SKaiNET

**SKaiNET** is an open-source deep learning framework written in Kotlin Mutliplatform, designed with developers in mind to enable the creation modern AI powered applications with ease.

## Use it

- From Kotlin code in apps, libraries, CLIs
- In Kotlin Notebooks for quick exploration
- With sample projects to learn patterns

See also CHANGELOG for what’s new in 0.2.0.

## Quick start

Gradle (Kotlin DSL):

```kotlin
dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

dependencies {
    // minimal dependency with simple CPU backend
    implementation("sk.ainet.core:skainet-lang-core:0.2.0")
    implementation("sk.ainet.core:skainet-backend-cpu:0.2.0")
    
    // simple model zoo
    implementation("sk.ainet.core:skainet-lang-models:0.2.0")
    
    // Optional I/O (e.g., GGUF loader, JSON)
    implementation("sk.ainet.core:skainet-io-core:0.2.0")
    implementation("sk.ainet.core:skainet-io-gguf:0.2.0")
}
```

Maven:

```xml
<dependency>
  <groupId>sk.ainet.core</groupId>
  <artifactId>skainet-lang-core</artifactId>
  <version>0.2.0</version>
</dependency>
```

## Samples and notebooks

- Sample app: https://github.com/sk-ai-net/skainet-samples/tree/feature/MNIST/SinusApproximator
- Kotlin Notebook: https://github.com/sk-ai-net/skainet-notebook


## 0.2.0 highlights (with tiny snippets)

- Training/Eval phases made easy

```kotlin
val base = DefaultNeuralNetworkExecutionContext() // default = EVAL
val yTrain = train(base) { ctx -> model.forward(x, ctx) }
val yEval  = eval(base)  { ctx -> model.forward(x, ctx) }
```

- Dropout and BatchNorm layers

```kotlin
val y = x
    .let { dropout(p = 0.1).forward(it, ctx) }
    .let { batchNorm(numFeatures = 64).forward(it, ctx) }
```

- Conv2D + MaxPool in the NN DSL

```kotlin
val model = nn {
    conv2d(outChannels = 16, kernel = 3)
    maxPool2d(kernel = 2)
    dense(out = 10)
}
```

- Data API with MNIST loader and JSON dataset support

```kotlin
val ds = MNIST.load(train = true) // platform-aware loader
val (batchX, batchY) = ds.nextBatch(64)
```

- GGUF model loading (initial)

```kotlin
val gguf = GGUF.read("/path/to/model.gguf")
println("Tensors: ${gguf.tensors.size}")
```

- SIMD/Vector API acceleration on JVM; MatMul, tril, pooling ops; forward hooks and simple tape recording; unified tensor creation contexts; nested data blocks returning tensors.

See CHANGELOG.md for the full list.

## License

MIT — see LICENSE.
