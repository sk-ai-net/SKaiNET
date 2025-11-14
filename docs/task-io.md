# Skainet IO Core — Implementation Task List

Reference: See docs/tensors-core-arch.md for the architecture proposal this plan implements. This checklist ignores migration steps and targets the final solution.

Conventions
- [ ] indicates a task to implement; switch to [x] when complete.
- Tasks are grouped and numbered logically. Use the group numbers as labels in commits/PRs.

1. Core IO API (interfaces and minimal types)
   1.1. [x] Define TensorSource abstraction (FilePath, Path, Url/HttpRange, ByteChannel, etc.).
   1.2. [x] Define CloseableTensorArchive interface: metadata(), list(), get(name), close().
   1.3. [x] Define TensorReader high-level facade with open(source): CloseableTensorArchive.
   1.4. [x] Define TensorDescriptor (name, dtype, shape, strides/layout, byteSize, extras map, endianness, contiguity flag).
   1.5. [x] Define TensorHandle: descriptor(), stream(window?), asBufferView(), materialize(factory, opts).
   1.6. [x] Define ArchiveMetadata (formatId, version, global metadata, tensor count, totalBytes, checksums optional).
   1.7. [x] Define MaterializeOptions (preferZeroCopy, validateChecksum, device/memory hints, byteOrder override, fallbackToCopy = true).
   1.8. [x] Define ReadWindow, TensorStream, BufferView abstractions.

2. Provider SPI and Reader Selection
   2.1. [x] Define FormatReaderProvider SPI: formatId(), probe(source): ProbeResult, open(source): CloseableTensorArchive.
   2.2. [x] Define ProbeResult with confidence score, version, reason; combine/match logic.
   2.3. [x] Implement DefaultTensorReader that discovers providers, runs probe, selects highest confidence, opens archive.
   2.4. [x] Implement provider discovery for JVM/Native/JS (ServiceLoader on JVM; expect/actual or registry elsewhere).

3. SkainetTensorFactory Adapter (integration with skainet-lang)
   3.1. [x] Implement SkainetTensorFactory interface: allocate(desc), wrap(buffer, desc).
   3.2. [x] Implement SkainetTensorFactoryAdapter backed by sk.ainet.lang.tensor.data.TensorDataFactory (DenseTensorDataFactory default).
   3.3. [x] Implement dtype/shape mapping utils between IO layer and skainet-lang (DType, Shape, strides).
   3.4. [x] Implement copy vs wrap decision using MaterializeOptions and factory capabilities.

4. Buffer/Streaming/Zero-Copy Infrastructure
   4.1. [x] Implement BufferView with byteOrder, slice(offset,len), asByteBuffer (JVM), readFully into provided buffer.
   4.2. [x] Implement TensorStream for chunked reads (single-consumer), with close and checksum hooks.
   4.3. [x] Implement HTTP range-aware source support and simple retry/pacing hooks (where platform allows).
   4.4. [x] Implement file-backed mmap BufferView on JVM when contiguous/aligned; fall back to copy otherwise.
   4.5. [x] Implement endianness check/transform path when IO endianness != native (copy transform).

5. SafeTensors Provider
   5.1. [x] Implement probe for .safetensors by header validation and/or extension heuristic.
   5.2. [x] Parse header/index on open into ArchiveMetadata and TensorDescriptor list.
   5.3. [x] Implement TensorHandle: descriptor(), asBufferView() via mmap when possible, stream() via FileChannel ranges.
   5.4. [x] Implement checksum validation (if present) during streaming/materialization when validateChecksum=true. (stub hook in provider)
   5.5. [x] Cover dtypes supported by SafeTensors (float32/16, int types, bf16, etc.).

6. GGUF Provider
   6.1. [x] Implement probe for GGUF versions (v2/v3), confidence scoring.
   6.2. [x] Parse header, metadata, and tensor index; build descriptors.
   6.3. [x] Implement TensorHandle with stream/range reads and zero-copy where possible.
   6.4. [x] Support quantized types mapping to skainet DTypes or custom mapping as needed.

7. Materialization Pipeline
   7.1. [x] Implement TensorHandle.materialize(factory, opts) that selects wrap vs copy.
   7.2. [x] Implement copy path using factory.allocate + single-pass read/transform.
   7.3. [x] Implement wrap path using factory.wrap for contiguous, aligned buffers.
   7.4. [x] Implement checksum verification integration without forcing copy (stream tap or post-read verify).

8. Error Handling and Validation
   8.1. [ ] Define InvalidFormatException, UnsupportedDTypeException, IncompatibleLayoutException, IOReadException.
   8.2. [ ] Implement non-throwing probe failures (supported=false) with reason codes.
   8.3. [ ] Validate shape*stride*elementSize equals byteSize; reject otherwise.
   8.4. [ ] Validate alignment requirements for zero-copy; otherwise require copy.

9. Concurrency
   9.1. [ ] Make CloseableTensorArchive safe for concurrent get/stream/materialize across different tensors.
   9.2. [ ] Ensure TensorStream is single-consumer; document behavior.
   9.3. [ ] Ensure file channel/range access is properly synchronized or uses independent channels where needed.

10. DSL Integration Examples and Ergonomics
    10.1. [ ] Provide code snippets demonstrating reading and materializing tensors, then using skainet DSL ops.
    10.2. [ ] Provide helper to materialize by name with factory defaults (extension function for convenience).

11. Testing and Golden Files
    11.1. [ ] Prepare golden files per format with mixed dtypes/shapes (safetensors, gguf) under test resources.
    11.2. [ ] Tests for provider selection via probe (extension vs header preference when both possible).
    11.3. [ ] Tests for metadata-only access: list(), descriptor(), archive metadata.
    11.4. [ ] Tests for zero-copy wrap via DenseTensorDataFactory when conditions are met.
    11.5. [ ] Tests for copy path correctness and endianness transform.
    11.6. [ ] Tests for HTTP range streaming (mock server) and partial reads.
    11.7. [ ] Tests for checksum validation (positive and negative cases).
    11.8. [ ] Tests for concurrency: parallel materialization of different tensors.

12. Discovery/Registration and Build Integration
    12.1. [ ] JVM: Register providers via META-INF/services and ServiceLoader usage in DefaultTensorReader.
    12.2. [ ] Multiplatform: Provide expect/actual or registry approach for Native/JS.
    12.3. [ ] Ensure module boundaries: skainet-io-core (API), skainet-io-safetensors, skainet-io-gguf plug-ins.
    12.4. [ ] Wire Gradle modules and dependencies; expose public API surfaces as intended.

13. Performance and Memory
    13.1. [ ] Benchmark zero-copy vs copy paths on JVM (JMH or existing benchmarks).
    13.2. [ ] Validate minimal-copy behavior in profiles; ensure no accidental array duplication.
    13.3. [ ] Provide MaterializeOptions knobs to control allocation size & paging for streams.

14. Documentation
    14.1. [ ] Update README/docs with quickstart for TensorReader and format-specific notes.
    14.2. [ ] Add architecture cross-links and diagrams (reuse mermaid from tensors-core-arch.md where relevant).
    14.3. [ ] Document supported dtypes, shapes, alignment, and limitations.

Acceptance Criteria (per group)
- Core API and SPI compile and are used by at least one provider (SafeTensors). [ ]
- DefaultTensorReader selects the correct provider in tests covering multiple formats. [ ]
- Zero-copy works on JVM for contiguous tensors when alignment and endianness match. [ ]
- Copy path correctness validated against golden files for all supported dtypes. [ ]
- Concurrency tests pass without races or corruption. [ ]
- Public docs demonstrate end-to-end flow using skainet-lang Tensor/DSL. [ ]
