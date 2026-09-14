<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# Standalone neural-statistics target E2E package

## Architecture and prerequisites

The E2E package starts on an already configured Linux target. It contains a self-contained Vulkan application, `neural_statistics_fixture`, plus a Python matrix orchestrator and structural capture validator. The fixture dynamically loads the target Vulkan loader, creates the instance and device, selects a `VK_QUEUE_DATA_GRAPH_BIT_ARM` queue family, enables `VK_ARM_data_graph`, `VK_ARM_tensors`, synchronization2 and the integer/storage features required by the graph, and owns all tensor, descriptor, pipeline, session, memory, command-buffer, semaphore, fence, and submission objects.

The graph module and its 45 constant payloads are checked in under `e2e/fixture/assets/` and embedded into the executable at build time. The fixture deterministically zero-initializes the input tensor: it maps and flushes compatible host-visible tensor memory directly, or uses a host-visible staging tensor plus a synchronized tensor copy when the input allocation is device-local. There is no runtime model decoder, shader compiler, SPIR-V tool, FlatBuffers, JSON, external application, scenario file, model file, input-file, NPY, or VGF dependency. The target still needs the Vulkan loader, real ICD, built neural-statistics layer and manifest, and Python 3.9 or newer for the orchestrator. Deployment, credentials, board management, driver installation, reboot/power/image operations, and libNXStatistics remain external to this repository.

## Build

Enable the opt-in fixture and tests in a native build:

```sh
cmake -S . -B build-e2e \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURAL_STATISTICS_BUILD_E2E=ON \
  -DBUILD_TESTING=ON
cmake --build build-e2e
ctest --test-dir build-e2e --output-on-failure
```

The fixture includes the repository-pinned Vulkan C headers from `source_third_party/khronos/vulkan/include`, uses `VK_NO_PROTOTYPES`, and resolves loader entry points at runtime. This allows the ordinary repository ARM64 toolchain to cross-link it without importing an ARM64 Vulkan loader library.

For capacity turnover, use a separate test-only layer build with collector capacity two:

```sh
cmake -S . -B build-capacity-2 -DCMAKE_BUILD_TYPE=Release \
  -DNEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY=2
cmake --build build-capacity-2
```

Never ship that capacity-override artifact.

## Fixture controls

`neural_statistics_fixture --help` lists only application-owned topology controls:

- `--graph-count 1|2|3` creates independent pipelines and sessions from the embedded graph;
- `--secondary-executions N` records one simultaneous-use secondary and executes it N times in a primary;
- `--primary-command-buffers-per-submit N` places the same primary handle N times in one submit;
- `--resubmissions N` sequentially resubmits the completed primary with bounded collector-completion retry;
- `--dispatch-repeats N --immediate-resubmission` performs immediate completed-primary reuse without retry;
- `--distinct-primary-submissions 2` records and submits two distinct one-time primaries using the same session;
- `--distinct-primary-stage-semaphore --queue-count 2` links those submissions only at `VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM`;
- `--second-dispatch-conditional-false` wraps the second distinct primary dispatch in a false conditional-rendering predicate, providing contrasting statistics for comparison with the first dispatch;
- `--submit-route legacy|core|khr` selects `vkQueueSubmit`, `vkQueueSubmit2`, or `vkQueueSubmit2KHR`;
- `--noncapturable` chains the pinned official `VkDataGraphPipelineNeuralStatisticsCreateInfoARM` with neural statistics allowed;
- `--destroy-after-fence` destroys application graph objects after the application fence while collector output remains pending;
- `--capacity-submissions 3 --graph-count 3` creates independent capacity-turnover submissions;
- `--spirv-reference PATH` writes the exact embedded SPIR-V used for pipeline creation;
- `--events PATH` writes process timing and exit metadata in addition to stdout `E2E_EVENT` records.

The fixture never sets layer settings. The orchestrator owns `VK_LAYER_CAPTURE_FOLDER`, `VK_LAYER_STATISTICS_MODE`, `VK_LAYER_DISPATCH_FILTER`, and layer activation variables. The same-session synchronization regression also owns the private test-only `VK_LAYER_TEST_POST_GRAPH_COPY_COUNT` environment hook. Values above one repeat the statistics-to-snapshot copy with transfer barriers before host visibility, widening the overwrite window without changing production behavior; the default is exactly one copy.

The fixture requires Vulkan 1.3 and synchronization2. Submit2 routes are entrypoint-exact: the core route requires `vkQueueSubmit2`; the KHR route additionally requires the enabled `VK_KHR_synchronization2` extension and `vkQueueSubmit2KHR`. The fixture does not alias one spelling to the other. Missing prerequisites or a missing required entrypoint fail the run rather than count as a successful capture test.

The core-submit case requires a successful submission, fence completion, and the exact captured dispatch with nonzero statistics. Both cross-queue same-session cases require two successful submissions through their selected entrypoint, ordered semaphore signal/wait events, and distinct statistics for the normal and conditionally suppressed dispatches.

`noncapturable-selected` exercises valid application-provided statistics configuration that the layer cannot instrument. It requires application submission and fence completion to succeed, while capture terminates with the specific configuration-conflict error and no pipeline/session/dispatch hierarchy. This is a capture error, not a warning or a forced application failure. The diagnostic is checked on stdout, where the Linux framework emits `LAYER_ERR`; matrix cases can require messages on either stream using `expected_stdout_contains` or `expected_stderr_contains`.

## Matrix and launch

The declarative matrix is `e2e/matrix/target.json`. Every case uses fixture-native arguments; no external path variables are required.

```sh
python3 e2e/neural_statistics_e2e.py \
  --matrix e2e/matrix/target.json \
  --fixture build-e2e/e2e/fixture/neural_statistics_fixture \
  --work-root /absolute/evidence/runs \
  --tag smoke \
  --summary-json /absolute/evidence/summary.json \
  --report /absolute/evidence/report.txt
```

Use repeated `--case NAME` for named subsets or `--list` to inspect selection. Each run receives a unique capture root. The orchestrator preserves stdout, stderr, command, exit status, nanosecond process timestamps, parsed `E2E_EVENT` records, validator output, and per-case JSON. A finite timeout kills the complete process group on Linux.

Point the runtime to the intended layer and real ICD in the invoking environment. Prove actual loader and driver selection from diagnostics that identify loaded absolute libraries rather than inferring it from environment variables alone.

## Validation contract

The validator reads actual output under the exact resolved capture root. It rejects unsafe paths, symlinks, non-regular files, duplicate IDs/references/artifacts, unknown artifact types, stale flat layouts, and unreferenced public files. It validates the full capture, pipeline, session and dispatch metadata contracts, exact hierarchy, mode-specific raw artifact names, optional all-zero/nonzero raw-statistics content expectations, complete lazy absence for filter misses, and expected error terminal state.

When a case supplies `--spirv-reference`, the validator compares every captured shader module byte-for-byte with the fixture reference and records hashes only in the evidence report. Graph constants are deliberately not published into capture JSON.

Matrix expectations should change only for an intentional application-topology or capture-contract change. Do not weaken exact hierarchy assertions to make a failing target run pass.

## Evidence and external decoding

Recommended evidence layout:

```text
evidence/
  commands.txt
  versions.json
  loader-proof.log
  health-before.txt
  health-after.txt
  runs/<unique-case>/...
  summary.json
  report.txt
  SHA256SUMS
```

Generate and verify the size-qualified integrity manifest with `e2e/evidence_manifest.py`. Optional mode-0/mode-1 decoding with libNXStatistics remains an external evidence step; keep decoder binaries, source and linkage outside the core repository.

## Cross-queue same-session regression

The two-queue same-session case deliberately supplies an application semaphore whose signal and wait are both scoped to VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM. The layer's pre-clear and post-copy bridge barriers extend that application-owned dependency around its injected transfer operations. Pipeline barriers alone do not synchronize queues; the test models the semaphore an application must already use to make shared-session reuse across queues valid.
