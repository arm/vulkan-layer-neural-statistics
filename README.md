<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# VK_LAYER_LGL_neural_statistics

`VK_LAYER_LGL_neural_statistics` is a Vulkan layer for capturing neural
accelerator statistics from applications that use
`VK_ARM_data_graph_neural_accelerator_statistics`.

The layer tracks data-graph pipelines and sessions, captures selected dispatches,
and writes a structured capture directory containing metadata, SPIR-V, debug
database data, neural-statistics information, and raw statistics payloads.

## Build

Clone this repository as `layer_neural_statistics` inside the libGPULayers
framework, update the generated Vulkan sources as described by libGPULayers, and
run one of the helper scripts:

```sh
./scripts/build_native.sh          # native Linux Release
./scripts/build_native.sh Debug    # native Linux Debug
./scripts/build_arm64.sh           # Linux ARM64 cross-build
./scripts/build_android.sh         # Android arm64-v8a
```

The layer library is written under the selected build directory as
`source/libVkLayerNeuralStatistics.so`.

With `BUILD_TESTING=ON`, CMake also builds the unit and live-hook tests. The
standalone real-ICD E2E package is opt-in:

```sh
cmake -S . -B build-e2e \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURAL_STATISTICS_BUILD_E2E=ON \
  -DBUILD_TESTING=ON
cmake --build build-e2e
ctest --test-dir build-e2e --output-on-failure
```

See [docs/E2E.md](docs/E2E.md) for the E2E fixture, matrix runner, validator,
and evidence workflow.

## Enable On Linux

Point the Vulkan loader at the layer manifest and make the layer library
discoverable:

```sh
export LD_LIBRARY_PATH="<layer-build-dir>/source:$LD_LIBRARY_PATH"
export VK_LAYER_PATH="<directory-containing-manifest-json>"
export VK_INSTANCE_LAYERS=VK_LAYER_LGL_neural_statistics
```

The manifest's `library_path` must resolve to `libVkLayerNeuralStatistics.so`.

## Enable On Android

For Android native runs, install the layer into Android's Vulkan debug-layer
directory and enable it through system properties:

```sh
adb shell mkdir -p /data/local/debug/vulkan
adb push libVkLayerNeuralStatistics.so \
  /data/local/debug/vulkan/libVkLayer_LGL_neural_statistics.so
adb shell setprop debug.vulkan.layers VK_LAYER_LGL_neural_statistics
adb shell setprop debug.vulkan.lgl_neural_statistics.capture_folder \
  /data/local/tmp/neural_capture
adb shell setprop debug.vulkan.lgl_neural_statistics.statistics_mode 0
```

Clear the global debug-layer setting after testing:

```sh
adb shell "setprop debug.vulkan.layers ''"
adb shell "setprop debug.vulkan.lgl_neural_statistics.capture_folder ''"
adb shell "setprop debug.vulkan.lgl_neural_statistics.statistics_mode ''"
adb shell rm /data/local/debug/vulkan/libVkLayer_LGL_neural_statistics.so
```

## Settings

The layer has three user-facing settings. They can be supplied through VkConfig,
`vk_layer_settings.txt`, environment variables, `VkLayerSettingsCreateInfoEXT`,
or Android's layer-setting mechanism.

| Setting | Environment variable | Meaning |
| --- | --- | --- |
| `lgl_neural_statistics.capture_folder` | `VK_LAYER_CAPTURE_FOLDER` | Exact capture output directory. If absent, the layer creates a timestamped directory. |
| `lgl_neural_statistics.statistics_mode` | `VK_LAYER_STATISTICS_MODE` | Statistics mode, either `0` or `1`. |
| `lgl_neural_statistics.dispatch_filter` | `VK_LAYER_DISPATCH_FILTER` | Dispatch occurrence filter. Blank captures all dispatches, `N` captures occurrence `N`, and `N-M` captures an inclusive range. |

Example:

```sh
export VK_LAYER_CAPTURE_FOLDER=/tmp/neural-capture
export VK_LAYER_STATISTICS_MODE=0
export VK_LAYER_DISPATCH_FILTER=10-19
```

If `capture_folder` is explicitly set, the directory must not already exist.
Invalid settings fail instance creation before the downstream Vulkan instance is
created.

## Output

A successful capture writes a directory shaped like this:

```text
<capture-root>/capture.json
<capture-root>/pipeline_000000/pipeline.json
<capture-root>/pipeline_000000/shader_module_0.spv
<capture-root>/pipeline_000000/debug_database.bin
<capture-root>/pipeline_000000/neural_statistics_info.bin
<capture-root>/pipeline_000000/session_000000/session.json
<capture-root>/pipeline_000000/session_000000/dispatch_000000/dispatch.json
<capture-root>/pipeline_000000/session_000000/dispatch_000000/statistics_mode0.bin
```

`capture.json` must finish with `"status": "complete"` for a successful capture.
Treat missing output, `"incomplete"`, or `"error"` as a failed capture.

## More Detail

- [docs/CAPTURE_FORMAT.md](docs/CAPTURE_FORMAT.md) describes the capture schema,
  file naming rules, filesystem publication model, and writer behavior.
- [docs/RUNTIME_BEHAVIOR.md](docs/RUNTIME_BEHAVIOR.md) describes failure
  behavior, Vulkan interception, logical-device handling, and the asynchronous
  GPU collector.
- [docs/SYNCHRONIZATION.md](docs/SYNCHRONIZATION.md) describes command-buffer
  synchronization and snapshot ordering.
