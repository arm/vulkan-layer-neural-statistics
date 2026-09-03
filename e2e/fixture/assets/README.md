<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# Embedded data-graph assets

These files are build-time inputs for `neural_statistics_fixture` and are embedded into the executable by CMake.

- `graph_partition_0.spv`: 192,316 bytes, SHA-256 `6f5dd6a66cfcf571386b1722ce9d7347ef28676743415b52067f6c1f53719b6e`, entry point `graph_partition_0`.
- `graph_constants.bin`: 135,456 bytes, SHA-256 `ef9bf7b2e3e82ff556840716ccfd7e04a2c5f4ba2fe1a792c88a96500df670a0`. It concatenates constant IDs 0 through 44 in ascending ID order. Shapes and formats are declared in `graph_assets.hpp`; all constants are linear, data-graph tensors and none is sparse.

## Provenance

The development source model was `/root/fixture-assets/model.vgf` (333,328 bytes, SHA-256 `c8d76c906353b3ee19aaceb6afd15f427e83a61c93baf29c996ae07256b5e8eb`). The module was independently compared byte-for-byte with the known-good captured module from `pipeline_000000/shader_module_0.spv` before check-in. Constants were decoded once with the VGF decoder tooling built from `/root/ai-ml-sdk-manifest/sw/vgf-lib` at commit `dee4f4f5d277d88144444fa4f210b628864165af`, validated against model resource metadata, and concatenated without transformation.

The source model, decoder, FlatBuffers and extraction tooling are development-only provenance. They are not linked, loaded, parsed, deployed, or otherwise required by the fixture at build or runtime.
