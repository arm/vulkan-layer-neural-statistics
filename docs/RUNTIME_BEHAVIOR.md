<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# Runtime Behavior

This document describes Vulkan integration, failure behavior, and asynchronous
GPU collection for `VK_LAYER_LGL_neural_statistics`.

## Failure Behavior

The layer does not have one blanket fail-open or fail-closed rule. The behavior
depends on whether downstream Vulkan work has happened and whether it can still
be safely undone.

| Failure point | Current behavior |
| --- | --- |
| Configuration, root claim, or writer startup | Fail `vkCreateInstance` before calling the downstream implementation. |
| Publication of an instance, shader module, data-graph pipeline, command pool, or command buffer | On synchronous layer bookkeeping failure, destroy or otherwise roll back the downstream-created object, clear returned handles where applicable, and return an allowed Vulkan error. |
| Logical-device metadata after downstream `vkCreateDevice` succeeds | Preserve the created device and `VK_SUCCESS`; if metadata admission fails, mark capture terminal. |
| Unsupported or failed data-graph session instrumentation for which retry is safe | Destroy any partial instrumented session and retry the application's original, unmodified session creation. Track a successful fallback session as noncapturable; selecting it later follows the unsupported-occurrence rule below. |
| Capture allocation, planning, or submit rewriting before queue submission | Return a Vulkan error without forwarding the application submission. |
| A selected but unsupported occurrence, or recording metadata whose occurrence order cannot be trusted | Mark capture terminal and forward the original application submission without capture instrumentation. No capture identities are committed for that submission. |
| Marker, collection, or asynchronous writer failure after application work has been accepted | Preserve the already accepted Vulkan result, mark capture terminal, log an error, and request terminal `error` metadata from the writer. Accepted work cannot be reported as unsubmitted or safely retried. |
| Calls made after capture is terminal | Forward without further capture instrumentation or publication. |

Terminal error publication is best-effort. If the filesystem or writer is the
failed component, `capture.json` may remain at the last complete `incomplete`
state rather than being updated to `error`. Capture consumers must require a
final `complete` state for success and treat `error`, `incomplete`, or missing
expected artifacts as a failed capture. Instance-driven terminal transitions also
emit their first diagnostic through `LAYER_ERR`; a writer-thread failure may be
observable only through capture state until a later intercepted call observes it.

## Vulkan Integration

One `CaptureWriter` is claimed and started before the downstream
`vkCreateInstance` call. Existing roots, unsupported filesystems, invalid
settings, and startup failures therefore fail before downstream instance state
exists. The instance drains and publishes `complete` or `error` during
`vkDestroyInstance`. Multiple logical devices may be created during one instance
capture; each receives a distinct capture-local ID recorded in capture,
pipeline, and session metadata.

If a logical device does not expose the neural-statistics extension and feature,
the device remains usable without layer instrumentation and `capture.json`
records an explicit per-device warning instead of silently presenting an empty
capture as fully supported.

Shader modules receive independent monotonic IDs and deep-copy SPIR-V at
successful creation. Pipelines and sessions receive their own independent
monotonic IDs synchronously under the Vulkan tracking lock. Writer jobs carry
those IDs and immutable metadata/bytes; object creation never waits for
filesystem completion, although the bounded writer queue can apply documented
backpressure. Destroy/recreate handle reuse cannot reuse IDs, and queued jobs
retain owned data after Vulkan mappings are erased.

Executed indices are planned per session at accepted queue submission. Every
occurrence advances its session counter, including unselected occurrences and
command-buffer re-submissions. Secondary execution is expanded in recorded
execution order. Capture-only legality checks are applied only after the filter
selects an occurrence, so unsupported unselected work passes through without
creating hierarchy directories. Legacy, core submit2, and KHR submit2 preserve
their matching downstream ABI route.

Recording failures are split by countability. If the layer still knows that an
application dispatch occurred and where it occurred, the command-buffer record
owns one occurrence with a noncapturable reason; an unselected occurrence remains
transparent. If the filter selects an unsupported occurrence, the layer
terminally disables capture and forwards that original application submit
unchanged, without committing execution indices or dispatch IDs. Failures that
make occurrence order or count unknowable also terminally disable capture. The
original recording command and current and later submissions are then forwarded
without capture rewriting or bookkeeping, so an internal metadata failure cannot
invalidate otherwise legal application work.

During queue submission while capture is operational, pre-forward
instrumentation allocation and materialization failures remain transactional and
may return a Vulkan error without forwarding application work. Exact forwarding
is the fallback after capture has become terminal or the selected topology is
unsupported.

Selected occurrences materialize the direct pipeline/session/dispatch hierarchy
at accepted-submit admission. Their raw `statistics_mode0.bin` or
`statistics_mode1.bin` artifact is enqueued later by the asynchronous collector,
after the marker fence completes and mapped bytes have been copied into an owned
host vector. Static artifacts remain pending until the first selected occurrence
materializes their pipeline. Friendly-name updates rewrite metadata without
renaming ID directories. `capture.json` includes layer version/commit, selected
settings, device/driver/Vulkan properties, capture state, and warnings;
production JSON contains no hashes.

On Vulkan 1.0 driver profiles, the layer enables
`VK_KHR_get_memory_requirements2` only when it appears in the physical device's
enumerated extension list, then uses the exact KHR command to obtain
`VkMemoryDedicatedRequirements`. Core Vulkan 1.1+ uses the core command. Drivers
exposing neither route remain conservatively noncapturable; the layer never
infers dedicated-allocation safety from legacy `vkGetBufferMemoryRequirements`
alone.

Fault-injection builds expose compile-time-only filesystem and collector
observers for live-hook tests. Production builds compile those observer paths
out.

## Asynchronous GPU Collector

Each logical device owns one central collector with a bounded number of jobs and
one collector thread. This bounds admitted job count, not the total number or
bytes of GPU snapshot allocations owned by those jobs. Submission admission
reserves collector capacity before forwarding selected work, so queue saturation
applies explicit lossless backpressure without risking a successfully submitted
marker fence that has no owner. After the application submission succeeds, the
layer commits capture IDs and indices, submits an empty marker through the same
legacy/core/KHR ABI route as the intercepted command, transfers an immutable job
to the reserved collector slot, and returns the exact application result.

Collector jobs own the accepted submission plan, selected
`shared_ptr<StatsSnapshot>` objects, session/snapshot reservations, marker
fence, statistics mode, device handle, and copied device-dispatch functions.
They retain no application submit arrays, command-buffer pointers, or later
tracking-map lookups. A command buffer may be reset or freed while work is
pending; its ownership is dropped, while collector ownership keeps the snapshot
allocation and session lifetime valid until completion. Reservations remain set
from accepted forwarding until the collector calls `submission::finish()` exactly
once.

The collector polls every pending fence with `vkGetFenceStatus`; `VK_NOT_READY`
jobs are skipped and the worker sleeps on a condition variable with a short
internal polling interval. Therefore one stalled queue cannot prevent a signaled
job on another same-family queue from completing. Completion maps the
host-visible allocation, invalidates noncoherent memory, copies bytes
immediately into an owned vector, and submits the raw artifact to
`CaptureWriter`. Fence destruction, reservation release, and collector snapshot
release happen only after completion or a terminal fence/device error.

Collector admission capacity and polling interval are internal/test construction
options, not public settings. Closing a collector rejects future capture
reservations and wakes blocked producers. A submit that observes closed
admission terminally disables capture and is forwarded unchanged without marker
or archive work. Device destruction first closes admission, asks the collector
thread to drain the device, joins it, releases command-buffer snapshot owners,
and only then calls downstream `vkDestroyDevice`; the collector can no longer
call the dispatch table after downstream destruction. Instance destruction waits
for accepted collector jobs before requesting writer completion, then drains and
joins the writer before downstream `vkDestroyInstance`.

Marker creation/submission failure after a successful application submission is
a capture-only post-forward failure: the application result is preserved, the
first terminal capture transition wins, and potentially pending archive resources
are retained until device teardown. Collector admission, polling, device-loss,
mapped-memory, allocation, or writer-enqueue failures likewise never rewrite an
accepted application result. Later pending jobs submit no new writer work after
terminal capture failure, but still poll/retire their fences, call `finish()`,
and release every reservation.

## Cross-Queue Session Synchronization

Each captured dispatch clears shared session statistics storage before the
data-graph command and copies it to a per-recorded-dispatch snapshot afterward.
Bridge barriers connect those transfer operations to the data-graph stage. On
one queue they preserve submission-order dependencies; across queues they extend
the application's existing semaphore dependency, for example a signal and wait
scoped to `VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM`, so the prior snapshot copy
completes before the next clear.

Barriers do not create cross-queue synchronization themselves. Applications
remain responsible for synchronizing reuse of the same session across queues, and
unsynchronized reuse is not made valid by the layer.
