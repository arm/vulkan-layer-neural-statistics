<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# Capture Format And Publication

This document describes the structured output written by
`VK_LAYER_LGL_neural_statistics`.

## Structured Capture Model

The current capture, pipeline, session, and dispatch schema version is `2`,
adding explicit logical-device identity. Capture-local IDs are monotonic and
independently allocated for logical devices, pipelines, sessions, and selected
dispatches. Folder names use IDs, never Vulkan handles or debug names.

```text
<capture-root>/capture.json
<capture-root>/pipeline_000000/pipeline.json
<capture-root>/pipeline_000000/session_000000/session.json
<capture-root>/pipeline_000000/session_000000/dispatch_000000/dispatch.json
```

There are no `pipelines`, `sessions`, or `dispatches` container directories.
Pipeline and session tracking identities may exist without model nodes; their
nodes are materialized only when a selected dispatch occurs. `VK_EXT_debug_utils`
names are mutable metadata and never rename paths. Vulkan object handles and
command-buffer handles are optional diagnostics only.

`capture.json` contains capture and device metadata together and references
materialized pipeline documents. Pipeline documents reference sessions, session
documents reference selected dispatches, and dispatch documents record the
per-session executed occurrence index. Artifact descriptors contain only a
relative path, artifact type, and byte size; hashes are intentionally absent.
Serialization orders descriptors canonically by `(generic path, artifact type,
byte size)`, so insertion order cannot change the emitted JSON. Capture status
is one of `incomplete`, `complete`, or `error` so a later writer can publish
completion atomically.

The capture namespace has one central policy. Metadata names are exactly
`capture.json`, `pipeline.json`, `session.json`, and `dispatch.json`; hierarchy
directory names are `pipeline_<id>`, `session_<id>`, and `dispatch_<id>` with the
documented zero-padding; internal publication names begin `.capture-internal-`
and end `.tmp`. Public artifact file names are portable ASCII only, 1-120 bytes,
begin with an ASCII letter or digit, and then contain only letters, digits,
period, underscore, or hyphen. The grammar rejects separators (`/` and `\\`),
colon syntax, controls, whitespace, non-ASCII/Unicode normalization ambiguity,
`.`/`..`, trailing periods, every metadata/directory/internal reserved name, and
every `.tmp` suffix. Collision keys preserve case-sensitive filesystem
semantics.

`CaptureModel` is externally synchronized and non-copyable/non-movable. It has
one owner, and model mutations are serialized on the writer thread rather than
protected by internal model locks. Capture-local IDs and executed indices
allocate through the complete `uint64_t` range and throw `std::overflow_error`
on the next allocation instead of wrapping.

## Root Policy

The root policy is `RejectExisting`. Model/configuration validation occurs
before the filesystem claim. `CaptureWriter` creates missing parent directories
using normal process path resolution, then claims the exact final root with one
exclusive `mkdir`; existing files, directories, and symlinks are collisions. The
new root is opened with `O_NOFOLLOW` and retained as a directory descriptor.
Hierarchy creation and publication below it use descriptor-relative `*at`
operations, and every opened directory component uses `O_NOFOLLOW`. Repeated
creation calls may reuse ordinary directories already created within this
exclusively owned root. Create-new publication requires Linux
`renameat2(RENAME_NOREPLACE)`; lack of that primitive is reported as
`UnsupportedFilesystem` rather than using an overwrite-prone fallback.

This is a single-writer diagnostic-output contract, not a same-user security
boundary. The layer assumes no other process edits, renames, or replaces entries
in an active capture. It does not retain the configured ancestor chain, record
inode identities, lock an ownership marker, or rescan output to detect external
mutation. Descriptor-relative/no-follow operations prevent ordinary path
traversal through a symlink component, but deliberate concurrent mutation is
unsupported and can leave output at a renamed location or otherwise invalidate
the capture.

If initial `capture.json` publication fails, the writer best-effort removes its
unique temporary and removes the claimed root if it is still empty.
Model-construction failures occur before the claim. Once the initial valid
document is published, the root is preserved so later failures retain the last
published diagnostic state.

## Asynchronous Writer And Filesystem Publication

`CaptureWriter` owns one `CaptureModel`, one FIFO with a bounded number of
entries, and one writer thread per capture. The bound applies to queued job
count, not to the total bytes retained by job payloads or pending pipeline
artifacts. The writer thread is the sole filesystem writer and the sole model
mutator/serializer. Submitted payloads own their strings, byte vectors, paths,
optional diagnostic integers, and typed capture-local IDs; they retain no
application pointers, mutable tracking records, or Vulkan objects that require
later dereference. Model reservations return typed asynchronous results without
throwing through a Vulkan ABI boundary.

Queue admission is entry-count-bounded and lossless while the writer is running.
Submission blocks while it is full, applying explicit backpressure. A later
terminal failure can reject previously queued work; its completion result and
the writer's terminal state report that rejection, so admission alone does not
prove filesystem publication. The concrete immutable `Job` variant is
compile-time constrained to nothrow move construction. Multiple producers are
supported, while one submission mutex establishes a total enqueue order.
Job/future construction and queue-allocation failures are translated to rejected
`WriterSubmission` values rather than escaping from the submission path. A
terminal complete/error request closes admission after it enters the FIFO. A
fatal model or filesystem failure closes the queue, wakes blocked producers,
rejects queued work with an explicit writer error, and makes a best-effort
coherent terminal publication.

The writer publishes `capture.json` with `incomplete` status immediately after
claiming the root. Every JSON document and artifact is written completely to a
uniquely named sibling created with `O_EXCL` in the reserved
`.capture-internal-<unique>.tmp` namespace. The file is closed successfully
before an atomic descriptor-relative rename. A failed operation best-effort
unlinks that unique temporary. First publication of every path uses atomic
no-replace; later metadata publication uses atomic replacement. Artifacts are
permanently create-new. There are no per-publication identity maps or
post-publication filesystem validation passes.

The publication scheme provides process-crash consistency: every visible
document is a complete atomic publication, so recovery sees the last valid
`incomplete`, `complete`, or `error` document and only fully published artifact
files. It does not promise power-loss durability; it deliberately does not issue
the file and containing-directory `fsync`/`fdatasync` sequence required for that
stronger contract.

Pending artifact intent is held separately from model descriptors. Descriptor
capacity is reserved before I/O; after atomic file publication the descriptor
commit is nothrow. Therefore a descriptor cannot precede its file, and a
successfully published artifact cannot become a silent orphan because descriptor
allocation failed. On a middle failure while materializing queued artifacts,
terminal metadata contains exactly the previously published files.

Pipeline/session/dispatch directories are created only when a selected dispatch
reservation materializes that hierarchy. Static pipeline artifacts may be queued
earlier; their owned bytes remain in writer memory and are written in canonical
file-name order only when that pipeline materializes. Never-selected pipelines
and sessions produce no directories. Friendly names rewrite metadata but cannot
change ID-derived directory names. Metadata serialization uses ordered keys,
stable IDs, canonical artifact descriptors, and no hashes.

`shutdown()` and destruction block while draining previously accepted work and
joining the writer thread. Explicit shutdown normally uses a terminal FIFO job,
but failure to construct that job or retrieve its future switches to a
pre-existing non-allocating automatic-completion signal: admission closes, the
queue drains, and the writer thread performs terminal publication directly. The
`noexcept` destructor catches every exception, closes/drains/joins without
constructing a promise, job, future, or error message, and leaves a valid
`incomplete`/`error` capture if automatic completion itself cannot allocate.

Queue capacity is an internal/test construction option and is not a public layer
setting. Cross-thread filesystem fault tests extend the existing capture
fault-injection architecture to cover root claim, directory creation,
temporary-file open/write, and rename failures.
