<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
SPDX-License-Identifier: MIT
-->

# Command-buffer statistics synchronization and snapshot ordering

## Purpose

The layer injects transfer commands around `vkCmdDispatchDataGraphARM` while preserving the application's Vulkan synchronization contract. This document defines the buffers, barriers, submission rewriting, and lifetime rules used to support primary reuse, repeated secondary execution, and `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`.

## Buffer hierarchy

For each captured dispatch the implementation uses these storage levels:

1. **Session statistics storage (`S`)** is the statistics bind-point memory shared by every dispatch using one data-graph session.
2. **Record-time snapshot (`R`)** is allocated while recording a data-graph dispatch. The injected sequence clears `S`, executes the graph, and copies `S` to `R`.
3. **Secondary occurrence snapshot (`O`)** is allocated when a secondary is referenced by `vkCmdExecuteCommands`. Immediately after each secondary execution, the primary copies `R` to a unique `O`. A direct primary dispatch uses `R` as its occurrence source.
4. **Submission snapshot (`U`)** is allocated for every selected executed occurrence during queue-submit admission. A private archive command buffer copies its `R` or `O` source to `U`. Only `U` is mapped by the collector.

`R` and `O` belong to recorded command-buffer metadata. `U` and the private archive command pool belong to the accepted collector job until its completion marker retires.

## Dispatch instrumentation

A captured dispatch records:

```text
barrier: prior DATA_GRAPH writes / prior TRANSFER reads -> TRANSFER write
fill S with zero
barrier: TRANSFER write -> DATA_GRAPH read/write
dispatch data graph using S
barrier: DATA_GRAPH write -> TRANSFER read
copy S -> R
barrier: TRANSFER read -> DATA_GRAPH
barrier: TRANSFER write -> HOST read
```

The first and penultimate barriers connect injected transfers to the application's data-graph dependency chain. They do not independently synchronize queues.

## Secondary command buffers

A call `vkCmdExecuteCommands(primary, N, secondaries)` is forwarded as `N` one-secondary calls. After each call, the layer records:

```text
barrier: R TRANSFER write -> TRANSFER read
copy R -> unique O
barrier: TRANSFER read/write -> DATA_GRAPH and HOST
```

Splitting is required because one archive after the entire array would observe only the final overwrite when a secondary appears repeatedly. Primary metadata references `O`, not the secondary's shared `R`.

## Queue-submit rewriting

Planning computes executed indices and selected dispatch IDs without reserving record-time buffers. It then allocates one unique `U` per selected occurrence and inserts one private archive command buffer after every application primary containing selected occurrences.

```text
application: [P, P]
forwarded:   [P, A0, P, A1]

A0: sources from first P  -> unique U0 buffers
A1: sources from second P -> unique U1 buffers
```

For submit2, each inserted `VkCommandBufferSubmitInfo` inherits the preceding application command buffer's device mask. For legacy submits, the layer clones `VkDeviceGroupSubmitInfo` and expands its command-buffer mask array in lockstep with the inserted archive command buffers.

Each archive command buffer records:

```text
barrier: occurrence TRANSFER writes -> archive TRANSFER reads
copy each selected occurrence source -> its unique U
barrier: archive TRANSFER reads/writes -> DATA_GRAPH and HOST
```

The archive is inside the application submit and therefore completes before that submit's signal semaphore operation. A later marker submit would be too late: another queue could pass the application semaphore and overwrite the record-time source first.

## Host-side lock ownership

Snapshot allocation and private archive command-buffer allocation/recording call the downstream Vulkan implementation without holding the layer-wide `g_vulkanLock`. Record metadata is staged or pinned with shared ownership, then committed only after the lock is reacquired.

Submission archive materialization retains `submissionAdmissionMutex` while the global lock is released. That admission lock serializes tentative dispatch/session identities, snapshot in-flight-use accounting, other submissions, and collector finalization until the submission is accepted or cancelled.

## Ordering cases

### One primary, one submission

The archive follows the primary on the same queue. Transfer barriers make the occurrence source visible to the archive and make `U` visible to the collector.

### Same primary multiple times in one submit

An archive is interleaved after every primary occurrence. The first archive reads the shared occurrence source before the next primary may overwrite it.

### Repeated secondary in one primary

Every secondary execution is immediately followed by its own `R -> O` copy. The submission archive later copies every distinct `O` to a distinct `U`.

### Reuse across submissions on one queue

Every accepted submission gets new `U` buffers. Queue order and barriers order archive reads before later reuse of record-time sources. Collection of an earlier `U` may remain pending while later submissions proceed.

### Reuse on different queues

The application must already synchronize uses of the same data-graph session and its bound transient/statistics memory. The layer adds no private semaphore and invents no cross-queue order. Its `TRANSFER -> DATA_GRAPH` and `DATA_GRAPH -> TRANSFER` bridge barriers extend the application's valid semaphore dependency to include clears, record-time copies, secondary occurrence copies, and submission archives.

If queue A signals and queue B waits with scopes that correctly order session use, A's archive is therefore ordered before B may overwrite the shared record-time source.

### Simultaneous-use command buffers

`SIMULTANEOUS_USE` permits the command-buffer object to be pending more than once; it does not remove synchronization requirements for the session and its bound memory. Every submission receives unique `U` storage, while valid application session synchronization orders reuse of `R` and `O`. The layer neither waits for CPU collection nor rejects a second pending submission solely because its command buffer is already in flight.

### Different sessions

Different sessions have independent `S` storage. Their submission snapshots and collector jobs are independent, so their fences may retire out of submission order.

## Completion and resource lifetime

After forwarding the rewritten application submit, the layer submits an empty marker on the same queue. Its fence proves that application commands and inserted archives completed. The collector maps only `U`, copies bytes to owned host vectors, and releases the submission snapshots and archive pool.

The record-time source keeps an aggregate in-flight-use count for diagnostics and lifetime tests, but it is not an exclusion lock. Unique `U` buffers are the actual collector reservations.

If the application submit fails, no identities are committed and tentative resources are destroyed. If marker creation or submission fails after application success, the application result is preserved and accepted GPU resources are retained until device teardown; they are not destroyed while possibly pending.

## Unsupported selected cases

These restrictions describe capture support, not necessarily invalid Vulkan application behavior.

### Protected command buffers and submissions

Protected data-graph execution is valid Vulkan usage, but the layer does not inject unprotected snapshot buffers, archive command buffers, or host-readable collection into protected work. Selecting a protected occurrence terminally disables capture and forwards the original application submit unchanged, without archive commands, markers, or committed capture identities. Unselected protected work is forwarded unchanged.

### Application-provided neural-statistics configuration

This does **not** mean that the application initialized a statistics buffer before passing it to Vulkan. The official `VK_ARM_data_graph_neural_accelerator_statistics` extension allows an application to configure statistics by chaining:

- `VkDataGraphPipelineNeuralStatisticsCreateInfoARM` into `VkDataGraphPipelineCreateInfoARM`; and/or
- `VkDataGraphPipelineSessionNeuralStatisticsCreateInfoARM` into `VkDataGraphPipelineSessionCreateInfoARM`.

The layer normally injects those same structure types to enable pipeline statistics and select the configured statistics mode. It currently refuses to replace or duplicate application-provided structures because that could override the application's requested state or mode. The application pipeline/session remains valid and is created without layer instrumentation. Selecting one of its dispatches terminally disables capture and forwards the original application submit unchanged, without archive commands, markers, or committed capture identities.

A future coexistence policy could accept an application configuration when it is compatible with the layer's requested mode and bind-point requirements. That is not implemented today.

### Cross-queue-family session reuse

The neural-statistics bind point provides one statistics allocation per session. The layer aliases that allocation with a hidden transfer buffer `S`, then copies each dispatch result into its record-time and submission-owned snapshots.

A session may be reused by command buffers from different queue families when every participating family supports data-graph and transfer operations. The application remains responsible for synchronizing the session executions, for example with a semaphore:

```text
Q0/family 0:
  clear S
  dispatch session
  copy S -> R0
  signal application dependency

Q1/family 1:
  wait application dependency
  clear S
  dispatch session
  copy S -> R1
```

When the logical device enables two or more queue families that support both data-graph and transfer operations, the hidden alias buffer `S` is created with `VK_SHARING_MODE_CONCURRENT` and lists every such enabled family. Consequently, accesses to `S` do not require queue-family ownership transfers. When the logical device enables only one compatible family, `S` remains exclusive because no cross-family access is possible.

The record-time, secondary-occurrence, and submission-owned snapshot buffers remain exclusive. Each is created for and used by one command-buffer queue family, so they do not cross ownership domains.

This support does not add execution ordering: unsynchronized dispatches using the same session may still race on the session statistics allocation. The application-provided dependency that orders the session executions also orders the layer's injected `DATA_GRAPH -> TRANSFER` archive and subsequent `TRANSFER -> DATA_GRAPH` preparation through the established synchronization chain.

### Legacy device-group command-buffer masks

A legacy device-group submission can associate one mask with each submitted command buffer through `VkDeviceGroupSubmitInfo::pCommandBufferDeviceMasks`. For example, on a two-device group:

```text
pCommandBuffers:            [P0, P1]
pCommandBufferDeviceMasks: [0b01, 0b10]
```

`P0` executes on physical device 0 and `P1` executes on physical device 1. If both contain selected occurrences, the layer must rewrite the submission as:

```text
pCommandBuffers:            [P0, A0, P1, A1]
pCommandBufferDeviceMasks: [0b01, 0b01, 0b10, 0b10]
```

Each inserted archive command buffer executes on the same physical-device mask as the application command buffer it follows. The layer clones the application's submit and `pNext` chain, expands `VkDeviceGroupSubmitInfo::pCommandBufferDeviceMasks` in lockstep with `pCommandBuffers`, and leaves the application-owned structures unchanged. Wait- and signal-semaphore device indices are preserved.

Submit2 does not have this limitation: every `VkCommandBufferSubmitInfo` carries its own `deviceMask`, and the layer copies the preceding application's mask into the inserted archive command-buffer info.

### Queue families without transfer capability

Statistics capture injects `vkCmdFillBuffer` and `vkCmdCopyBuffer`. A command buffer that can execute data graphs but belongs to a queue family without transfer capability cannot legally record those injected commands. The application workload may be valid, but the layer cannot capture it using the current transfer-based mechanism. Unselected unsupported occurrences remain transparent. Selecting one terminally disables capture and forwards the original application submit unchanged, without archive commands, markers, or committed capture identities.

### Other required capabilities

Selected capture also requires an available synchronization2 command route and memory-requirements2 dedicated-allocation reporting. These are layer instrumentation requirements rather than general validity requirements for an uninstrumented application; if a selected occurrence cannot meet them, capture terminates while the application submit is forwarded unchanged.
