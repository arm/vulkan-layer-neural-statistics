/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>

namespace capture::fault
{
enum class Point
{
    TrackPipelineAfterInsertion,
    TrackSessionAfterInsertion,
    MaterializeAfterPipelineInsertion,
    MaterializeAfterSessionInsertion,
    ReserveDispatchAfterInsertionBeforeLinkage,
    UpdatePipelineNameBeforeCommit,
    UpdateSessionNameBeforeCommit,
    AddPipelineArtifactBeforeCommit,
    AddDispatchArtifactBeforeCommit,
    MarkErrorBeforeCommit,
    InstanceAfterDownstreamCreateBeforePublication,
    DeviceAfterDownstreamCreateBeforePublication,
    ShaderRecordBeforeInsertion,
    ShaderMapInsertion,
    ShaderRecordAfterInsertionBeforeCommit,
    PipelineRecordBeforeInsertion,
    PipelineMapInsertion,
    PipelineRecordAfterInsertionBeforeCommit,
    SessionMapInsertion,
    CommandBufferMapInsertion,
    DispatchMetadataRecording,
    DispatchMetadataPublication,
    SecondaryExecutionMetadataPropagation,
    LegacySubmitRewrite,
    Submit2Rewrite,
    WriterClaimCreateRoot,
    WriterCreateDirectory,
    WriterOpenTemporaryFile,
    WriterWriteTemporaryFile,
    WriterRenameTemporaryFile,
    WriterTerminalJobCreation,
    WriterFutureRetrieval,
    WriterPipelineArtifactSubmission,
    WriterDispatchArtifactSubmission,
    WriterDestructorAutomaticCompletion,
    CollectorJobConstruction,
    CollectorQueueAdmission,
    CollectorPollingBookkeeping,
    CollectorCompletionCopy,
};

enum class Failure
{
    BadAllocation,
    CounterExhaustion,
    Unexpected,
};

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
class ScopedInjection
{
  public:
    explicit ScopedInjection(Point point, Failure failure = Failure::BadAllocation,
                             std::size_t occurrence = 1) noexcept;
    ~ScopedInjection() noexcept;

    ScopedInjection(const ScopedInjection &) = delete;
    ScopedInjection &operator=(const ScopedInjection &) = delete;

  private:
    Point previousPoint_;
    Failure previousFailure_;
    std::size_t previousRemaining_;
    bool previousEnabled_;
};

// Cross-thread injection used by the asynchronous writer tests. Only one global
// injection scope may be active at a time.
class ScopedGlobalInjection
{
  public:
    explicit ScopedGlobalInjection(Point point, Failure failure = Failure::Unexpected,
                                   std::size_t occurrence = 1) noexcept;
    ~ScopedGlobalInjection() noexcept;

    ScopedGlobalInjection(const ScopedGlobalInjection &) = delete;
    ScopedGlobalInjection &operator=(const ScopedGlobalInjection &) = delete;

  private:
    Point previousPoint_;
    Failure previousFailure_;
    std::size_t previousRemaining_;
    bool previousEnabled_;
};

// Makes every allocation-sensitive production checkpoint fail with bad_alloc
// from the selected occurrence onward. This is independent of the ordinary
// one-shot global injection so a primary failure and persistent translation or
// publication failure can be exercised together.
class ScopedGlobalPersistentAllocationFailure
{
  public:
    explicit ScopedGlobalPersistentAllocationFailure(std::size_t failFromOccurrence = 1) noexcept;
    ~ScopedGlobalPersistentAllocationFailure() noexcept;

    ScopedGlobalPersistentAllocationFailure(const ScopedGlobalPersistentAllocationFailure &) =
        delete;
    ScopedGlobalPersistentAllocationFailure &operator=(
        const ScopedGlobalPersistentAllocationFailure &) = delete;

  private:
    std::size_t previousRemaining_{0};
    bool previousEnabled_{false};
    bool previousFailing_{false};
};

// Deterministic cross-thread checkpoint barrier for asynchronous writer tests.
class ScopedGlobalPause
{
  public:
    explicit ScopedGlobalPause(Point point) noexcept;
    ~ScopedGlobalPause() noexcept;

    ScopedGlobalPause(const ScopedGlobalPause &) = delete;
    ScopedGlobalPause &operator=(const ScopedGlobalPause &) = delete;

    void waitUntilReached();
    void release() noexcept;

  private:
    Point point_;
};

void Checkpoint(Point point);
void AllocationCheckpoint();
#else
inline void Checkpoint(Point) noexcept {}
inline void AllocationCheckpoint() noexcept {}
#endif
} // namespace capture::fault
