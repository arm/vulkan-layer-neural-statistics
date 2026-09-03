/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace capture
{
enum class WriterErrorCode
{
    None,
    NotStarted,
    AlreadyStarted,
    RootCollision,
    InvalidRequest,
    QueueClosed,
    ModelFailure,
    IoFailure,
    UnsupportedFilesystem,
    Terminal,
    InternalFailure,
    OutOfHostMemory,
};

struct WriterError
{
    WriterErrorCode code{WriterErrorCode::None};
    std::string message;
    std::filesystem::path path;
};

template <typename T> class WriterResult
{
  public:
    static WriterResult Success(T value)
    {
        WriterResult result;
        result.value_.emplace(std::move(value));
        return result;
    }

    static WriterResult Failure(WriterError error)
    {
        WriterResult result;
        result.error_ = std::move(error);
        return result;
    }

    static WriterResult FailureCode(WriterErrorCode code) noexcept
    {
        return WriterResult(CodeFailureTag{}, code);
    }

    bool ok() const noexcept
    {
        return value_.has_value();
    }
    const T &value() const
    {
        return value_.value();
    }
    T &value()
    {
        return value_.value();
    }
    const WriterError &error() const noexcept
    {
        return error_;
    }

  private:
    struct CodeFailureTag
    {
    };

    WriterResult() = default;
    WriterResult(CodeFailureTag, WriterErrorCode code) noexcept
    {
        error_.code = code;
    }

    std::optional<T> value_;
    WriterError error_;
};

template <> class WriterResult<void>
{
  public:
    static WriterResult Success()
    {
        return WriterResult(true, {});
    }
    static WriterResult Failure(WriterError error)
    {
        return WriterResult(false, std::move(error));
    }
    static WriterResult FailureCode(WriterErrorCode code) noexcept
    {
        return WriterResult(code);
    }

    bool ok() const noexcept
    {
        return ok_;
    }
    const WriterError &error() const noexcept
    {
        return error_;
    }

  private:
    WriterResult(bool ok, WriterError error) : ok_(ok), error_(std::move(error)) {}

    explicit WriterResult(WriterErrorCode code) noexcept
    {
        error_.code = code;
    }

    bool ok_{false};
    WriterError error_;
};

template <typename T> struct WriterSubmission
{
    bool accepted{false};
    WriterError error;
    std::future<WriterResult<T>> completion;

    WriterResult<T> wait() noexcept
    {
        try
        {
            if (!accepted)
            {
                return WriterResult<T>::Failure(std::move(error));
            }
            return completion.get();
        }
        catch (...)
        {
            return WriterResult<T>::FailureCode(WriterErrorCode::InternalFailure);
        }
    }
};

enum class WriterState
{
    NeverStarted,
    Starting,
    Running,
    Complete,
    Error,
};

// Nonallocating lifecycle snapshot suitable for noexcept/ABI-boundary
// diagnostics. Rich strings and paths are intentionally separate.
struct WriterSnapshot
{
    WriterState state{WriterState::NeverStarted};
    bool accepting{false};
    uint64_t acceptedJobs{0};
    uint64_t completedJobs{0};
    WriterErrorCode terminalErrorCode{WriterErrorCode::None};
};

struct CaptureWriterOptions
{
    std::filesystem::path captureRoot;
    CaptureMetadata metadata;
    // Test/runtime-planner input only; the writer never evaluates filters.
    DispatchFilter dispatchFilter{DispatchFilter::All()};
    // Internal/test tuning only. This is deliberately not a layer setting.
    std::size_t queueCapacity{64};
};

// CaptureWriter owns the CaptureModel and is its only mutator. Public requests
// are copied/moved into immutable, self-contained queue payloads containing
// only owned values and capture-local typed IDs. Submission blocks when the
// entry-count-bounded queue is full, providing lossless backpressure. The queue
// does not impose a byte bound on payloads or pending artifact storage.
class CaptureWriter
{
  public:
    explicit CaptureWriter(CaptureWriterOptions options);
    ~CaptureWriter() noexcept;

    CaptureWriter(const CaptureWriter &) = delete;
    CaptureWriter &operator=(const CaptureWriter &) = delete;
    CaptureWriter(CaptureWriter &&) = delete;
    CaptureWriter &operator=(CaptureWriter &&) = delete;

    WriterResult<void> start();

    WriterSubmission<void> trackPipeline(
        PipelineId pipelineId, PipelineMetadata metadata,
        std::optional<uint64_t> diagnosticHandle = std::nullopt) noexcept;
    WriterSubmission<void> trackSession(
        SessionId sessionId, PipelineId pipelineId, uint64_t flags,
        std::optional<uint64_t> diagnosticHandle = std::nullopt) noexcept;
    WriterSubmission<void> materializeDispatch(
        SessionId sessionId, DispatchReservation reservation,
        std::optional<uint64_t> diagnosticCommandBufferHandle = std::nullopt) noexcept;
    WriterSubmission<void> updateMetadata(CaptureMetadata metadata) noexcept;

    WriterSubmission<void> updatePipelineFriendlyName(PipelineId pipelineId,
                                                      std::string_view friendlyName) noexcept;
    WriterSubmission<void> updatePipelineMetadata(PipelineId pipelineId,
                                                  PipelineMetadata metadata) noexcept;
    WriterSubmission<void> updateSessionFriendlyName(SessionId sessionId,
                                                     std::string_view friendlyName) noexcept;

    WriterSubmission<void> writePipelineBinaryArtifact(PipelineId pipelineId,
                                                       std::string_view fileName, ArtifactType type,
                                                       std::span<const uint8_t> contents) noexcept;
    WriterSubmission<void> writePipelineTextArtifact(PipelineId pipelineId,
                                                     std::string_view fileName, ArtifactType type,
                                                     std::string_view contents) noexcept;
    WriterSubmission<void> writeDispatchBinaryArtifact(DispatchId dispatchId,
                                                       std::string_view fileName, ArtifactType type,
                                                       std::span<const uint8_t> contents) noexcept;
    WriterSubmission<void> writeDispatchTextArtifact(DispatchId dispatchId,
                                                     std::string_view fileName, ArtifactType type,
                                                     std::string_view contents) noexcept;

    // Explicit deterministic republish job. Normal mutations publish their
    // affected metadata automatically; this is useful at lifecycle boundaries.
    WriterSubmission<void> publishDocuments() noexcept;
    WriterSubmission<void> finishComplete() noexcept;
    WriterSubmission<void> finishError(std::string_view message) noexcept;

    // Atomically run a short, non-reentrant identity commit while the writer
    // is both Running and still accepting jobs.
    bool commitIfAccepting(void (*callback)(void *) noexcept, void *context) noexcept;

    // Blocks until all previously accepted work and the terminal request have
    // completed, then joins the writer thread. Destructor performs the same
    // complete-and-drain shutdown when the writer is still running.
    WriterResult<void> shutdown();

    // Always nonallocating. Use terminalError() only when rich diagnostic text
    // or a path is needed; that separate acquisition may allocate and throw.
    WriterSnapshot snapshot() const noexcept;
    WriterError terminalError() const;

    // Deterministic test/diagnostic synchronization; not a layer setting.
    bool waitForBlockedSubmitters(std::size_t count);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace capture
