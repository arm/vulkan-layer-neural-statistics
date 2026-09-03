/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_writer.hpp"

#include "bounded_queue.hpp"
#include "capture_filesystem.hpp"
#include "capture_store.hpp"
#include "fault_injection.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <variant>

namespace capture
{
namespace
{
WriterError Error(WriterErrorCode code, std::string message, std::filesystem::path path = {})
{
    return WriterError{code, std::move(message), std::move(path)};
}

WriterErrorCode FallbackExceptionCode(std::exception_ptr exception) noexcept
{
    try
    {
        if (exception)
        {
            std::rethrow_exception(exception);
        }
    }
    catch (const std::bad_alloc &)
    {
        return WriterErrorCode::OutOfHostMemory;
    }
    catch (...)
    {
        return WriterErrorCode::InternalFailure;
    }
    return WriterErrorCode::InternalFailure;
}

WriterError RichExceptionError(std::exception_ptr exception)
{
    fault::AllocationCheckpoint();
    try
    {
        if (exception)
        {
            std::rethrow_exception(exception);
        }
    }
    catch (const CaptureFileSystemError &error)
    {
        WriterErrorCode code = error.operation() == FileSystemOperation::UnsupportedCapability
                                   ? WriterErrorCode::UnsupportedFilesystem
                                   : WriterErrorCode::IoFailure;
        if (error.operation() == FileSystemOperation::ClaimRoot)
        {
            std::error_code ignored;
            const auto status = std::filesystem::symlink_status(error.path(), ignored);
            if (!ignored && status.type() != std::filesystem::file_type::not_found)
            {
                code = WriterErrorCode::RootCollision;
            }
        }
        return Error(code, error.what(), error.path());
    }
    catch (const CaptureTerminalError &error)
    {
        return Error(WriterErrorCode::ModelFailure, error.what());
    }
    catch (const std::invalid_argument &error)
    {
        return Error(WriterErrorCode::InvalidRequest, error.what());
    }
    catch (const std::out_of_range &error)
    {
        return Error(WriterErrorCode::InvalidRequest, error.what());
    }
    catch (const std::bad_alloc &)
    {
        return Error(WriterErrorCode::OutOfHostMemory, "capture writer host allocation failure");
    }
    catch (const std::exception &error)
    {
        return Error(WriterErrorCode::InternalFailure, error.what());
    }
    catch (...)
    {
        return Error(WriterErrorCode::InternalFailure, "unknown capture writer failure");
    }
    return Error(WriterErrorCode::InternalFailure, "unknown capture writer failure");
}

struct ErrorReport
{
    WriterErrorCode code{WriterErrorCode::InternalFailure};
    WriterError diagnostics;
    bool hasDiagnostics{false};
};

void CaptureException(std::exception_ptr exception, ErrorReport &report) noexcept
{
    report.code = FallbackExceptionCode(exception);
    try
    {
        report.diagnostics = RichExceptionError(exception);
        report.code = report.diagnostics.code;
        report.hasDiagnostics = true;
    }
    catch (...)
    {
        report.hasDiagnostics = false;
    }
}

void SetErrorNoThrow(WriterError &target, ErrorReport &report) noexcept
{
    target.code = report.code;
    if (!report.hasDiagnostics)
    {
        return;
    }
    try
    {
        fault::AllocationCheckpoint();
        target = std::move(report.diagnostics);
    }
    catch (...)
    {
        target.code = report.code;
    }
}

template <typename T>
void CompleteFailure(std::promise<WriterResult<T>> &completion, const ErrorReport &report) noexcept
{
    try
    {
        if (report.hasDiagnostics)
        {
            try
            {
                fault::AllocationCheckpoint();
                completion.set_value(WriterResult<T>::Failure(report.diagnostics));
                return;
            }
            catch (...)
            {
            }
        }
        completion.set_value(WriterResult<T>::FailureCode(report.code));
    }
    catch (...)
    {
        try
        {
            completion.set_value(WriterResult<T>::FailureCode(report.code));
        }
        catch (...)
        {
        }
    }
}

template <typename ResultType, typename Operation>
void CompleteDomainRequest(std::promise<WriterResult<ResultType>> &completion, Operation operation)
{
    try
    {
        completion.set_value(WriterResult<ResultType>::Success(operation()));
    }
    catch (const std::invalid_argument &error)
    {
        completion.set_value(WriterResult<ResultType>::Failure(
            Error(WriterErrorCode::InvalidRequest, error.what())));
    }
    catch (const std::out_of_range &error)
    {
        completion.set_value(WriterResult<ResultType>::Failure(
            Error(WriterErrorCode::InvalidRequest, error.what())));
    }
}

template <typename Operation>
void CompleteDomainRequest(std::promise<WriterResult<void>> &completion, Operation operation)
{
    try
    {
        operation();
        completion.set_value(WriterResult<void>::Success());
    }
    catch (const std::invalid_argument &error)
    {
        completion.set_value(
            WriterResult<void>::Failure(Error(WriterErrorCode::InvalidRequest, error.what())));
    }
    catch (const std::out_of_range &error)
    {
        completion.set_value(
            WriterResult<void>::Failure(Error(WriterErrorCode::InvalidRequest, error.what())));
    }
}

template <typename Operation> auto SafeSubmission(Operation &&operation) -> decltype(operation())
{
    using Submission = decltype(operation());
    try
    {
        return operation();
    }
    catch (...)
    {
        Submission submission;
        ErrorReport report;
        CaptureException(std::current_exception(), report);
        SetErrorNoThrow(submission.error, report);
        return submission;
    }
}
} // namespace

class CaptureWriter::Impl
{
  public:
    explicit Impl(CaptureWriterOptions options)
        : options_(std::move(options)), queue_(options_.queueCapacity)
    {
    }

    ~Impl() noexcept
    {
        shutdownNoThrow();
    }

    WriterResult<void> start()
    {
        std::lock_guard lifecycleLock(lifecycleMutex_);
        {
            std::lock_guard lock(stateMutex_);
            if (state_ != WriterState::NeverStarted)
            {
                return WriterResult<void>::Failure(
                    Error(WriterErrorCode::AlreadyStarted, "capture writer was already started"));
            }
            state_ = WriterState::Starting;
        }

        try
        {
            thread_ = std::thread(&Impl::threadMain, this);
        }
        catch (...)
        {
            ErrorReport report;
            CaptureException(std::current_exception(), report);
            publishStartupFailure(report);
            return startupResult();
        }

        std::unique_lock lock(stateMutex_);
        startCondition_.wait(lock, [this] { return startReady_; });
        if (startSucceeded_)
        {
            return WriterResult<void>::Success();
        }
        return terminalFailureResultLocked();
    }

    struct TrackPipelineJob
    {
        using ResultType = void;
        PipelineId pipelineId;
        PipelineMetadata metadata;
        std::optional<uint64_t> diagnosticHandle;
        std::promise<WriterResult<void>> completion;
    };

    struct TrackSessionJob
    {
        using ResultType = void;
        SessionId sessionId;
        PipelineId pipelineId;
        uint64_t flags{0};
        std::optional<uint64_t> diagnosticHandle;
        std::promise<WriterResult<void>> completion;
    };

    struct MaterializeDispatchJob
    {
        using ResultType = void;
        SessionId sessionId;
        DispatchReservation reservation;
        std::optional<uint64_t> diagnosticCommandBufferHandle;
        std::promise<WriterResult<void>> completion;
    };

    struct UpdateMetadataJob
    {
        using ResultType = void;
        CaptureMetadata metadata;
        std::promise<WriterResult<void>> completion;
    };

    struct UpdatePipelineNameJob
    {
        using ResultType = void;
        PipelineId pipelineId;
        std::string friendlyName;
        std::promise<WriterResult<void>> completion;
    };

    struct UpdatePipelineMetadataJob
    {
        using ResultType = void;
        PipelineId pipelineId;
        PipelineMetadata metadata;
        std::promise<WriterResult<void>> completion;
    };

    struct UpdateSessionNameJob
    {
        using ResultType = void;
        SessionId sessionId;
        std::string friendlyName;
        std::promise<WriterResult<void>> completion;
    };

    struct PipelineBinaryArtifactJob
    {
        using ResultType = void;
        PipelineId pipelineId;
        std::filesystem::path fileName;
        ArtifactType type{};
        std::vector<uint8_t> contents;
        std::promise<WriterResult<void>> completion;
    };

    struct PipelineTextArtifactJob
    {
        using ResultType = void;
        PipelineId pipelineId;
        std::filesystem::path fileName;
        ArtifactType type{};
        std::string contents;
        std::promise<WriterResult<void>> completion;
    };

    struct DispatchBinaryArtifactJob
    {
        using ResultType = void;
        DispatchId dispatchId;
        std::filesystem::path fileName;
        ArtifactType type{};
        std::vector<uint8_t> contents;
        std::promise<WriterResult<void>> completion;
    };

    struct DispatchTextArtifactJob
    {
        using ResultType = void;
        DispatchId dispatchId;
        std::filesystem::path fileName;
        ArtifactType type{};
        std::string contents;
        std::promise<WriterResult<void>> completion;
    };

    struct PublishDocumentsJob
    {
        using ResultType = void;
        std::promise<WriterResult<void>> completion;
    };

    struct FinishCompleteJob
    {
        using ResultType = void;
        std::promise<WriterResult<void>> completion;
    };

    struct FinishErrorJob
    {
        using ResultType = void;
        std::string message;
        std::promise<WriterResult<void>> completion;
    };

    using Job =
        std::variant<TrackPipelineJob, TrackSessionJob, MaterializeDispatchJob, UpdateMetadataJob,
                     UpdatePipelineNameJob, UpdatePipelineMetadataJob, UpdateSessionNameJob,
                     PipelineBinaryArtifactJob, PipelineTextArtifactJob, DispatchBinaryArtifactJob,
                     DispatchTextArtifactJob, PublishDocumentsJob, FinishCompleteJob,
                     FinishErrorJob>;

    static_assert(std::is_nothrow_move_constructible_v<Job>);

    template <typename JobType>
    WriterSubmission<typename JobType::ResultType> submit(JobType job, bool terminal = false)
    {
        using ResultType = typename JobType::ResultType;
        WriterSubmission<ResultType> submission;
        try
        {
            fault::Checkpoint(fault::Point::WriterFutureRetrieval);
            submission.completion = job.completion.get_future();
        }
        catch (...)
        {
            submission.error.code = WriterErrorCode::InternalFailure;
            return submission;
        }

        std::lock_guard submitLock(submitMutex_);
        {
            std::lock_guard stateLock(stateMutex_);
            if (state_ == WriterState::NeverStarted || state_ == WriterState::Starting)
            {
                submission.error =
                    Error(WriterErrorCode::NotStarted, "capture writer is not running");
                return submission;
            }
            if (!accepting_ || state_ != WriterState::Running)
            {
                submission.error.code = terminalErrorCode_ == WriterErrorCode::None
                                            ? WriterErrorCode::Terminal
                                            : terminalErrorCode_;
                if (hasTerminalDiagnostics_)
                {
                    try
                    {
                        fault::AllocationCheckpoint();
                        submission.error = terminalError_;
                    }
                    catch (...)
                    {
                        submission.error.code = terminalErrorCode_;
                    }
                }
                return submission;
            }
        }

        try
        {
            if (!queue_.push(Job(std::move(job))))
            {
                submission.error =
                    Error(WriterErrorCode::QueueClosed, "capture writer queue is closed");
                return submission;
            }
        }
        catch (...)
        {
            ErrorReport report;
            CaptureException(std::current_exception(), report);
            SetErrorNoThrow(submission.error, report);
            return submission;
        }

        {
            std::lock_guard stateLock(stateMutex_);
            ++acceptedJobs_;
            if (terminal)
            {
                accepting_ = false;
            }
        }
        submission.accepted = true;
        return submission;
    }

    WriterResult<void> shutdown()
    {
        std::lock_guard lifecycleLock(lifecycleMutex_);
        WriterResult<void> result = WriterResult<void>::Success();
        bool requestComplete = false;
        {
            std::lock_guard lock(stateMutex_);
            requestComplete = state_ == WriterState::Running && accepting_;
        }

        if (requestComplete)
        {
            WriterSubmission<void> terminal;
            try
            {
                fault::Checkpoint(fault::Point::WriterTerminalJobCreation);
                terminal = submit(FinishCompleteJob{}, true);
            }
            catch (...)
            {
                terminal.error.code = WriterErrorCode::InternalFailure;
            }
            if (terminal.accepted)
            {
                result = terminal.wait();
            }
            else
            {
                result = WriterResult<void>::Failure(std::move(terminal.error));
                beginAutomaticCompletion();
            }
        }

        if (thread_.joinable())
        {
            thread_.join();
        }

        const auto current = snapshot();
        if (current.state == WriterState::Complete)
        {
            return WriterResult<void>::Success();
        }
        if (current.state == WriterState::Error)
        {
            std::lock_guard lock(stateMutex_);
            if (terminalErrorCode_ != WriterErrorCode::None)
            {
                return terminalFailureResultLocked();
            }
            if (!result.ok() && result.error().code != WriterErrorCode::None)
            {
                try
                {
                    return WriterResult<void>::Failure(result.error());
                }
                catch (...)
                {
                    return WriterResult<void>::FailureCode(result.error().code);
                }
            }
            return WriterResult<void>::FailureCode(WriterErrorCode::InternalFailure);
        }
        return result;
    }

    WriterSnapshot snapshot() const noexcept
    {
        std::lock_guard lock(stateMutex_);
        return WriterSnapshot{state_, accepting_, acceptedJobs_, completedJobs_,
                              terminalErrorCode_};
    }

    bool commitIfAccepting(void (*callback)(void *) noexcept, void *context) noexcept
    {
        std::lock_guard lock(stateMutex_);
        if (callback == nullptr || state_ != WriterState::Running || !accepting_)
        {
            return false;
        }
        callback(context);
        return true;
    }

    WriterError terminalError() const
    {
        std::lock_guard lock(stateMutex_);
        if (hasTerminalDiagnostics_)
        {
            fault::AllocationCheckpoint();
            return terminalError_;
        }
        WriterError error;
        error.code = terminalErrorCode_;
        return error;
    }

    bool waitForBlockedSubmitters(std::size_t count)
    {
        return queue_.waitForBlockedProducers(count);
    }

  private:
    WriterResult<void> startupResult() const noexcept
    {
        std::lock_guard lock(stateMutex_);
        if (startSucceeded_)
        {
            return WriterResult<void>::Success();
        }
        return terminalFailureResultLocked();
    }

    WriterResult<void> terminalFailureResultLocked() const noexcept
    {
        const WriterErrorCode code = terminalErrorCode_ == WriterErrorCode::None
                                         ? WriterErrorCode::InternalFailure
                                         : terminalErrorCode_;
        if (hasTerminalDiagnostics_)
        {
            try
            {
                fault::AllocationCheckpoint();
                return WriterResult<void>::Failure(terminalError_);
            }
            catch (...)
            {
            }
        }
        return WriterResult<void>::FailureCode(code);
    }

    void publishStartupFailure(ErrorReport &report) noexcept
    {
        try
        {
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Error;
            accepting_ = false;
            terminalErrorCode_ = report.code;
            hasTerminalDiagnostics_ = false;
            if (report.hasDiagnostics)
            {
                try
                {
                    fault::AllocationCheckpoint();
                    terminalError_ = std::move(report.diagnostics);
                    hasTerminalDiagnostics_ = true;
                }
                catch (...)
                {
                }
            }
            startSucceeded_ = false;
            startReady_ = true;
        }
        catch (...)
        {
        }
        queue_.close();
        startCondition_.notify_all();
    }

    void publishRuntimeFailure(const ErrorReport &report) noexcept
    {
        try
        {
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Error;
            accepting_ = false;
            terminalErrorCode_ = report.code;
            hasTerminalDiagnostics_ = false;
            if (report.hasDiagnostics)
            {
                try
                {
                    fault::AllocationCheckpoint();
                    terminalError_ = report.diagnostics;
                    hasTerminalDiagnostics_ = true;
                }
                catch (...)
                {
                }
            }
        }
        catch (...)
        {
        }
    }

    void publishCodeOnlyFailure(WriterErrorCode code) noexcept
    {
        try
        {
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Error;
            accepting_ = false;
            terminalErrorCode_ = code;
            hasTerminalDiagnostics_ = false;
            if (!startReady_)
            {
                startSucceeded_ = false;
                startReady_ = true;
            }
        }
        catch (...)
        {
        }
    }

    void beginAutomaticCompletion() noexcept
    {
        try
        {
            std::lock_guard submitLock(submitMutex_);
            {
                std::lock_guard stateLock(stateMutex_);
                if (state_ == WriterState::Running && accepting_)
                {
                    accepting_ = false;
                    automaticCompletionRequested_.store(true, std::memory_order_release);
                }
            }
            queue_.close();
        }
        catch (...)
        {
            automaticCompletionRequested_.store(true, std::memory_order_release);
            queue_.close();
        }
    }

    void shutdownNoThrow() noexcept
    {
        try
        {
            beginAutomaticCompletion();
        }
        catch (...)
        {
            queue_.close();
        }
        try
        {
            if (thread_.joinable())
            {
                thread_.join();
            }
        }
        catch (...)
        {
            try
            {
                if (thread_.joinable())
                {
                    thread_.detach();
                }
            }
            catch (...)
            {
            }
        }
    }

    void threadMain() noexcept
    {
        try
        {
            threadMainImpl();
        }
        catch (...)
        {
            const WriterErrorCode code = FallbackExceptionCode(std::current_exception());
            publishCodeOnlyFailure(code);
            queue_.close();
            ErrorReport report;
            report.code = code;
            rejectRemaining(report);
            startCondition_.notify_all();
        }
    }

    void threadMainImpl()
    {
        try
        {
            store_ = std::make_unique<CaptureStore>(options_.captureRoot, options_.metadata);
            {
                std::lock_guard lock(stateMutex_);
                state_ = WriterState::Running;
                accepting_ = true;
                startSucceeded_ = true;
                startReady_ = true;
            }
            startCondition_.notify_all();
        }
        catch (...)
        {
            ErrorReport report;
            CaptureException(std::current_exception(), report);
            publishStartupFailure(report);
            return;
        }

        while (auto queued = queue_.pop())
        {
            const bool terminal = std::holds_alternative<FinishCompleteJob>(*queued) ||
                                  std::holds_alternative<FinishErrorJob>(*queued);
            try
            {
                std::visit([this](auto &job) { process(job); }, *queued);
                incrementCompleted();
                if (terminal)
                {
                    queue_.close();
                    ErrorReport report;
                    report.code = WriterErrorCode::Terminal;
                    rejectRemaining(report);
                    return;
                }
            }
            catch (...)
            {
                ErrorReport report;
                CaptureException(std::current_exception(), report);
                if (report.hasDiagnostics && store_)
                {
                    store_->bestEffortRecordError(report.diagnostics.message);
                }
                publishCodeOnlyFailure(report.code);
                queue_.close();
                publishRuntimeFailure(report);
                reject(*queued, report);
                incrementCompleted();
                rejectRemaining(report);
                return;
            }
        }

        if (!automaticCompletionRequested_.exchange(false, std::memory_order_acq_rel))
        {
            return;
        }
        try
        {
            fault::Checkpoint(fault::Point::WriterDestructorAutomaticCompletion);
            store_->finishComplete();
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Complete;
            accepting_ = false;
            terminalErrorCode_ = WriterErrorCode::None;
            hasTerminalDiagnostics_ = false;
        }
        catch (...)
        {
            ErrorReport report;
            CaptureException(std::current_exception(), report);
            if (report.hasDiagnostics && store_)
            {
                store_->bestEffortRecordError(report.diagnostics.message);
            }
            publishRuntimeFailure(report);
        }
    }

    void incrementCompleted() noexcept
    {
        try
        {
            std::lock_guard lock(stateMutex_);
            ++completedJobs_;
        }
        catch (...)
        {
        }
    }

    void rejectRemaining(const ErrorReport &report) noexcept
    {
        try
        {
            while (auto queued = queue_.pop())
            {
                reject(*queued, report);
                incrementCompleted();
            }
        }
        catch (...)
        {
        }
    }

    static void reject(Job &job, const ErrorReport &report) noexcept
    {
        try
        {
            std::visit([&report](auto &value) { CompleteFailure(value.completion, report); }, job);
        }
        catch (...)
        {
        }
    }

    void process(TrackPipelineJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job] {
                                  store_->trackPipeline(job.pipelineId, std::move(job.metadata),
                                                        job.diagnosticHandle);
                              });
    }

    void process(TrackSessionJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job] {
                                  store_->trackSession(job.sessionId, job.pipelineId, job.flags,
                                                       job.diagnosticHandle);
                              });
    }

    void process(MaterializeDispatchJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job]
                              {
                                  store_->materializeDispatch(job.sessionId, job.reservation,
                                                              job.diagnosticCommandBufferHandle);
                              });
    }

    void process(UpdateMetadataJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job] { store_->updateMetadata(std::move(job.metadata)); });
    }

    void process(UpdatePipelineNameJob &job)
    {
        CompleteDomainRequest(
            job.completion,
            [this, &job] { store_->updatePipelineFriendlyName(job.pipelineId, job.friendlyName); });
    }

    void process(UpdatePipelineMetadataJob &job)
    {
        CompleteDomainRequest(
            job.completion, [this, &job]
            { store_->updatePipelineMetadata(job.pipelineId, std::move(job.metadata)); });
    }

    void process(UpdateSessionNameJob &job)
    {
        CompleteDomainRequest(
            job.completion,
            [this, &job] { store_->updateSessionFriendlyName(job.sessionId, job.friendlyName); });
    }

    void process(PipelineBinaryArtifactJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job] {
                                  store_->writePipelineArtifact(job.pipelineId, job.fileName,
                                                                job.type, job.contents);
                              });
    }

    void process(PipelineTextArtifactJob &job)
    {
        const std::vector<uint8_t> contents(job.contents.begin(), job.contents.end());
        CompleteDomainRequest(
            job.completion, [this, &job, &contents]
            { store_->writePipelineArtifact(job.pipelineId, job.fileName, job.type, contents); });
    }

    void process(DispatchBinaryArtifactJob &job)
    {
        CompleteDomainRequest(job.completion,
                              [this, &job] {
                                  store_->writeDispatchArtifact(job.dispatchId, job.fileName,
                                                                job.type, job.contents);
                              });
    }

    void process(DispatchTextArtifactJob &job)
    {
        const std::vector<uint8_t> contents(job.contents.begin(), job.contents.end());
        CompleteDomainRequest(
            job.completion, [this, &job, &contents]
            { store_->writeDispatchArtifact(job.dispatchId, job.fileName, job.type, contents); });
    }

    void process(PublishDocumentsJob &job)
    {
        store_->publishDocuments();
        job.completion.set_value(WriterResult<void>::Success());
    }

    void process(FinishCompleteJob &job)
    {
        store_->finishComplete();
        {
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Complete;
            accepting_ = false;
            terminalErrorCode_ = WriterErrorCode::None;
            hasTerminalDiagnostics_ = false;
        }
        job.completion.set_value(WriterResult<void>::Success());
    }

    void process(FinishErrorJob &job)
    {
        store_->finishError(job.message);
        {
            std::lock_guard lock(stateMutex_);
            state_ = WriterState::Error;
            accepting_ = false;
            terminalErrorCode_ = WriterErrorCode::Terminal;
            hasTerminalDiagnostics_ = false;
            try
            {
                fault::AllocationCheckpoint();
                terminalError_ = Error(WriterErrorCode::Terminal, store_->errorMessage());
                hasTerminalDiagnostics_ = true;
            }
            catch (...)
            {
            }
        }
        job.completion.set_value(WriterResult<void>::Success());
    }

    CaptureWriterOptions options_;
    BoundedQueue<Job> queue_;
    std::unique_ptr<CaptureStore> store_;

    mutable std::mutex stateMutex_;
    std::condition_variable startCondition_;
    WriterState state_{WriterState::NeverStarted};
    bool startReady_{false};
    bool startSucceeded_{false};
    bool accepting_{false};
    uint64_t acceptedJobs_{0};
    uint64_t completedJobs_{0};
    WriterErrorCode terminalErrorCode_{WriterErrorCode::None};
    WriterError terminalError_;
    bool hasTerminalDiagnostics_{false};

    std::mutex submitMutex_;
    std::mutex lifecycleMutex_;
    std::atomic<bool> automaticCompletionRequested_{false};
    std::thread thread_;
};

CaptureWriter::CaptureWriter(CaptureWriterOptions options)
    : impl_(std::make_unique<Impl>(std::move(options)))
{
}

CaptureWriter::~CaptureWriter() noexcept = default;

WriterResult<void> CaptureWriter::start()
{
    return impl_->start();
}

WriterSubmission<void> CaptureWriter::trackPipeline(
    PipelineId pipelineId, PipelineMetadata metadata,
    std::optional<uint64_t> diagnosticHandle) noexcept
{
    return SafeSubmission(
        [&]
        {
            return impl_->submit(
                Impl::TrackPipelineJob{pipelineId, std::move(metadata), diagnosticHandle, {}});
        });
}

WriterSubmission<void> CaptureWriter::trackSession(
    SessionId sessionId, PipelineId pipelineId, uint64_t flags,
    std::optional<uint64_t> diagnosticHandle) noexcept
{
    return SafeSubmission(
        [&]
        {
            return impl_->submit(
                Impl::TrackSessionJob{sessionId, pipelineId, flags, diagnosticHandle, {}});
        });
}

WriterSubmission<void> CaptureWriter::materializeDispatch(
    SessionId sessionId, DispatchReservation reservation,
    std::optional<uint64_t> diagnosticCommandBufferHandle) noexcept
{
    return SafeSubmission(
        [&]
        {
            return impl_->submit(Impl::MaterializeDispatchJob{
                sessionId, reservation, diagnosticCommandBufferHandle, {}});
        });
}

WriterSubmission<void> CaptureWriter::updateMetadata(CaptureMetadata metadata) noexcept
{
    return SafeSubmission(
        [&] { return impl_->submit(Impl::UpdateMetadataJob{std::move(metadata), {}}); });
}

WriterSubmission<void> CaptureWriter::updatePipelineFriendlyName(
    PipelineId pipelineId, std::string_view friendlyName) noexcept
{
    return SafeSubmission(
        [&] {
            return impl_->submit(
                Impl::UpdatePipelineNameJob{pipelineId, std::string(friendlyName), {}});
        });
}

WriterSubmission<void> CaptureWriter::updatePipelineMetadata(PipelineId pipelineId,
                                                             PipelineMetadata metadata) noexcept
{
    return SafeSubmission(
        [&] {
            return impl_->submit(
                Impl::UpdatePipelineMetadataJob{pipelineId, std::move(metadata), {}});
        });
}

WriterSubmission<void> CaptureWriter::updateSessionFriendlyName(
    SessionId sessionId, std::string_view friendlyName) noexcept
{
    return SafeSubmission(
        [&] {
            return impl_->submit(
                Impl::UpdateSessionNameJob{sessionId, std::string(friendlyName), {}});
        });
}

WriterSubmission<void> CaptureWriter::writePipelineBinaryArtifact(
    PipelineId pipelineId, std::string_view fileName, ArtifactType type,
    std::span<const uint8_t> contents) noexcept
{
    return SafeSubmission(
        [&]
        {
            fault::Checkpoint(fault::Point::WriterPipelineArtifactSubmission);
            return impl_->submit(Impl::PipelineBinaryArtifactJob{
                pipelineId,
                std::filesystem::path(fileName),
                type,
                std::vector<uint8_t>(contents.begin(), contents.end()),
                {}});
        });
}

WriterSubmission<void> CaptureWriter::writePipelineTextArtifact(PipelineId pipelineId,
                                                                std::string_view fileName,
                                                                ArtifactType type,
                                                                std::string_view contents) noexcept
{
    return SafeSubmission(
        [&]
        {
            fault::Checkpoint(fault::Point::WriterPipelineArtifactSubmission);
            return impl_->submit(Impl::PipelineTextArtifactJob{
                pipelineId, std::filesystem::path(fileName), type, std::string(contents), {}});
        });
}

WriterSubmission<void> CaptureWriter::writeDispatchBinaryArtifact(
    DispatchId dispatchId, std::string_view fileName, ArtifactType type,
    std::span<const uint8_t> contents) noexcept
{
    return SafeSubmission(
        [&]
        {
            fault::Checkpoint(fault::Point::WriterDispatchArtifactSubmission);
            return impl_->submit(Impl::DispatchBinaryArtifactJob{
                dispatchId,
                std::filesystem::path(fileName),
                type,
                std::vector<uint8_t>(contents.begin(), contents.end()),
                {}});
        });
}

WriterSubmission<void> CaptureWriter::writeDispatchTextArtifact(DispatchId dispatchId,
                                                                std::string_view fileName,
                                                                ArtifactType type,
                                                                std::string_view contents) noexcept
{
    return SafeSubmission(
        [&]
        {
            fault::Checkpoint(fault::Point::WriterDispatchArtifactSubmission);
            return impl_->submit(Impl::DispatchTextArtifactJob{
                dispatchId, std::filesystem::path(fileName), type, std::string(contents), {}});
        });
}

WriterSubmission<void> CaptureWriter::publishDocuments() noexcept
{
    return SafeSubmission([&] { return impl_->submit(Impl::PublishDocumentsJob{}); });
}

WriterSubmission<void> CaptureWriter::finishComplete() noexcept
{
    return SafeSubmission([&] { return impl_->submit(Impl::FinishCompleteJob{}, true); });
}

WriterSubmission<void> CaptureWriter::finishError(std::string_view message) noexcept
{
    return SafeSubmission(
        [&] { return impl_->submit(Impl::FinishErrorJob{std::string(message), {}}, true); });
}

WriterResult<void> CaptureWriter::shutdown()
{
    try
    {
        return impl_->shutdown();
    }
    catch (...)
    {
        ErrorReport report;
        CaptureException(std::current_exception(), report);
        if (report.hasDiagnostics)
        {
            try
            {
                return WriterResult<void>::Failure(std::move(report.diagnostics));
            }
            catch (...)
            {
            }
        }
        return WriterResult<void>::FailureCode(report.code);
    }
}

WriterSnapshot CaptureWriter::snapshot() const noexcept
{
    return impl_->snapshot();
}

bool CaptureWriter::commitIfAccepting(void (*callback)(void *) noexcept, void *context) noexcept
{
    return impl_->commitIfAccepting(callback, context);
}

WriterError CaptureWriter::terminalError() const
{
    return impl_->terminalError();
}

bool CaptureWriter::waitForBlockedSubmitters(std::size_t count)
{
    return impl_->waitForBlockedSubmitters(count);
}
} // namespace capture
