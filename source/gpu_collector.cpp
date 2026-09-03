/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "gpu_collector.hpp"

#include "fault_injection.hpp"
#include "instance.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>
#include <stdexcept>

extern std::mutex g_vulkanLock;

namespace capture
{
namespace
{
#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
std::atomic<CollectorObserver> observer{nullptr};
void Notify(CollectorOperation operation) noexcept
{
    if (const auto callback = observer.load(std::memory_order_relaxed))
    {
        callback(operation);
    }
}
#else
void Notify(CollectorOperation) noexcept {}
#endif

const char *StatisticsFileName(uint32_t mode) noexcept
{
    return mode == 0 ? "statistics_mode0.bin" : "statistics_mode1.bin";
}
} // namespace

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
void SetCollectorObserver(CollectorObserver callback) noexcept
{
    observer.store(callback, std::memory_order_relaxed);
}
#endif

GpuCollector::Reservation::~Reservation() noexcept
{
    cancel();
}

GpuCollector::Reservation::Reservation(Reservation &&other) noexcept : owner_(other.owner_)
{
    other.owner_ = nullptr;
}

GpuCollector::Reservation &GpuCollector::Reservation::operator=(Reservation &&other) noexcept
{
    if (this != &other)
    {
        cancel();
        owner_ = other.owner_;
        other.owner_ = nullptr;
    }
    return *this;
}

bool GpuCollector::Reservation::commit(std::unique_ptr<GpuCollectorJob> job) noexcept
{
    if (owner_ == nullptr || job == nullptr)
    {
        return false;
    }
    GpuCollector *owner = owner_;
    owner_ = nullptr;
    return owner->commitReserved(std::move(job));
}

void GpuCollector::Reservation::cancel() noexcept
{
    if (owner_ != nullptr)
    {
        GpuCollector *owner = owner_;
        owner_ = nullptr;
        owner->cancelReserved();
    }
}

GpuCollector::GpuCollector(Instance &instance, VkDevice device, const DeviceDispatchTable &dispatch,
                           GpuCollectorOptions options)
    : instance_(instance), device_(device), dispatch_(dispatch), capacity_(options.capacity),
      pollInterval_(options.pollInterval)
{
    if (capacity_ == 0 || pollInterval_.count() < 0)
    {
        throw std::invalid_argument("invalid GPU collector options");
    }
    incoming_.reserve(capacity_);
    pending_.reserve(capacity_);
    worker_ = std::thread([this] { run(); });
}

GpuCollector::~GpuCollector() noexcept
{
    shutdown();
}

std::optional<GpuCollector::Reservation> GpuCollector::tryReserve(CapacityState &state) noexcept
{
    std::lock_guard lock(mutex_);
    state = {capacityGeneration_, accepting_};
    if (!accepting_ || outstanding_ >= capacity_)
    {
        return std::nullopt;
    }
    ++outstanding_;
    instance_.collectorJobAccepted();
    return Reservation(this);
}

bool GpuCollector::waitForCapacityChange(CapacityState state) noexcept
{
    std::unique_lock lock(mutex_);
    if (!accepting_ || capacityGeneration_ != state.generation)
    {
        return accepting_;
    }

    ++blockedProducers_;
    blockedChanged_.notify_all();
    capacityChanged_.wait(lock, [this, state]
                          { return !accepting_ || capacityGeneration_ != state.generation; });
    --blockedProducers_;
    blockedChanged_.notify_all();
    return accepting_;
}

bool GpuCollector::commitReserved(std::unique_ptr<GpuCollectorJob> job) noexcept
{
    bool accepted = true;
    try
    {
        fault::Checkpoint(fault::Point::CollectorQueueAdmission);
    }
    catch (...)
    {
        accepted = false;
        job->admissionFailed = true;
    }

    {
        std::lock_guard lock(mutex_);
        // tryReserve() accounts for every committed entry and both vectors reserve
        // the full capacity at construction, so this transfer cannot allocate.
        incoming_.push_back(std::move(job));
    }
    changed_.notify_one();
    return accepted;
}

void GpuCollector::cancelReserved() noexcept
{
    instance_.collectorJobFinished();
    {
        std::lock_guard lock(mutex_);
        if (outstanding_ != 0)
        {
            --outstanding_;
            ++capacityGeneration_;
        }
    }
    capacityChanged_.notify_all();
    changed_.notify_all();
}

void GpuCollector::closeAdmission() noexcept
{
    {
        std::lock_guard lock(mutex_);
        if (accepting_)
        {
            accepting_ = false;
            ++capacityGeneration_;
        }
        shutdownRequested_ = true;
    }
    capacityChanged_.notify_all();
    changed_.notify_all();
    blockedChanged_.notify_all();
}

void GpuCollector::shutdown() noexcept
{
    closeAdmission();
    if (worker_.joinable())
    {
        try
        {
            worker_.join();
        }
        catch (...)
        {
        }
    }
}

bool GpuCollector::waitForBlockedProducers(std::size_t count, std::chrono::milliseconds timeout)
{
    std::unique_lock lock(mutex_);
    (void)blockedChanged_.wait_for(lock, timeout, [this, count]
                                   { return !accepting_ || blockedProducers_ >= count; });
    return blockedProducers_ >= count;
}

std::size_t GpuCollector::pendingCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return outstanding_;
}

void GpuCollector::moveIncomingLocked() noexcept
{
    for (auto &job : incoming_)
    {
        pending_.push_back(std::move(job));
    }
    incoming_.clear();
}

void GpuCollector::run() noexcept
{
    try
    {
        bool idleWaitAttempted = false;
        for (;;)
        {
            {
                std::unique_lock lock(mutex_);
                changed_.wait_for(lock, pollInterval_,
                                  [this] { return shutdownRequested_ || !incoming_.empty(); });
                moveIncomingLocked();
                if (shutdownRequested_ && !idleWaitAttempted && !pending_.empty())
                {
                    idleWaitAttempted = true;
                    lock.unlock();
                    Notify(CollectorOperation::DeviceWaitIdle);
                    const VkResult idle = dispatch_.vkDeviceWaitIdle != nullptr
                                              ? dispatch_.vkDeviceWaitIdle(device_)
                                              : VK_ERROR_UNKNOWN;
                    if (idle != VK_SUCCESS)
                    {
                        std::vector<std::unique_ptr<GpuCollectorJob>> failed;
                        lock.lock();
                        moveIncomingLocked();
                        failed.swap(pending_);
                        lock.unlock();
                        for (auto &job : failed)
                        {
                            failJob(std::move(job),
                                    "GPU collector device drain failed during shutdown");
                        }
                    }
                    continue;
                }
                if (shutdownRequested_ && outstanding_ == 0)
                {
                    shutdownComplete_ = true;
                    changed_.notify_all();
                    return;
                }
            }

            bool madeProgress = false;
            for (std::size_t i = 0; i < pending_.size();)
            {
                auto &job = pending_[i];
                Notify(CollectorOperation::FenceStatus);
                VkResult status = VK_ERROR_UNKNOWN;
                try
                {
                    fault::Checkpoint(fault::Point::CollectorPollingBookkeeping);
                    status = dispatch_.vkGetFenceStatus != nullptr
                                 ? dispatch_.vkGetFenceStatus(device_, job->fence)
                                 : VK_ERROR_UNKNOWN;
                }
                catch (...)
                {
                    status = VK_ERROR_UNKNOWN;
                }
                if (status == VK_NOT_READY)
                {
                    ++i;
                    continue;
                }

                std::unique_ptr<GpuCollectorJob> ready = std::move(job);
                pending_[i] = std::move(pending_.back());
                pending_.pop_back();
                madeProgress = true;
                if (status == VK_SUCCESS)
                {
                    processReadyJob(std::move(ready));
                }
                else
                {
                    failJob(std::move(ready),
                            status == VK_ERROR_DEVICE_LOST
                                ? "GPU collector observed device loss while polling marker fence"
                                : "GPU collector fence-status polling failed");
                }
            }

            if (!madeProgress)
            {
                std::unique_lock lock(mutex_);
                changed_.wait_for(lock, pollInterval_,
                                  [this] { return shutdownRequested_ || !incoming_.empty(); });
            }
        }
    }
    catch (...)
    {
        completeOutstanding();
    }
}

VkResult GpuCollector::copySnapshot(const StatsSnapshot &snapshot,
                                    std::vector<uint8_t> &bytes) noexcept
{
    try
    {
        fault::Checkpoint(fault::Point::CollectorCompletionCopy);
        bytes.resize(static_cast<std::size_t>(snapshot.dataSize));
        void *mapped = nullptr;
        Notify(CollectorOperation::MapMemory);
        VkResult result =
            dispatch_.vkMapMemory(device_, snapshot.memory, 0, snapshot.allocationSize, 0, &mapped);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        if ((snapshot.memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            const VkMappedMemoryRange range{
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, snapshot.memory, 0, VK_WHOLE_SIZE,
            };
            Notify(CollectorOperation::InvalidateMemory);
            result = dispatch_.vkInvalidateMappedMemoryRanges(device_, 1, &range);
            if (result != VK_SUCCESS)
            {
                Notify(CollectorOperation::UnmapMemory);
                dispatch_.vkUnmapMemory(device_, snapshot.memory);
                return result;
            }
        }
        Notify(CollectorOperation::CopyMemory);
        if (!bytes.empty())
        {
            std::memcpy(bytes.data(), mapped, bytes.size());
        }
        Notify(CollectorOperation::UnmapMemory);
        dispatch_.vkUnmapMemory(device_, snapshot.memory);
        return VK_SUCCESS;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

void GpuCollector::processReadyJob(std::unique_ptr<GpuCollectorJob> job) noexcept
{
    const char *failure = nullptr;
    if (job->admissionFailed)
    {
        failure = "GPU collector admission failed after marker submission";
    }
    else if (!instance_.isCaptureEnabled())
    {
        failure = "capture was disabled before GPU collector completion";
    }
    else
    {
        for (const auto &snapshot : job->plan.selectedSnapshots)
        {
            std::vector<uint8_t> bytes;
            const VkResult copy = copySnapshot(*snapshot, bytes);
            if (copy != VK_SUCCESS)
            {
                failure = copy == VK_ERROR_OUT_OF_HOST_MEMORY
                              ? "GPU collector host-copy allocation failed"
                              : "GPU collector mapped-memory invalidation/copy failed";
                break;
            }
            if (!snapshot->dispatchId.has_value())
            {
                continue;
            }
            const auto status = instance_.submitCaptureJob(
                [&](CaptureWriter &writer)
                {
                    return writer.writeDispatchBinaryArtifact(
                        *snapshot->dispatchId, StatisticsFileName(job->statisticsMode),
                        ArtifactType::DispatchStatistics, bytes);
                },
                "raw-statistics writer job was rejected by the GPU collector");
            if (status != CaptureJobStatus::Accepted)
            {
                failure = "raw-statistics writer job was rejected by the GPU collector";
                break;
            }
        }
    }
    destroyFence(job->fence);
    job->fence = VK_NULL_HANDLE;
    finalizeJob(std::move(job), failure);
}

void GpuCollector::failJob(std::unique_ptr<GpuCollectorJob> job, const char *message) noexcept
{
    destroyFence(job->fence);
    job->fence = VK_NULL_HANDLE;
    finalizeJob(std::move(job), message);
}

void GpuCollector::finalizeJob(std::unique_ptr<GpuCollectorJob> job, const char *failure) noexcept
{
    {
        std::unique_lock admission(instance_.submissionAdmissionMutex);
        {
            std::lock_guard vulkan(g_vulkanLock);
            submission::finish(job->plan, failure);
        }
        if (failure != nullptr)
        {
            (void)instance_.disableCapture(failure);
        }
    }
    job.reset();
    instance_.collectorJobFinished();
    {
        std::lock_guard lock(mutex_);
        if (outstanding_ != 0)
        {
            --outstanding_;
            ++capacityGeneration_;
        }
    }
    capacityChanged_.notify_all();
    changed_.notify_all();
}

void GpuCollector::destroyFence(VkFence fence) noexcept
{
    if (fence != VK_NULL_HANDLE)
    {
        Notify(CollectorOperation::DestroyFence);
        dispatch_.vkDestroyFence(device_, fence, nullptr);
    }
}

void GpuCollector::completeOutstanding() noexcept
{
    std::vector<std::unique_ptr<GpuCollectorJob>> failed;
    {
        std::lock_guard lock(mutex_);
        moveIncomingLocked();
        failed.swap(pending_);
    }
    for (auto &job : failed)
    {
        failJob(std::move(job), "unexpected GPU collector worker failure");
    }
}
} // namespace capture
