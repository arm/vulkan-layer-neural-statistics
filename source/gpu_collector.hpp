/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "framework/device_dispatch_table.hpp"
#include "submission_planner.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

class Instance;

namespace capture
{
enum class CollectorOperation
{
    FenceStatus,
    DeviceWaitIdle,
    MapMemory,
    InvalidateMemory,
    CopyMemory,
    UnmapMemory,
    DestroyFence,
};

using CollectorObserver = void (*)(CollectorOperation) noexcept;

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
void SetCollectorObserver(CollectorObserver observer) noexcept;
#else
inline void SetCollectorObserver(CollectorObserver) noexcept {}
#endif

#if defined(NEURAL_STATISTICS_COLLECTOR_CAPACITY)
inline constexpr std::size_t kDefaultCollectorCapacity = NEURAL_STATISTICS_COLLECTOR_CAPACITY;
#else
inline constexpr std::size_t kDefaultCollectorCapacity = 64;
#endif

static_assert(kDefaultCollectorCapacity > 0, "collector capacity must be non-zero");

struct GpuCollectorOptions
{
    std::size_t capacity{kDefaultCollectorCapacity};
    std::chrono::milliseconds pollInterval{1};
};

struct GpuCollectorJob
{
    submission::Plan plan;
    std::shared_ptr<SubmissionArchive> archive;
    VkFence fence{VK_NULL_HANDLE};
    uint32_t statisticsMode{0};
    bool admissionFailed{false};
};

class GpuCollector
{
  public:
    class Reservation
    {
      public:
        Reservation() noexcept = default;
        ~Reservation() noexcept;
        Reservation(Reservation &&other) noexcept;
        Reservation &operator=(Reservation &&other) noexcept;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;

        // A reserved slot makes ownership transfer lossless. A false return
        // reports an injected/admission failure, but the collector still owns
        // and retires the marker fence and snapshots after GPU completion.
        bool commit(std::unique_ptr<GpuCollectorJob> job) noexcept;
        void cancel() noexcept;
        explicit operator bool() const noexcept
        {
            return owner_ != nullptr;
        }

      private:
        friend class GpuCollector;
        explicit Reservation(GpuCollector *owner) noexcept : owner_(owner) {}
        GpuCollector *owner_{nullptr};
    };

    GpuCollector(Instance &instance, VkDevice device, const DeviceDispatchTable &dispatch,
                 GpuCollectorOptions options = {});
    ~GpuCollector() noexcept;

    GpuCollector(const GpuCollector &) = delete;
    GpuCollector &operator=(const GpuCollector &) = delete;

    struct CapacityState
    {
        uint64_t generation{0};
        bool accepting{false};
    };

    // Admission ordering is determined by submissionAdmissionMutex. Capacity
    // waiters do not own a slot and may race after a wake; every loser rebuilds
    // its tentative submission plan before trying again.
    std::optional<Reservation> tryReserve(CapacityState &state) noexcept;
    bool waitForCapacityChange(CapacityState state) noexcept;
    void closeAdmission() noexcept;
    void shutdown() noexcept;

    bool waitForBlockedProducers(std::size_t count,
                                 std::chrono::milliseconds timeout = std::chrono::seconds(2));
    std::size_t pendingCount() const noexcept;

  private:
    friend class Reservation;
    bool commitReserved(std::unique_ptr<GpuCollectorJob> job) noexcept;
    void cancelReserved() noexcept;
    void run() noexcept;
    void moveIncomingLocked() noexcept;
    void processReadyJob(std::unique_ptr<GpuCollectorJob> job) noexcept;
    void failJob(std::unique_ptr<GpuCollectorJob> job, const char *message) noexcept;
    void finalizeJob(std::unique_ptr<GpuCollectorJob> job, const char *failure) noexcept;
    VkResult copySnapshot(const StatsSnapshot &snapshot, std::vector<uint8_t> &bytes) noexcept;
    void destroyFence(VkFence fence) noexcept;
    void completeOutstanding() noexcept;

    Instance &instance_;
    const VkDevice device_;
    const DeviceDispatchTable dispatch_;
    const std::size_t capacity_;
    const std::chrono::milliseconds pollInterval_;

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::condition_variable capacityChanged_;
    std::condition_variable blockedChanged_;
    std::vector<std::unique_ptr<GpuCollectorJob>> incoming_;
    std::vector<std::unique_ptr<GpuCollectorJob>> pending_;
    std::size_t outstanding_{0};
    uint64_t capacityGeneration_{0};
    std::size_t blockedProducers_{0};
    bool accepting_{true};
    bool shutdownRequested_{false};
    bool shutdownComplete_{false};
    std::thread worker_;
};
} // namespace capture
