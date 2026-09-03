/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * Declares the root class for layer management of VkInstance objects.
 *
 * Role summary
 * ============
 *
 * Instances represent the core context used by the application to connect to
 * the OS graphics subsystem prior to connection to a specific device instance.
 * An instance object is the dispatch root for the Vulkan subsystem, so
 * instance commands all take some form of dispatchable handle that can be
 * resolved into a unique per-instance key. For the driver this key would
 * simply be a pointer directly to the driver-internal instance object, but for
 * our layer we use a instance dispatch key as an index in to the map to find
 * the layer's instance object.
 *
 * Key properties
 * ==============
 *
 * Vulkan instances are designed to be used concurrently by multiple
 * application threads. An application can have multiple concurrent instances,
 * and use each instance from multiple threads.
 *
 * Access to application-facing dispatch maps and tracked Vulkan state is
 * generally protected by the layer-wide Vulkan lock. Application-facing
 * hooks normally release it around downstream driver calls. Capture
 * admission/lifecycle state uses the narrower mutexes documented below. Writer
 * admission and backpressure can still run while a caller holds the broad lock;
 * collector and writer threads themselves do not take it.
 */

#pragma once

#include "capture_writer.hpp"
#include "framework/instance_dispatch_table.hpp"
#include "layer_options.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

/**
 * @brief This class implements the layer state tracker for a single instance.
 */
enum class CaptureJobStatus
{
    Skipped,
    Accepted,
    Rejected,
};

struct InstanceCaptureStateSnapshot
{
    bool enabled{true};
    bool terminalRequestIssued{false};
    uint64_t terminalTransitionCount{0};
    std::string failure;
};

class Instance
{
  public:
    /**
     * @brief Store a new instance into the global store of dispatchable instances.
     *
     * @param handle     The dispatchable instance handle to use as an indirect key.
     * @param instance   The @c Instance object to store.
     */
    static void store(VkInstance handle, std::unique_ptr<Instance> &instance);

    /**
     * @brief Fetch an instance from the global store of dispatchable instances.
     *
     * @param handle   The dispatchable instance handle to use as an indirect lookup.
     *
     * @return The layer instance context.
     */
    static Instance *retrieve(VkInstance handle);

    /**
     * @brief Fetch an instance from the global store of dispatchable instances.
     *
     * @param handle   The dispatchable physical device handle to use as an indirect lookup.
     *
     * @return The layer instance context.
     */
    static Instance *retrieve(VkPhysicalDevice handle);

    /**
     * @brief Drop an instance from the global store of dispatchable instances.
     *
     * This must be called before the driver VkInstance has been destroyed, as
     * we deference the native instance handle to get the dispatch key.
     *
     * @param handle   The dispatchable instance handle to use as an indirect lookup.
     *
     * @return Returns the ownership of the Instance object to the caller.
     */
    static std::unique_ptr<Instance> destroy(VkInstance handle);

    /**
     * @brief Create a new layer instance object.
     *
     * @param instance               The instance handle this instance is created with.
     * @param nlayerGetProcAddress   The vkGetProcAddress function in the driver/next layer down.
     */
    Instance(VkInstance instance, PFN_vkGetInstanceProcAddr nlayerGetProcAddress);

    /**
     * @brief Return whether capture is operational for new layer work.
     *
     * This requires both the raw capture-enabled flag and, when a writer is
     * installed, a writer that is Running and accepting jobs. Use
     * captureStateSnapshot() when the raw lifecycle flag and terminal
     * diagnostic are required.
     */
    bool isCaptureEnabled() const noexcept;
    bool isCaptureWriterRunning() const noexcept;
    /** Return the raw capture lifecycle state and its diagnostic counters. */
    InstanceCaptureStateSnapshot captureStateSnapshot() const;
    bool disableCapture(const char *message) noexcept;
    capture::CaptureWriter *beginCaptureCompletion() noexcept;
    void collectorJobAccepted() noexcept;
    void collectorJobFinished() noexcept;
    void waitForCollectorJobs() noexcept;
    uint64_t collectorJobCount() const noexcept;

    // Capture-state synchronization policy and lock order:
    //   submissionAdmissionMutex -> g_vulkanLock -> collector reservation mutex
    //   or submissionAdmissionMutex -> g_vulkanLock -> captureSubmissionMutex_
    //   -> captureStateMutex_. Capacity waits hold none of these locks. Writer
    // submission/backpressure runs while holding captureSubmissionMutex_ but
    // never captureStateMutex_. Callers may omit earlier locks, but must never
    // acquire them after entering these APIs. The writer thread takes neither
    // instance mutex, and terminal publication runs after both are released, so
    // the state mutex never spans filesystem work.
    template <typename Committer> bool commitCaptureIfEnabled(Committer &&committer) noexcept
    {
        static_assert(std::is_nothrow_invocable_v<Committer &>,
                      "capture identity commit callback must be noexcept and lvalue-invocable");

        // Instance-driven terminal transitions take captureSubmissionMutex_.
        // The writer gate additionally serializes this commit against terminal
        // state published asynchronously by the writer thread.
        std::unique_lock<std::mutex> submissionLock{captureSubmissionMutex_};
        capture::CaptureWriter *writer = nullptr;
        {
            std::lock_guard<std::mutex> stateLock{captureStateMutex_};
            if (!captureEnabled_)
            {
                return false;
            }
            writer = captureWriter.get();
        }
        if (writer == nullptr)
        {
            committer();
            return true;
        }

        using CommitterType = std::remove_reference_t<Committer>;
        return writer->commitIfAccepting(
            [](void *context) noexcept
            {
                auto *callback = static_cast<CommitterType *>(context);
                (*callback)();
            },
            const_cast<void *>(static_cast<const void *>(std::addressof(committer))));
    }

    template <typename Submitter>
    CaptureJobStatus submitCaptureJob(Submitter &&submitter, const char *failureMessage) noexcept
    {
        capture::CaptureWriter *writer = nullptr;
        capture::CaptureWriter *writerToTerminate = nullptr;
        bool transitioned = false;
        std::unique_lock<std::mutex> submissionLock{captureSubmissionMutex_};
        {
            std::lock_guard<std::mutex> stateLock{captureStateMutex_};
            if (!captureEnabled_ || captureWriter == nullptr)
            {
                return CaptureJobStatus::Skipped;
            }
            writer = captureWriter.get();
        }

        const auto writerSnapshot = writer->snapshot();
        if (writerSnapshot.state != capture::WriterState::Running || !writerSnapshot.accepting)
        {
            {
                std::lock_guard<std::mutex> stateLock{captureStateMutex_};
                transitioned = transitionCaptureFailureLocked(failureMessage, writerToTerminate);
            }
            submissionLock.unlock();
            if (transitioned)
            {
                publishCaptureFailure(writerToTerminate, failureMessage);
            }
            return CaptureJobStatus::Skipped;
        }

        bool accepted = false;
        try
        {
            accepted = submitter(*writer).accepted;
        }
        catch (...)
        {
        }
        if (accepted)
        {
            return CaptureJobStatus::Accepted;
        }
        const auto rejectionSnapshot = writer->snapshot();
        const bool writerStillAccepting =
            rejectionSnapshot.state == capture::WriterState::Running && rejectionSnapshot.accepting;
        {
            std::lock_guard<std::mutex> stateLock{captureStateMutex_};
            transitioned = transitionCaptureFailureLocked(failureMessage, writerToTerminate);
        }
        submissionLock.unlock();
        if (transitioned)
        {
            publishCaptureFailure(writerToTerminate, failureMessage);
        }
        // A writer that became terminal independently did not reject this
        // object's publication transaction. Report Skipped so downstream
        // object success is preserved; reserve Rejected for a failed current
        // submission while the writer was still operational.
        return writerStillAccepting ? CaptureJobStatus::Rejected : CaptureJobStatus::Skipped;
    }

  public:
    /**
     * @brief The instance handle this instance is created with.
     */
    VkInstance instance;

    /**
     * @brief The next layer's @c vkGetInstanceProcAddr() function pointer.
     */
    PFN_vkGetInstanceProcAddr nlayerGetProcAddress;

    /**
     * @brief The driver function dispatch table.
     */
    InstanceDispatchTable driver{};

    /**
     * @brief The minimum API version needed by this layer.
     */
    static const APIVersion minAPIVersion;

    /**
     * @brief Required instance extensions from the driver.
     *
     * The layer will attempt to enable these even if the application does not.
     */
    static const std::vector<std::string> requiredDriverExtensions;

    /**
     * @brief Additional instance extensions injected by the layer.
     *
     * The layer will expose these even if the driver does not.
     */
    static const std::vector<std::pair<std::string, uint32_t>> injectedInstanceExtensions;

    /**
     * @brief Additional device extensions injected by the layer.
     *
     * The layer will expose these even if the driver does not. Items are
     * removed from the list if the driver already exposes the extension.
     */
    static std::vector<std::pair<std::string, uint32_t>> injectedDeviceExtensions;

    /**
     * @brief User layer settings
     */
    std::unique_ptr<const LayerOptions> layerOptions;

    /** One structured capture lifecycle is owned by the instance. */
    std::unique_ptr<capture::CaptureWriter> captureWriter;
    capture::CaptureMetadata captureMetadata;
    DispatchFilter dispatchFilter{DispatchFilter::All()};
    capture::IdAllocator<capture::LogicalDeviceId> logicalDeviceIds;
    capture::IdAllocator<capture::ShaderModuleId> shaderModuleIds;
    capture::IdAllocator<capture::PipelineId> pipelineIds;
    capture::IdAllocator<capture::SessionId> sessionIds;
    capture::IdAllocator<capture::DispatchId> dispatchIds;
    // Serializes submission admission, downstream forwarding, and the
    // success/failure commit. This gives accepted submissions a deterministic
    // order without holding the broad Vulkan state lock across the driver call.
    std::mutex submissionAdmissionMutex;

  private:
    // Requires captureSubmissionMutex_. Writerless instances remain
    // operational for unit fixtures; an existing writer must be Running.
    bool isCaptureOperationalLocked() const noexcept;
    bool transitionCaptureFailureLocked(const char *message,
                                        capture::CaptureWriter *&writerToTerminate) noexcept;
    static void publishCaptureFailure(capture::CaptureWriter *writer, const char *message) noexcept;

    mutable std::mutex captureSubmissionMutex_;
    mutable std::mutex captureStateMutex_;
    mutable std::mutex collectorJobsMutex_;
    std::condition_variable collectorJobsDrained_;
    uint64_t collectorJobs_{0};
    bool captureEnabled_{true};
    bool captureTerminalRequestIssued_{false};
    uint64_t captureTerminalTransitionCount_{0};
    std::string captureFailure_;
};
