/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "instance.hpp"

#include "framework/utils.hpp"

#include <cassert>

/**
 * @brief The dispatch lookup for all of the created Vulkan instances.
 */
static std::unordered_map<void *, std::unique_ptr<Instance>> g_instances;

/* See header for documentation. */
const APIVersion Instance::minAPIVersion{1, 1};

/* See header for documentation. */
const std::vector<std::string> Instance::requiredDriverExtensions{
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
};

/* See header for documentation. */
const std::vector<std::pair<std::string, uint32_t>> Instance::injectedInstanceExtensions{};

/* See header for documentation. */
std::vector<std::pair<std::string, uint32_t>> Instance::injectedDeviceExtensions{};

/* See header for documentation. */
void Instance::store(VkInstance handle, std::unique_ptr<Instance> &instance)
{
    void *key = getDispatchKey(handle);
    g_instances.insert({key, std::move(instance)});
}

/* See header for documentation. */
Instance *Instance::retrieve(VkInstance handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_instances));
    return g_instances.at(key).get();
}

/* See header for documentation. */
Instance *Instance::retrieve(VkPhysicalDevice handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_instances));
    return g_instances.at(key).get();
}

/* See header for documentation. */
std::unique_ptr<Instance> Instance::destroy(VkInstance handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_instances));

    auto instance = std::move(g_instances.at(key));
    g_instances.erase(key);
    return instance;
}

/* See header for documentation. */
Instance::Instance(VkInstance _instance, PFN_vkGetInstanceProcAddr _nlayerGetProcAddress)
    : instance(_instance), nlayerGetProcAddress(_nlayerGetProcAddress)
{
    initDriverInstanceDispatchTable(instance, nlayerGetProcAddress, driver);
}

bool Instance::isCaptureOperationalLocked() const noexcept
{
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
        return true;
    }
    const auto snapshot = writer->snapshot();
    return snapshot.state == capture::WriterState::Running && snapshot.accepting;
}

bool Instance::isCaptureEnabled() const noexcept
{
    std::lock_guard<std::mutex> submissionLock{captureSubmissionMutex_};
    return isCaptureOperationalLocked();
}

bool Instance::isCaptureWriterRunning() const noexcept
{
    std::lock_guard<std::mutex> submissionLock{captureSubmissionMutex_};
    const capture::CaptureWriter *writer = nullptr;
    {
        std::lock_guard<std::mutex> stateLock{captureStateMutex_};
        if (!captureEnabled_ || captureWriter == nullptr)
        {
            return false;
        }
        writer = captureWriter.get();
    }
    const auto snapshot = writer->snapshot();
    return snapshot.state == capture::WriterState::Running && snapshot.accepting;
}

InstanceCaptureStateSnapshot Instance::captureStateSnapshot() const
{
    std::lock_guard<std::mutex> lock{captureStateMutex_};
    return {
        captureEnabled_,
        captureTerminalRequestIssued_,
        captureTerminalTransitionCount_,
        captureFailure_,
    };
}

bool Instance::transitionCaptureFailureLocked(const char *message,
                                              capture::CaptureWriter *&writerToTerminate) noexcept
{
    writerToTerminate = nullptr;
    if (!captureEnabled_)
    {
        return false;
    }

    captureEnabled_ = false;
    ++captureTerminalTransitionCount_;
    try
    {
        captureFailure_ = message == nullptr ? "capture failed" : message;
    }
    catch (...)
    {
    }

    if (!captureTerminalRequestIssued_ && captureWriter != nullptr)
    {
        captureTerminalRequestIssued_ = true;
        writerToTerminate = captureWriter.get();
    }
    return true;
}

void Instance::publishCaptureFailure(capture::CaptureWriter *writer, const char *message) noexcept
{
    const char *diagnostic = message == nullptr ? "capture failed" : message;
    if (writer != nullptr)
    {
        (void)writer->finishError(diagnostic);
    }
    LAYER_ERR("%s", diagnostic);
}

bool Instance::disableCapture(const char *message) noexcept
{
    capture::CaptureWriter *writerToTerminate = nullptr;
    bool transitioned = false;
    {
        std::lock_guard<std::mutex> submissionLock{captureSubmissionMutex_};
        std::lock_guard<std::mutex> stateLock{captureStateMutex_};
        transitioned = transitionCaptureFailureLocked(message, writerToTerminate);
    }
    if (transitioned)
    {
        publishCaptureFailure(writerToTerminate, message);
    }
    return transitioned;
}

capture::CaptureWriter *Instance::beginCaptureCompletion() noexcept
{
    std::lock_guard<std::mutex> submissionLock{captureSubmissionMutex_};
    std::lock_guard<std::mutex> stateLock{captureStateMutex_};
    if (!captureEnabled_ || captureWriter == nullptr)
    {
        return nullptr;
    }
    captureEnabled_ = false;
    return captureWriter.get();
}

void Instance::collectorJobAccepted() noexcept
{
    std::lock_guard<std::mutex> lock{collectorJobsMutex_};
    ++collectorJobs_;
}

void Instance::collectorJobFinished() noexcept
{
    std::lock_guard<std::mutex> lock{collectorJobsMutex_};
    if (collectorJobs_ != 0)
    {
        --collectorJobs_;
    }
    if (collectorJobs_ == 0)
    {
        collectorJobsDrained_.notify_all();
    }
}

void Instance::waitForCollectorJobs() noexcept
{
    std::unique_lock<std::mutex> lock{collectorJobsMutex_};
    collectorJobsDrained_.wait(lock, [this] { return collectorJobs_ == 0; });
}

uint64_t Instance::collectorJobCount() const noexcept
{
    std::lock_guard<std::mutex> lock{collectorJobsMutex_};
    return collectorJobs_;
}
