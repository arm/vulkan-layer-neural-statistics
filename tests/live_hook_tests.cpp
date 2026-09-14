/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_filesystem.hpp"
#include "device.hpp"
#include "fault_injection.hpp"
#include "framework/manual_functions.hpp"
#include "framework/utils.hpp"
#include "instance.hpp"
#include "layer_device_functions.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <unistd.h>

std::mutex g_vulkanLock;

namespace
{
struct Dispatchable
{
    void *key;
};

struct BufferCreateObservation
{
    VkBufferUsageFlags usage{0};
    VkSharingMode sharingMode{VK_SHARING_MODE_EXCLUSIVE};
    std::vector<uint32_t> queueFamilyIndices;
};

struct DriverState
{
    VkResult shaderResult{VK_SUCCESS};
    bool throwShaderCreate{false};
    uint32_t shaderCreates{0};
    uint32_t shaderDestroys{0};

    VkResult pipelineResult{VK_SUCCESS};
    std::vector<uintptr_t> pipelineOutputs;
    uint32_t pipelineCreates{0};
    uint32_t pipelineDestroys{0};
    bool exposeStaticPipelineProperties{false};
    uint32_t pipelinePropertyQueries{0};
    std::vector<VkPipeline> lastPipelineHandles;
    bool lastPipelineCreateHadPNext {false};
    bool lastPipelineCreateHadNeuralStatistics {false};
    bool lastPipelineCreateInfoObserved {false};
    VkDataGraphPipelineCreateInfoARM lastPipelineCreateInfo {};

    VkResult instrumentedSessionResult{VK_SUCCESS};
    VkResult originalSessionResult{VK_SUCCESS};
    uint32_t instrumentedSessionCreates{0};
    uint32_t originalSessionCreates{0};
    uint32_t sessionDestroys{0};
    uint32_t bufferCreates{0};
    VkResult bufferCreateResult {VK_SUCCESS};
    uint32_t bufferDestroys{0};
    std::vector<BufferCreateObservation> bufferCreateObservations;
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;
    uint32_t memoryAllocations{0};
    uint32_t memoryFrees{0};
    uint32_t requirements2CoreCalls{0};
    uint32_t requirements2KhrCalls{0};
    bool exposeRequirements2Core{true};
    bool exposeRequirements2Khr{true};
    VkMemoryPropertyFlags memoryTypeProperties{
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT};

    VkResult applicationSubmitResult{VK_SUCCESS};
    VkResult markerSubmitResult{VK_SUCCESS};
    VkResult createFenceResult{VK_SUCCESS};
    VkResult deviceWaitIdleResult{VK_SUCCESS};
    VkResult mapMemoryResult{VK_SUCCESS};
    VkResult invalidateMemoryResult{VK_SUCCESS};
    uint32_t deviceWaitIdleCalls{0};
    uint32_t fenceCreates{0};
    uint32_t fenceDestroys{0};
    uint32_t duplicateFenceDestroys{0};
    uint64_t collectorJobsAtDeviceDestroy{0};
    std::size_t liveFencesAtDeviceDestroy{0};
    uint32_t mapMemoryCalls{0};
    uint32_t invalidateMemoryCalls{0};
    uint32_t cmdExecuteCommandsCalls{0};
    uint32_t cmdDispatchDataGraphCalls {0};
    uint32_t cmdFillBufferCalls {0};
    uint32_t cmdPipelineBarrierCalls {0};
    std::vector<std::vector<VkCommandBuffer>> cmdExecuteCommandBuffers;
    uint32_t cmdCopyBufferCalls{0};
    uint32_t commandBufferFrees {0};
    uint32_t legacyApplicationSubmits{0};
    uint32_t markerSubmits{0};
    std::vector<VkCommandBuffer> lastLegacyCommandBuffers;
    std::vector<uint32_t> lastLegacyCommandBufferDeviceMasks;
    std::vector<uint32_t> lastLegacyWaitSemaphoreDeviceIndices;
    std::vector<uint32_t> lastLegacySignalSemaphoreDeviceIndices;
    bool lastLegacyDeviceGroupHadProtectedPrefix{false};
    uint32_t coreSubmit2Calls{0};
    uint32_t khrSubmit2Calls{0};
    uint32_t coreMarkerSubmit2Calls{0};
    uint32_t khrMarkerSubmit2Calls{0};
    std::vector<VkCommandBuffer> lastSubmit2CommandBuffers;
    bool exposeCoreSubmit2{true};
    bool exposeKhrSubmit2{true};

    VkResult instanceCreateResult{VK_SUCCESS};
    VkResult deviceCreateResult{VK_SUCCESS};
    uint32_t instanceCreates{0};
    VkInstance* applicationInstanceOutput{nullptr};
    uint32_t applicationInstanceCreates{0};
    uint32_t auxiliaryInstanceCreates{0};
    uint32_t deviceCreates{0};
    uint32_t deviceExtensionQueries {0};
    uint32_t physicalDeviceFeatureQueries {0};
    uint32_t physicalDevicePropertyQueries {0};
    uint32_t queueFamilyPropertyQueries {0};
    uint32_t lastDeviceCreateEnabledExtensionCount {0};
    bool synchronization2FeatureEnabled {false};
    bool writerRootObservedBeforeEveryInstanceCreate{true};
    bool lastInstanceCreateHadApplicationInfo {false};
    uint32_t lastInstanceCreateApiVersion {0};
    bool requirements2ExtensionEnabled{false};
    bool exposeNeuralStatisticsExtension {true};
    bool neuralStatisticsFeatureSupported {true};
    uint32_t neuralStatisticsFeatureQueries {0};
    bool neuralStatisticsExtensionEnabled {false};
    bool neuralStatisticsFeatureEnabled {false};
    uint32_t deviceDestroys{0};
    bool probeDeviceDestroyLock{false};
    bool deviceDestroyLockAvailable{false};
    uint32_t instanceDestroys{0};
};

DriverState driver;
int failures = 0;
std::atomic<uintptr_t> nextHandle{0x10000};
std::atomic<bool> concurrentCollectionMode{false};
std::atomic<uint32_t> concurrentApplicationSubmits{0};
std::atomic<uint32_t> concurrentMarkerSubmits{0};
std::mutex fakeFenceMutex;
std::unordered_map<uint64_t, VkResult> fakeFenceStatuses;
std::vector<VkFence> submittedMarkerFences;
bool autoSignalMarkerFences{true};
std::atomic<size_t> observerProducer0{0};
std::atomic<size_t> observerProducer1{0};
std::atomic<size_t> observerWriter{0};
std::atomic<bool> observerRanOnProducer{false};
std::atomic<bool> observerSawMultipleWorkers{false};
std::atomic<size_t> observerCollector{0};
std::atomic<bool> observerCollectorRanOnProducer{false};
std::atomic<uint32_t> observerCollectorCopies{0};
std::atomic<bool> pauseDeviceCreate{false};
std::atomic<bool> deviceCreateEntered{false};
std::atomic<bool> releaseDeviceCreate{false};
std::atomic<bool> pauseBufferCreate {false};
std::atomic<bool> bufferCreateEntered {false};
std::atomic<bool> releaseBufferCreate {false};
std::atomic<bool> pauseApplicationSubmit {false};
std::atomic<bool> applicationSubmitEntered {false};
std::atomic<bool> releaseApplicationSubmit {false};
Instance* terminalizeCaptureDuringPipelineCreate {nullptr};
Instance* terminalizeCaptureDuringSessionCreate {nullptr};
std::filesystem::path expectedInstanceRoot;

size_t CurrentThreadHash() noexcept
{
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

void ArmBufferCreatePause() noexcept
{
    releaseBufferCreate.store(false, std::memory_order_relaxed);
    bufferCreateEntered.store(false, std::memory_order_relaxed);
    pauseBufferCreate.store(true, std::memory_order_release);
}

bool ReleaseBufferCreatePause(Instance* terminalizeCapture)
{
    const auto enterDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!bufferCreateEntered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < enterDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool entered = bufferCreateEntered.load(std::memory_order_acquire);

    bool lockAvailable = false;
    if (entered)
    {
        const auto lockDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        do
        {
            if (g_vulkanLock.try_lock())
            {
                lockAvailable = true;
                g_vulkanLock.unlock();
                break;
            }
            std::this_thread::yield();
        }
        while (std::chrono::steady_clock::now() < lockDeadline);
    }

    if (terminalizeCapture != nullptr)
    {
        (void) terminalizeCapture->disableCapture("capture terminalized while a snapshot allocation was paused");
    }
    releaseBufferCreate.store(true, std::memory_order_release);
    releaseBufferCreate.notify_all();
    pauseBufferCreate.store(false, std::memory_order_release);
    return entered && lockAvailable;
}

bool FinishBufferCreatePause(std::thread& producer, Instance* terminalizeCapture)
{
    const bool releasedUnlocked = ReleaseBufferCreatePause(terminalizeCapture);
    producer.join();
    return releasedUnlocked;
}

void ArmApplicationSubmitPause() noexcept
{
    releaseApplicationSubmit.store(false, std::memory_order_relaxed);
    applicationSubmitEntered.store(false, std::memory_order_relaxed);
    pauseApplicationSubmit.store(true, std::memory_order_release);
}

void PauseApplicationSubmitIfRequested() noexcept
{
    if (!pauseApplicationSubmit.load(std::memory_order_acquire))
    {
        return;
    }
    applicationSubmitEntered.store(true, std::memory_order_release);
    applicationSubmitEntered.notify_all();
    while (!releaseApplicationSubmit.load(std::memory_order_acquire))
    {
        releaseApplicationSubmit.wait(false, std::memory_order_relaxed);
    }
}

bool FinishApplicationSubmitPause(std::thread& producer, Instance* instance)
{
    const auto enterDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!applicationSubmitEntered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < enterDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool entered = applicationSubmitEntered.load(std::memory_order_acquire);

    bool lockAvailable = false;
    if (entered)
    {
        const auto lockDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        do
        {
            if (g_vulkanLock.try_lock())
            {
                lockAvailable = true;
                g_vulkanLock.unlock();
                break;
            }
            std::this_thread::yield();
        }
        while (std::chrono::steady_clock::now() < lockDeadline);
    }

    const bool terminalized =
        entered && instance->disableCapture("capture terminalized while the application submit was paused");
    releaseApplicationSubmit.store(true, std::memory_order_release);
    releaseApplicationSubmit.notify_all();
    producer.join();
    pauseApplicationSubmit.store(false, std::memory_order_release);
    return entered && lockAvailable && terminalized;
}

std::string UniqueTestSuffix()
{
    const auto processId = static_cast<uint64_t>(getpid());
    return std::to_string(processId) + "-" + std::to_string(++nextHandle);
}

void ObserveFileSystem(capture::FileSystemOperation) noexcept
{
    const size_t current = CurrentThreadHash();
    if (current == observerProducer0.load(std::memory_order_relaxed)
        || current == observerProducer1.load(std::memory_order_relaxed))
    {
        observerRanOnProducer.store(true, std::memory_order_relaxed);
    }
    size_t expected = 0;
    if (!observerWriter.compare_exchange_strong(expected, current, std::memory_order_relaxed) && expected != current)
    {
        observerSawMultipleWorkers.store(true, std::memory_order_relaxed);
    }
}

void ResetFileSystemObserverState() noexcept
{
    observerProducer0.store(CurrentThreadHash(), std::memory_order_relaxed);
    observerProducer1.store(0, std::memory_order_relaxed);
    observerWriter.store(0, std::memory_order_relaxed);
    observerRanOnProducer.store(false, std::memory_order_relaxed);
    observerSawMultipleWorkers.store(false, std::memory_order_relaxed);
}

void ObserveCollector(capture::CollectorOperation operation) noexcept
{
    const size_t current = CurrentThreadHash();
    if (current == observerProducer0.load(std::memory_order_relaxed)
        || current == observerProducer1.load(std::memory_order_relaxed))
    {
        observerCollectorRanOnProducer.store(true, std::memory_order_relaxed);
    }
    size_t expected = 0;
    (void)observerCollector.compare_exchange_strong(expected, current, std::memory_order_relaxed);
    if (operation == capture::CollectorOperation::CopyMemory)
    {
        observerCollectorCopies.fetch_add(1, std::memory_order_relaxed);
    }
}

void ResetCollectorObserverState() noexcept
{
    observerCollector.store(0, std::memory_order_relaxed);
    observerCollectorRanOnProducer.store(false, std::memory_order_relaxed);
    observerCollectorCopies.store(0, std::memory_order_relaxed);
}
int dispatchKeyStorage = 0;
int secondDeviceDispatchKeyStorage = 0;
Dispatchable instanceObject{&dispatchKeyStorage};
Dispatchable physicalDeviceObject{&dispatchKeyStorage};
Dispatchable deviceObject{&dispatchKeyStorage};
Dispatchable deviceObject1{&secondDeviceDispatchKeyStorage};
Dispatchable queueObject{&dispatchKeyStorage};
Dispatchable queueObject1{&dispatchKeyStorage};
Dispatchable commandPoolObject{&dispatchKeyStorage};
Dispatchable commandPoolObject1{&dispatchKeyStorage};
Dispatchable commandBufferObject{&dispatchKeyStorage};
Dispatchable commandBufferObject1{&dispatchKeyStorage};

void Check(bool condition, std::string_view message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Handle> Handle DispatchHandle(Dispatchable &object)
{
    return reinterpret_cast<Handle>(&object);
}

template <typename Handle> Handle NewHandle()
{
    const uintptr_t value = nextHandle.fetch_add(0x10, std::memory_order_relaxed) + 0x10;
    return reinterpret_cast<Handle>(value);
}

VkInstance InstanceHandle()
{
    return DispatchHandle<VkInstance>(instanceObject);
}

VkPhysicalDevice PhysicalDeviceHandle()
{
    return DispatchHandle<VkPhysicalDevice>(physicalDeviceObject);
}

VkDevice DeviceHandle()
{
    return DispatchHandle<VkDevice>(deviceObject);
}

VkDevice DeviceHandle1()
{
    return DispatchHandle<VkDevice>(deviceObject1);
}

VkQueue QueueHandle()
{
    return DispatchHandle<VkQueue>(queueObject);
}

VkQueue QueueHandle1()
{
    return DispatchHandle<VkQueue>(queueObject1);
}

VkCommandPool CommandPoolHandle()
{
    return DispatchHandle<VkCommandPool>(commandPoolObject);
}

VkCommandPool CommandPoolHandle1()
{
    return DispatchHandle<VkCommandPool>(commandPoolObject1);
}

VkCommandBuffer CommandBufferHandle()
{
    return DispatchHandle<VkCommandBuffer>(commandBufferObject);
}

VkCommandBuffer CommandBufferHandle1()
{
    return DispatchHandle<VkCommandBuffer>(commandBufferObject1);
}

bool HasStructure(const void *chain, VkStructureType target)
{
    auto *current = reinterpret_cast<const VkBaseInStructure *>(chain);
    while (current != nullptr)
    {
        if (current->sType == target)
        {
            return true;
        }
        current = current->pNext;
    }
    return false;
}

bool HasDeviceExtension(const vku::safe_VkDeviceCreateInfo& createInfo, const char* target)
{
    for (uint32_t i = 0; i < createInfo.enabledExtensionCount; ++i)
    {
        if (std::strcmp(createInfo.ppEnabledExtensionNames[i], target) == 0)
        {
            return true;
        }
    }
    return false;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeEnumerateInstanceVersion(uint32_t *version)
{
    *version = VK_API_VERSION_1_1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateInstance(const VkInstanceCreateInfo* info,
                                                  const VkAllocationCallbacks *,
                                                  VkInstance *instance)
{
    ++driver.instanceCreates;
    // The application call uses its caller-owned output slot. Framework
    // version probes use separate temporary slots; their number may vary.
    if (instance == driver.applicationInstanceOutput)
    {
        ++driver.applicationInstanceCreates;
    }
    else
    {
        ++driver.auxiliaryInstanceCreates;
    }
    driver.lastInstanceCreateHadApplicationInfo = info->pApplicationInfo != nullptr;
    driver.lastInstanceCreateApiVersion = info->pApplicationInfo == nullptr ? 0 : info->pApplicationInfo->apiVersion;
    driver.writerRootObservedBeforeEveryInstanceCreate &=
        !expectedInstanceRoot.empty() && std::filesystem::exists(expectedInstanceRoot);
    if (driver.instanceCreateResult == VK_SUCCESS)
    {
        *instance = InstanceHandle();
    }
    return driver.instanceCreateResult;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeEnumerateDeviceExtensionProperties(VkPhysicalDevice,
                                                                      const char*,
                                                                      uint32_t* count,
                                                                      VkExtensionProperties* properties)
{
    ++driver.deviceExtensionQueries;
    static constexpr std::array<const char *, 3> names{
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME,
    };
    const uint32_t availableCount = driver.exposeNeuralStatisticsExtension ? 3u : 2u;
    if (properties == nullptr)
    {
        *count = availableCount;
        return VK_SUCCESS;
    }
    const uint32_t written = std::min(*count, availableCount);
    for (uint32_t i = 0; i < written; ++i)
    {
        properties[i] = {};
        std::strncpy(properties[i].extensionName, names[i], VK_MAX_EXTENSION_NAME_SIZE - 1);
        properties[i].specVersion = 1;
    }
    *count = written;
    return written == availableCount ? VK_SUCCESS : VK_INCOMPLETE;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceFeatures2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features)
{
    ++driver.physicalDeviceFeatureQueries;
    auto *current = reinterpret_cast<VkBaseOutStructure *>(features->pNext);
    while (current != nullptr)
    {
        if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES)
        {
            reinterpret_cast<VkPhysicalDeviceSynchronization2Features *>(current)
                ->synchronization2 = VK_TRUE;
        }
        else if (current->sType
                 == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM)
        {
            ++driver.neuralStatisticsFeatureQueries;
            reinterpret_cast<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM*>(current)
                ->dataGraphNeuralAcceleratorStatistics = driver.neuralStatisticsFeatureSupported ? VK_TRUE : VK_FALSE;
        }
        current = current->pNext;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateDevice(VkPhysicalDevice,
                                                const VkDeviceCreateInfo* info,
                                                const VkAllocationCallbacks*,
                                                VkDevice* device)
{
    ++driver.deviceCreates;
    driver.lastDeviceCreateEnabledExtensionCount = info->enabledExtensionCount;
    driver.requirements2ExtensionEnabled = false;
    driver.neuralStatisticsExtensionEnabled = false;
    for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
    {
        driver.requirements2ExtensionEnabled |=
            std::strcmp(info->ppEnabledExtensionNames[i], VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) == 0;
        driver.neuralStatisticsExtensionEnabled |=
            std::strcmp(info->ppEnabledExtensionNames[i],
                        VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME)
            == 0;
    }
    const auto* neuralStatistics =
        vku::FindStructInPNextChain<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(info->pNext);
    driver.neuralStatisticsFeatureEnabled =
        neuralStatistics != nullptr && neuralStatistics->dataGraphNeuralAcceleratorStatistics == VK_TRUE;
    const auto* synchronization2 = vku::FindStructInPNextChain<VkPhysicalDeviceSynchronization2Features>(info->pNext);
    driver.synchronization2FeatureEnabled = synchronization2 != nullptr && synchronization2->synchronization2;
    if (pauseDeviceCreate.load(std::memory_order_acquire))
    {
        deviceCreateEntered.store(true, std::memory_order_release);
        deviceCreateEntered.notify_all();
        while (!releaseDeviceCreate.load(std::memory_order_acquire))
        {
            releaseDeviceCreate.wait(false, std::memory_order_relaxed);
        }
    }
    if (driver.deviceCreateResult == VK_SUCCESS)
    {
        *device = driver.deviceCreates == 1 ? DeviceHandle() : DeviceHandle1();
    }
    return driver.deviceCreateResult;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyInstance(VkInstance, const VkAllocationCallbacks *)
{
    ++driver.instanceDestroys;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyDevice(VkDevice, const VkAllocationCallbacks *)
{
    ++driver.deviceDestroys;
    if (driver.probeDeviceDestroyLock)
    {
        std::thread probe([] {
            if (g_vulkanLock.try_lock())
            {
                driver.deviceDestroyLockAvailable = true;
                g_vulkanLock.unlock();
            }
        });
        probe.join();
    }
    driver.collectorJobsAtDeviceDestroy = Instance::retrieve(InstanceHandle())->collectorJobCount();
    std::lock_guard lock(fakeFenceMutex);
    driver.liveFencesAtDeviceDestroy = fakeFenceStatuses.size();
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties* properties)
{
    ++driver.physicalDevicePropertyQueries;
    *properties = {};
    properties->apiVersion = VK_API_VERSION_1_0;
    properties->driverVersion = 56;
    properties->vendorID = 0x13B5;
    properties->deviceID = 0x1000;
    std::strncpy(properties->deviceName, "Fake r56 data-graph device",
                 VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,
                                                                 VkPhysicalDeviceMemoryProperties* properties)
{
    *properties = {};
    properties->memoryTypeCount = 1;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[0].propertyFlags = driver.memoryTypeProperties;
    properties->memoryHeapCount = 1;
    properties->memoryHeaps[0].size = 1 << 20;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice,
                                                                      uint32_t* count,
                                                                      VkQueueFamilyProperties* properties)
{
    ++driver.queueFamilyPropertyQueries;
    if (driver.queueFamilyProperties.empty())
    {
        if (properties == nullptr)
        {
            *count = 1;
            return;
        }
        *count = 1;
        properties[0] = {};
        properties[0].queueFlags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_DATA_GRAPH_BIT_ARM;
        properties[0].queueCount = 4;
        return;
    }

    if (properties == nullptr)
    {
        *count = static_cast<uint32_t>(driver.queueFamilyProperties.size());
        return;
    }
    const uint32_t copied = std::min(*count, static_cast<uint32_t>(driver.queueFamilyProperties.size()));
    std::copy_n(driver.queueFamilyProperties.begin(), copied, properties);
    *count = copied;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateShaderModule(VkDevice,
                                                      const VkShaderModuleCreateInfo*,
                                                      const VkAllocationCallbacks *,
                                                      VkShaderModule *shader)
{
    ++driver.shaderCreates;
    if (driver.throwShaderCreate)
    {
        throw std::runtime_error("downstream shader exception");
    }
    if (driver.shaderResult == VK_SUCCESS)
    {
        *shader = NewHandle<VkShaderModule>();
    }
    return driver.shaderResult;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
{
    ++driver.shaderDestroys;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateDataGraphPipelines(VkDevice,
                                                            VkDeferredOperationKHR,
                                                            VkPipelineCache,
                                                            uint32_t count,
                                                            const VkDataGraphPipelineCreateInfoARM* infos,
                                                            const VkAllocationCallbacks*,
                                                            VkPipeline* pipelines)
{
    ++driver.pipelineCreates;
    driver.lastPipelineHandles.clear();
    driver.lastPipelineCreateHadPNext = false;
    driver.lastPipelineCreateHadNeuralStatistics = false;
    driver.lastPipelineCreateInfoObserved = count != 0;
    if (count != 0)
    {
        std::memcpy(&driver.lastPipelineCreateInfo, &infos[0], sizeof(infos[0]));
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        driver.lastPipelineCreateHadPNext |= infos[i].pNext != nullptr;
        driver.lastPipelineCreateHadNeuralStatistics |=
            HasStructure(infos[i].pNext, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_NEURAL_STATISTICS_CREATE_INFO_ARM);
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        const uintptr_t configured = i < driver.pipelineOutputs.size() ? driver.pipelineOutputs[i] : uintptr_t {1};
        pipelines[i] = configured == 0 ? VK_NULL_HANDLE : NewHandle<VkPipeline>();
        driver.lastPipelineHandles.push_back(pipelines[i]);
    }
    if (terminalizeCaptureDuringPipelineCreate != nullptr)
    {
        Instance* instance = terminalizeCaptureDuringPipelineCreate;
        terminalizeCaptureDuringPipelineCreate = nullptr;
        (void) instance->disableCapture("capture terminalized during downstream pipeline creation");
    }
    return driver.pipelineResult;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyPipeline(VkDevice, VkPipeline, const VkAllocationCallbacks *)
{
    ++driver.pipelineDestroys;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetDataGraphPipelineAvailableProperties(VkDevice,
                                                                           const VkDataGraphPipelineInfoARM*,
                                                                           uint32_t* count,
    VkDataGraphPipelinePropertyARM *properties)
{
    if (!driver.exposeStaticPipelineProperties)
    {
        *count = 0;
        return VK_SUCCESS;
    }
    constexpr std::array values{
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM,
        VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM,
    };
    if (properties == nullptr)
    {
        *count = static_cast<uint32_t>(values.size());
        return VK_SUCCESS;
    }
    const uint32_t written = std::min(*count, static_cast<uint32_t>(values.size()));
    std::copy_n(values.begin(), written, properties);
    *count = written;
    return written == values.size() ? VK_SUCCESS : VK_INCOMPLETE;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetDataGraphPipelineProperties(VkDevice,
                                                                  const VkDataGraphPipelineInfoARM*,
                                                                  uint32_t count,
                                   VkDataGraphPipelinePropertyQueryResultARM *queries)
{
    ++driver.pipelinePropertyQueries;
    static constexpr std::array<uint8_t, 4> debugDatabase{0x10, 0x20, 0x30, 0x40};
    static constexpr std::array<uint8_t, 6> statisticsInfo{'s', 't', 'a', 't', 's', 0};
    for (uint32_t i = 0; i < count; ++i)
    {
        const bool statistics =
            queries[i].property == VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM;
        const auto *source = statistics ? statisticsInfo.data() : debugDatabase.data();
        const size_t size = statistics ? statisticsInfo.size() : debugDatabase.size();
        queries[i].isText = statistics ? VK_TRUE : VK_FALSE;
        if (queries[i].pData != nullptr)
        {
            std::memcpy(queries[i].pData, source, size);
        }
        queries[i].dataSize = size;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateSession(VkDevice,
                                                 const VkDataGraphPipelineSessionCreateInfoARM* info,
                                                 const VkAllocationCallbacks*,
                                                 VkDataGraphPipelineSessionARM* session)
{
    const bool instrumented =
        HasStructure(info->pNext, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_NEURAL_STATISTICS_CREATE_INFO_ARM);
    VkResult result = VK_SUCCESS;
    if (instrumented)
    {
        ++driver.instrumentedSessionCreates;
        result = driver.instrumentedSessionResult;
    }
    else
    {
        ++driver.originalSessionCreates;
        result = driver.originalSessionResult;
    }
    if (result == VK_SUCCESS)
    {
        *session = NewHandle<VkDataGraphPipelineSessionARM>();
        if (instrumented && terminalizeCaptureDuringSessionCreate != nullptr)
        {
            Instance* instance = terminalizeCaptureDuringSessionCreate;
            terminalizeCaptureDuringSessionCreate = nullptr;
            (void) instance->disableCapture("capture terminalized during downstream session creation");
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroySession(VkDevice, VkDataGraphPipelineSessionARM, const VkAllocationCallbacks*)
{
    ++driver.sessionDestroys;
}

VKAPI_ATTR VkResult VKAPI_CALL
    FakeGetSessionBindPointRequirements(VkDevice,
                                        const VkDataGraphPipelineSessionBindPointRequirementsInfoARM*,
                                        uint32_t* count,
    VkDataGraphPipelineSessionBindPointRequirementARM *requirements)
{
    if (requirements == nullptr)
    {
        *count = 1;
        return VK_SUCCESS;
    }
    *count = 1;
    requirements[0] = {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM,
        nullptr,
        VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_NEURAL_ACCELERATOR_STATISTICS_ARM,
        VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM,
        1,
    };
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeGetSessionMemoryRequirements(VkDevice,
                                                            const VkDataGraphPipelineSessionMemoryRequirementsInfoARM*,
    VkMemoryRequirements2 *requirements)
{
    requirements->memoryRequirements = {64, 64, 1};
    if (requirements->pNext != nullptr)
    {
        auto *dedicated = reinterpret_cast<VkMemoryDedicatedRequirements *>(requirements->pNext);
        dedicated->requiresDedicatedAllocation = VK_FALSE;
        dedicated->prefersDedicatedAllocation = VK_FALSE;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateBuffer(VkDevice,
                                                const VkBufferCreateInfo* createInfo,
                                                const VkAllocationCallbacks*,
                                                VkBuffer* buffer)
{
    ++driver.bufferCreates;
    if (pauseBufferCreate.load(std::memory_order_acquire))
    {
        bufferCreateEntered.store(true, std::memory_order_release);
        bufferCreateEntered.notify_all();
        while (!releaseBufferCreate.load(std::memory_order_acquire))
        {
            releaseBufferCreate.wait(false, std::memory_order_relaxed);
        }
    }
    if (driver.bufferCreateResult != VK_SUCCESS)
    {
        return driver.bufferCreateResult;
    }
    BufferCreateObservation observation;
    observation.usage = createInfo->usage;
    observation.sharingMode = createInfo->sharingMode;
    if (createInfo->queueFamilyIndexCount != 0 && createInfo->pQueueFamilyIndices != nullptr)
    {
        observation.queueFamilyIndices.assign(createInfo->pQueueFamilyIndices,
                                              createInfo->pQueueFamilyIndices + createInfo->queueFamilyIndexCount);
    }
    driver.bufferCreateObservations.emplace_back(std::move(observation));
    *buffer = NewHandle<VkBuffer>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks *)
{
    ++driver.bufferDestroys;
}

void FillRequirements(VkMemoryRequirements2 *requirements)
{
    requirements->memoryRequirements = {64, 64, 1};
    if (requirements->pNext != nullptr)
    {
        auto *dedicated = reinterpret_cast<VkMemoryDedicatedRequirements *>(requirements->pNext);
        dedicated->requiresDedicatedAllocation = VK_FALSE;
        dedicated->prefersDedicatedAllocation = VK_FALSE;
    }
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements2(VkDevice,
                                                            const VkBufferMemoryRequirementsInfo2 *,
                                                            VkMemoryRequirements2 *requirements)
{
    ++driver.requirements2CoreCalls;
    FillRequirements(requirements);
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements2KHR(VkDevice,
                                                               const VkBufferMemoryRequirementsInfo2*,
                                                               VkMemoryRequirements2* requirements)
{
    ++driver.requirements2KhrCalls;
    FillRequirements(requirements);
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements* requirements)
{
    *requirements = {64, 64, 1};
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateMemory(VkDevice,
                                                  const VkMemoryAllocateInfo*,
                                                  const VkAllocationCallbacks *,
                                                  VkDeviceMemory *memory)
{
    ++driver.memoryAllocations;
    *memory = NewHandle<VkDeviceMemory>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *)
{
    ++driver.memoryFrees;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory,
                                                    VkDeviceSize)
{
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindSessionMemory(VkDevice,
                                                     uint32_t,
                                                     const VkBindDataGraphPipelineSessionMemoryInfoARM*)
{
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateFence(VkDevice,
                                               const VkFenceCreateInfo*,
                                               const VkAllocationCallbacks*,
                                               VkFence* fence)
{
    if (driver.createFenceResult == VK_SUCCESS)
    {
        *fence = NewHandle<VkFence>();
        std::lock_guard lock(fakeFenceMutex);
        ++driver.fenceCreates;
        fakeFenceStatuses[as_key(*fence)] = VK_NOT_READY;
    }
    return driver.createFenceResult;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks *)
{
    std::lock_guard lock(fakeFenceMutex);
    ++driver.fenceDestroys;
    if (fakeFenceStatuses.erase(as_key(fence)) != 1)
    {
        ++driver.duplicateFenceDestroys;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetFenceStatus(VkDevice, VkFence fence)
{
    std::lock_guard lock(fakeFenceMutex);
    const auto it = fakeFenceStatuses.find(as_key(fence));
    return it == fakeFenceStatuses.end() ? VK_ERROR_UNKNOWN : it->second;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeDeviceWaitIdle(VkDevice)
{
    ++driver.deviceWaitIdleCalls;
    std::lock_guard lock(fakeFenceMutex);
    if (driver.deviceWaitIdleResult == VK_SUCCESS)
    {
        for (auto &[_, status] : fakeFenceStatuses)
        {
            status = VK_SUCCESS;
        }
    }
    return driver.deviceWaitIdleResult;
}

void SignalMarkerFence(std::size_t index, VkResult status = VK_SUCCESS)
{
    std::lock_guard lock(fakeFenceMutex);
    Check(index < submittedMarkerFences.size(), "requested marker fence exists");
    if (index < submittedMarkerFences.size())
    {
        fakeFenceStatuses[as_key(submittedMarkerFences[index])] = status;
    }
}

std::size_t MarkerFenceCount()
{
    std::lock_guard lock(fakeFenceMutex);
    return submittedMarkerFences.size();
}

bool WaitForFlag(std::atomic<bool>& flag, std::chrono::milliseconds timeout = std::chrono::seconds(2))
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (flag.load(std::memory_order_acquire))
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag.load(std::memory_order_acquire);
}

void CheckFenceAccounting(std::string_view context)
{
    std::lock_guard lock(fakeFenceMutex);
    const bool valid =
        fakeFenceStatuses.empty() && driver.fenceCreates == driver.fenceDestroys && driver.duplicateFenceDestroys == 0;
    if (!valid)
    {
        std::cerr << "FAIL: " << context << " (created=" << driver.fenceCreates
                  << ", destroyed=" << driver.fenceDestroys << ", duplicate-destroys=" << driver.duplicateFenceDestroys
                  << ", live=" << fakeFenceStatuses.size() << ")\n";
        ++failures;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueSubmit(VkQueue,
                                               uint32_t submitCount,
                                               const VkSubmitInfo* pSubmits,
                                               VkFence fence)
{
    if (fence == VK_NULL_HANDLE)
    {
        driver.lastLegacyCommandBuffers.clear();
        driver.lastLegacyCommandBufferDeviceMasks.clear();
        driver.lastLegacyWaitSemaphoreDeviceIndices.clear();
        driver.lastLegacySignalSemaphoreDeviceIndices.clear();
        driver.lastLegacyDeviceGroupHadProtectedPrefix = false;
        for (uint32_t i = 0; i < submitCount; ++i)
        {
            driver.lastLegacyCommandBuffers.insert(driver.lastLegacyCommandBuffers.end(),
                                                   pSubmits[i].pCommandBuffers,
                                                   pSubmits[i].pCommandBuffers + pSubmits[i].commandBufferCount);
            const auto* deviceGroup = vku::FindStructInPNextChain<VkDeviceGroupSubmitInfo>(pSubmits[i].pNext);
            if (deviceGroup != nullptr)
            {
                driver.lastLegacyCommandBufferDeviceMasks.insert(driver.lastLegacyCommandBufferDeviceMasks.end(),
                    deviceGroup->pCommandBufferDeviceMasks,
                                                                 deviceGroup->pCommandBufferDeviceMasks
                                                                     + deviceGroup->commandBufferCount);
                driver.lastLegacyWaitSemaphoreDeviceIndices.insert(driver.lastLegacyWaitSemaphoreDeviceIndices.end(),
                    deviceGroup->pWaitSemaphoreDeviceIndices,
                                                                   deviceGroup->pWaitSemaphoreDeviceIndices
                                                                       + deviceGroup->waitSemaphoreCount);
                driver.lastLegacySignalSemaphoreDeviceIndices.insert(
                    driver.lastLegacySignalSemaphoreDeviceIndices.end(),
                    deviceGroup->pSignalSemaphoreDeviceIndices,
                    deviceGroup->pSignalSemaphoreDeviceIndices + deviceGroup->signalSemaphoreCount);
                const auto *head = reinterpret_cast<const VkBaseInStructure *>(pSubmits[i].pNext);
                driver.lastLegacyDeviceGroupHadProtectedPrefix =
                    head != nullptr && head->sType == VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO
                    && head->pNext == reinterpret_cast<const VkBaseInStructure*>(deviceGroup);
            }
        }
        if (concurrentCollectionMode.load(std::memory_order_relaxed))
        {
            concurrentApplicationSubmits.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            ++driver.legacyApplicationSubmits;
        }
        PauseApplicationSubmitIfRequested();
        return driver.applicationSubmitResult;
    }
    if (concurrentCollectionMode.load(std::memory_order_relaxed))
    {
        concurrentMarkerSubmits.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        ++driver.markerSubmits;
    }
    if (driver.markerSubmitResult == VK_SUCCESS)
    {
        std::lock_guard lock(fakeFenceMutex);
        submittedMarkerFences.push_back(fence);
        if (autoSignalMarkerFences)
        {
            fakeFenceStatuses[as_key(fence)] = VK_SUCCESS;
        }
    }
    return driver.markerSubmitResult;
}

VKAPI_ATTR VkResult VKAPI_CALL
    FakeQueueSubmit2(VkQueue, uint32_t submitCount, const VkSubmitInfo2* submits, VkFence fence)
{
    if (fence == VK_NULL_HANDLE)
    {
        driver.lastSubmit2CommandBuffers.clear();
        for (uint32_t i = 0; i < submitCount; ++i)
        {
            for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; ++j)
            {
                driver.lastSubmit2CommandBuffers.emplace_back(submits[i].pCommandBufferInfos[j].commandBuffer);
            }
        }
        ++driver.coreSubmit2Calls;
        PauseApplicationSubmitIfRequested();
        return driver.applicationSubmitResult;
    }
    ++driver.coreMarkerSubmit2Calls;
    if (driver.markerSubmitResult == VK_SUCCESS)
    {
        std::lock_guard lock(fakeFenceMutex);
        submittedMarkerFences.push_back(fence);
        if (autoSignalMarkerFences)
        {
            fakeFenceStatuses[as_key(fence)] = VK_SUCCESS;
        }
    }
    return driver.markerSubmitResult;
}

VKAPI_ATTR VkResult VKAPI_CALL
    FakeQueueSubmit2KHR(VkQueue, uint32_t submitCount, const VkSubmitInfo2* submits, VkFence fence)
{
    if (fence == VK_NULL_HANDLE)
    {
        driver.lastSubmit2CommandBuffers.clear();
        for (uint32_t i = 0; i < submitCount; ++i)
        {
            for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; ++j)
            {
                driver.lastSubmit2CommandBuffers.emplace_back(submits[i].pCommandBufferInfos[j].commandBuffer);
            }
        }
        ++driver.khrSubmit2Calls;
        PauseApplicationSubmitIfRequested();
        return driver.applicationSubmitResult;
    }
    ++driver.khrMarkerSubmit2Calls;
    if (driver.markerSubmitResult == VK_SUCCESS)
    {
        std::lock_guard lock(fakeFenceMutex);
        submittedMarkerFences.push_back(fence);
        if (autoSignalMarkerFences)
        {
            fakeFenceStatuses[as_key(fence)] = VK_SUCCESS;
        }
    }
    return driver.markerSubmitResult;
}

VKAPI_ATTR VkResult VKAPI_CALL
    FakeMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** data)
{
    ++driver.mapMemoryCalls;
    if (driver.mapMemoryResult != VK_SUCCESS)
    {
        return driver.mapMemoryResult;
    }
    static std::array<uint8_t, 64> bytes{};
    *data = bytes.data();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeUnmapMemory(VkDevice, VkDeviceMemory) {}

VKAPI_ATTR VkResult VKAPI_CALL FakeInvalidateMappedMemoryRanges(VkDevice, uint32_t, const VkMappedMemoryRange*)
{
    ++driver.invalidateMemoryCalls;
    return driver.invalidateMemoryResult;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
{
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateCommandPool(VkDevice,
                                                     const VkCommandPoolCreateInfo*,
                                                     const VkAllocationCallbacks *,
                                                     VkCommandPool *commandPool)
{
    *commandPool = NewHandle<VkCommandPool>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
{
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateCommandBuffers(VkDevice,
                                                          const VkCommandBufferAllocateInfo *info,
                                                          VkCommandBuffer *commandBuffers)
{
    for (uint32_t i = 0; i < info->commandBufferCount; ++i)
    {
        commandBuffers[i] = NewHandle<VkCommandBuffer>();
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeFreeCommandBuffers(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*)
{
    ++driver.commandBufferFrees;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
{
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeEndCommandBuffer(VkCommandBuffer)
{
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdDispatchDataGraphARM(VkCommandBuffer,
                                                       VkDataGraphPipelineSessionARM,
                                                       const VkDataGraphPipelineDispatchInfoARM*)
{
    ++driver.cmdDispatchDataGraphCalls;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdFillBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t)
{
    ++driver.cmdFillBufferCalls;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*)
{
    ++driver.cmdCopyBufferCalls;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdExecuteCommands(VkCommandBuffer,
                                                  uint32_t count,
                                                  const VkCommandBuffer* commandBuffers)
{
    ++driver.cmdExecuteCommandsCalls;
    driver.cmdExecuteCommandBuffers.emplace_back();
    if (count != 0)
    {
        driver.cmdExecuteCommandBuffers.back().assign(commandBuffers, commandBuffers + count);
    }
}

VKAPI_ATTR void VKAPI_CALL FakeCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo*)
{
    ++driver.cmdPipelineBarrierCalls;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdPipelineBarrier2KHR(VkCommandBuffer, const VkDependencyInfo*)
{
    ++driver.cmdPipelineBarrierCalls;
}

PFN_vkVoidFunction Function(PFN_vkVoidFunction function)
{
    return function;
}

template <typename FunctionType> PFN_vkVoidFunction Function(FunctionType function)
{
    return reinterpret_cast<PFN_vkVoidFunction>(function);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FakeGetInstanceProcAddr(VkInstance, const char *name)
{
    if (std::strcmp(name, "vkEnumerateInstanceVersion") == 0)
    {
        return Function(FakeEnumerateInstanceVersion);
    }
    if (std::strcmp(name, "vkCreateInstance") == 0)
    {
        return Function(FakeCreateInstance);
    }
    if (std::strcmp(name, "vkCreateDevice") == 0)
    {
        return Function(FakeCreateDevice);
    }
    if (std::strcmp(name, "vkDestroyInstance") == 0)
    {
        return Function(FakeDestroyInstance);
    }
    if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
    {
        return Function(FakeEnumerateDeviceExtensionProperties);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties") == 0)
    {
        return Function(FakeGetPhysicalDeviceProperties);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0)
    {
        return Function(FakeGetPhysicalDeviceFeatures2);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0)
    {
        return Function(FakeGetPhysicalDeviceMemoryProperties);
    }
    if (std::strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties") == 0)
    {
        return Function(FakeGetPhysicalDeviceQueueFamilyProperties);
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FakeGetDeviceProcAddr(VkDevice, const char *name)
{
    if (std::strcmp(name, "vkDestroyDevice") == 0)
    {
        return Function(FakeDestroyDevice);
    }
    if (std::strcmp(name, "vkCreateShaderModule") == 0)
    {
        return Function(FakeCreateShaderModule);
    }
    if (std::strcmp(name, "vkDestroyShaderModule") == 0)
    {
        return Function(FakeDestroyShaderModule);
    }
    if (std::strcmp(name, "vkCreateDataGraphPipelinesARM") == 0)
    {
        return Function(FakeCreateDataGraphPipelines);
    }
    if (std::strcmp(name, "vkDestroyPipeline") == 0)
    {
        return Function(FakeDestroyPipeline);
    }
    if (std::strcmp(name, "vkGetDataGraphPipelineAvailablePropertiesARM") == 0)
    {
        return Function(FakeGetDataGraphPipelineAvailableProperties);
    }
    if (std::strcmp(name, "vkGetDataGraphPipelinePropertiesARM") == 0)
    {
        return Function(FakeGetDataGraphPipelineProperties);
    }
    if (std::strcmp(name, "vkCreateDataGraphPipelineSessionARM") == 0)
    {
        return Function(FakeCreateSession);
    }
    if (std::strcmp(name, "vkDestroyDataGraphPipelineSessionARM") == 0)
    {
        return Function(FakeDestroySession);
    }
    if (std::strcmp(name, "vkGetDataGraphPipelineSessionBindPointRequirementsARM") == 0)
    {
        return Function(FakeGetSessionBindPointRequirements);
    }
    if (std::strcmp(name, "vkGetDataGraphPipelineSessionMemoryRequirementsARM") == 0)
    {
        return Function(FakeGetSessionMemoryRequirements);
    }
    if (std::strcmp(name, "vkCreateBuffer") == 0)
    {
        return Function(FakeCreateBuffer);
    }
    if (std::strcmp(name, "vkDestroyBuffer") == 0)
    {
        return Function(FakeDestroyBuffer);
    }
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
    {
        return Function(FakeGetBufferMemoryRequirements);
    }
    if (std::strcmp(name, "vkGetBufferMemoryRequirements2") == 0)
    {
        return driver.exposeRequirements2Core ? Function(FakeGetBufferMemoryRequirements2) : nullptr;
    }
    if (std::strcmp(name, "vkGetBufferMemoryRequirements2KHR") == 0)
    {
        return driver.exposeRequirements2Khr ? Function(FakeGetBufferMemoryRequirements2KHR) : nullptr;
    }
    if (std::strcmp(name, "vkAllocateMemory") == 0)
    {
        return Function(FakeAllocateMemory);
    }
    if (std::strcmp(name, "vkFreeMemory") == 0)
    {
        return Function(FakeFreeMemory);
    }
    if (std::strcmp(name, "vkBindBufferMemory") == 0)
    {
        return Function(FakeBindBufferMemory);
    }
    if (std::strcmp(name, "vkBindDataGraphPipelineSessionMemoryARM") == 0)
    {
        return Function(FakeBindSessionMemory);
    }
    if (std::strcmp(name, "vkCreateFence") == 0)
    {
        return Function(FakeCreateFence);
    }
    if (std::strcmp(name, "vkDestroyFence") == 0)
    {
        return Function(FakeDestroyFence);
    }
    if (std::strcmp(name, "vkGetFenceStatus") == 0)
    {
        return Function(FakeGetFenceStatus);
    }
    if (std::strcmp(name, "vkDeviceWaitIdle") == 0)
    {
        return Function(FakeDeviceWaitIdle);
    }
    if (std::strcmp(name, "vkQueueSubmit") == 0)
    {
        return Function(FakeQueueSubmit);
    }
    if (std::strcmp(name, "vkQueueSubmit2") == 0)
    {
        return driver.exposeCoreSubmit2 ? Function(FakeQueueSubmit2) : nullptr;
    }
    if (std::strcmp(name, "vkQueueSubmit2KHR") == 0)
    {
        return driver.exposeKhrSubmit2 ? Function(FakeQueueSubmit2KHR) : nullptr;
    }
    if (std::strcmp(name, "vkMapMemory") == 0)
    {
        return Function(FakeMapMemory);
    }
    if (std::strcmp(name, "vkUnmapMemory") == 0)
    {
        return Function(FakeUnmapMemory);
    }
    if (std::strcmp(name, "vkInvalidateMappedMemoryRanges") == 0)
    {
        return Function(FakeInvalidateMappedMemoryRanges);
    }
    if (std::strcmp(name, "vkResetCommandBuffer") == 0)
    {
        return Function(FakeResetCommandBuffer);
    }
    if (std::strcmp(name, "vkCreateCommandPool") == 0)
    {
        return Function(FakeCreateCommandPool);
    }
    if (std::strcmp(name, "vkDestroyCommandPool") == 0)
    {
        return Function(FakeDestroyCommandPool);
    }
    if (std::strcmp(name, "vkAllocateCommandBuffers") == 0)
    {
        return Function(FakeAllocateCommandBuffers);
    }
    if (std::strcmp(name, "vkFreeCommandBuffers") == 0)
    {
        return Function(FakeFreeCommandBuffers);
    }
    if (std::strcmp(name, "vkBeginCommandBuffer") == 0)
    {
        return Function(FakeBeginCommandBuffer);
    }
    if (std::strcmp(name, "vkEndCommandBuffer") == 0)
    {
        return Function(FakeEndCommandBuffer);
    }
    if (std::strcmp(name, "vkCmdDispatchDataGraphARM") == 0)
    {
        return Function(FakeCmdDispatchDataGraphARM);
    }
    if (std::strcmp(name, "vkCmdFillBuffer") == 0)
    {
        return Function(FakeCmdFillBuffer);
    }
    if (std::strcmp(name, "vkCmdCopyBuffer") == 0)
    {
        return Function(FakeCmdCopyBuffer);
    }
    if (std::strcmp(name, "vkCmdExecuteCommands") == 0)
    {
        return Function(FakeCmdExecuteCommands);
    }
    if (std::strcmp(name, "vkCmdPipelineBarrier2") == 0)
    {
        return Function(FakeCmdPipelineBarrier2);
    }
    if (std::strcmp(name, "vkCmdPipelineBarrier2KHR") == 0)
    {
        return Function(FakeCmdPipelineBarrier2KHR);
    }
    return nullptr;
}

VkInstanceCreateInfo MakeInstanceCreateInfo(const char *&captureRoot,
                                            VkApplicationInfo &applicationInfo,
                                            std::array<VkLayerSettingEXT, 3> &settings,
                                            VkLayerSettingsCreateInfoEXT &settingsInfo,
                                            VkLayerInstanceLink &link,
                                            VkLayerInstanceCreateInfo &loaderInfo)
{
    static constexpr const char *layerName = "VK_LAYER_LGL_neural_statistics";
    static constexpr const char *mode = "0";
    static constexpr const char *filter = "";
    settings = {{
        {layerName, "capture_folder", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &captureRoot},
        {layerName, "statistics_mode", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &mode},
        {layerName, "dispatch_filter", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &filter},
    }};
    link = {nullptr, FakeGetInstanceProcAddr, nullptr};
    loaderInfo = {};
    loaderInfo.sType = VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO;
    loaderInfo.pNext = nullptr;
    loaderInfo.function = VK_LAYER_LINK_INFO;
    loaderInfo.u.pLayerInfo = &link;
    settingsInfo = {
        VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        &loaderInfo,
        static_cast<uint32_t>(settings.size()),
        settings.data(),
    };
    applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "live-hook-test",
        1,
        "live-hook-test",
        1,
        VK_API_VERSION_1_0,
    };
    return {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        &settingsInfo,
        0,
        &applicationInfo,
        0,
        nullptr,
        0,
        nullptr,
    };
}

VkDeviceCreateInfo MakeDeviceCreateInfo(VkLayerDeviceLink &link,
                                        VkLayerDeviceCreateInfo &loaderInfo,
                                        VkDeviceQueueCreateInfo &queueInfo, float &priority)
{
    link = {nullptr, FakeGetInstanceProcAddr, FakeGetDeviceProcAddr};
    loaderInfo = {};
    loaderInfo.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    loaderInfo.function = VK_LAYER_LINK_INFO;
    loaderInfo.u.pLayerInfo = &link;
    priority = 1.0f;
    queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &priority};
    return {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &loaderInfo,
        0,
        1,
        &queueInfo,
        0,
        nullptr,
        0,
        nullptr,
        nullptr,
    };
}

class LiveFixture
{
  public:
    explicit LiveFixture(bool coreRequirements2 = true, bool khrRequirements2 = true,
                         bool coreSubmit2 = true, bool khrSubmit2 = true,
                         std::size_t collectorCapacity = 64,
                         std::vector<VkQueueFamilyProperties> queueFamilyProperties = {},
                         std::vector<uint32_t> requestedQueueFamilies = {},
                         bool neuralStatisticsEnabled = true)
    {
        driver = {};
        terminalizeCaptureDuringPipelineCreate = nullptr;
        terminalizeCaptureDuringSessionCreate = nullptr;
        pauseBufferCreate.store(false, std::memory_order_relaxed);
        bufferCreateEntered.store(false, std::memory_order_relaxed);
        releaseBufferCreate.store(false, std::memory_order_relaxed);
        pauseApplicationSubmit.store(false, std::memory_order_relaxed);
        applicationSubmitEntered.store(false, std::memory_order_relaxed);
        releaseApplicationSubmit.store(false, std::memory_order_relaxed);
        driver.queueFamilyProperties = std::move(queueFamilyProperties);
        driver.exposeRequirements2Core = coreRequirements2;
        driver.exposeRequirements2Khr = khrRequirements2;
        driver.exposeCoreSubmit2 = coreSubmit2;
        driver.exposeKhrSubmit2 = khrSubmit2;
        {
            std::lock_guard lock(fakeFenceMutex);
            fakeFenceStatuses.clear();
            submittedMarkerFences.clear();
            autoSignalMarkerFences = true;
        }

        auto instance = std::make_unique<Instance>(InstanceHandle(), FakeGetInstanceProcAddr);
        instance->layerOptions = std::make_unique<LayerOptions>();
        Instance *instancePointer = instance.get();
        Instance::store(InstanceHandle(), instance);

        if (requestedQueueFamilies.empty())
        {
            requestedQueueFamilies.emplace_back(0);
        }
        std::vector<float> priorities(requestedQueueFamilies.size(), 1.0f);
        std::vector<VkDeviceQueueCreateInfo> queueInfos(requestedQueueFamilies.size());
        for (size_t i = 0; i < requestedQueueFamilies.size(); ++i)
        {
            queueInfos[i] = {
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                nullptr,
                0,
                requestedQueueFamilies[i],
                1,
                &priorities[i],
            };
        }
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM neuralStatistics {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
            nullptr,
            neuralStatisticsEnabled ? VK_TRUE : VK_FALSE,
        };
        const char* neuralStatisticsExtension = VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME;
        const VkDeviceCreateInfo createInfo{
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            neuralStatisticsEnabled ? &neuralStatistics : nullptr,
            0,
            static_cast<uint32_t>(queueInfos.size()),
            queueInfos.data(),
            0,
            nullptr,
            neuralStatisticsEnabled ? 1u : 0u,
            neuralStatisticsEnabled ? &neuralStatisticsExtension : nullptr,
            nullptr,
        };
        auto device =
            std::make_unique<Device>(instancePointer,
                                     PhysicalDeviceHandle(),
                                     DeviceHandle(),
                                     FakeGetDeviceProcAddr,
            createInfo,
            capture::GpuCollectorOptions{collectorCapacity, std::chrono::milliseconds(1)});
        device->synchronization2Enabled = true;
        Device::store(DeviceHandle(), std::move(device));
    }

    ~LiveFixture()
    {
        destroyDevice();
        destroyInstance();
    }

    void destroyDevice()
    {
        if (deviceAlive_)
        {
            layer_vkDestroyDevice<user_tag>(DeviceHandle(), nullptr);
            deviceAlive_ = false;
        }
    }

    void destroyInstance()
    {
        if (instanceAlive_)
        {
            layer_vkDestroyInstance<user_tag>(InstanceHandle(), nullptr);
            instanceAlive_ = false;
        }
    }

    Device *device()
    {
        return Device::retrieve(DeviceHandle());
    }
    Instance *instance()
    {
        return Instance::retrieve(InstanceHandle());
    }

  private:
    bool deviceAlive_{true};
    bool instanceAlive_{true};
};

void StartCaptureWriter(LiveFixture& fixture, const std::filesystem::path& root, std::size_t queueCapacity = 64)
{
    capture::CaptureMetadata metadata;
    metadata.layerName = "live-hook-test";
    metadata.layerVersion = "1.0.0";
    metadata.commitIdentity = "test";
    metadata.layerImplementationVersion = 1;
    metadata.statisticsMode = 0;
    metadata.dispatchFilter = "all";
    metadata.devices.push_back(capture::DeviceMetadata {capture::LogicalDeviceId(0),
                                                        "fake-device",
                                                        std::nullopt,
                                                        std::nullopt,
                                                        std::nullopt,
                                                       VK_API_VERSION_1_0});
    fixture.instance()->captureMetadata = metadata;
    fixture.instance()->dispatchFilter = DispatchFilter::All();
    fixture.instance()->captureWriter = std::make_unique<capture::CaptureWriter>(
        capture::CaptureWriterOptions{root, metadata, DispatchFilter::All(), queueCapacity});
    Check(fixture.instance()->captureWriter->start().ok(), "live-hook writer starts before producer work");
}

bool WaitForCollectorCount(Instance &instance, uint64_t count,
                           std::chrono::milliseconds timeout = std::chrono::seconds(2))
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (instance.collectorJobCount() == count)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return instance.collectorJobCount() == count;
}

void TestSafeMicromapUsageDeepCopy()
{
    // The direct and indirect arrays are mutually exclusive, and every indirect entry is valid.
    for (const bool useIndirect : {false, true})
    {
        VkMicromapUsageKHR usage[2] {
            {3, 4, VK_OPACITY_MICROMAP_FORMAT_2_STATE_KHR},
            {5, 6, VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR},
        };
        const VkMicromapUsageKHR* indirect[2] {&usage[0], &usage[1]};
        const VkAccelerationStructureGeometryMicromapDataKHR source {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR,
            nullptr,
            2,
            useIndirect ? nullptr : usage,
            useIndirect ? indirect : nullptr,
            0x1000,
            0x2000,
            16,
        };

        const auto ownsIndependentCopy = [useIndirect](const auto& owned, const auto& borrowed)
        {
            if (owned.usageCountsCount != 2 || borrowed.usageCountsCount != 2)
            {
                return false;
            }
            if (useIndirect)
            {
                return owned.pUsageCounts == nullptr && owned.ppUsageCounts != nullptr
                    && borrowed.ppUsageCounts != nullptr && owned.ppUsageCounts != borrowed.ppUsageCounts
                    && owned.ppUsageCounts[0] != nullptr && owned.ppUsageCounts[1] != nullptr
                    && owned.ppUsageCounts[0] != borrowed.ppUsageCounts[0]
                    && owned.ppUsageCounts[1] != borrowed.ppUsageCounts[1]
                    && owned.ppUsageCounts[0]->count == 3 && owned.ppUsageCounts[1]->count == 5;
            }
            return owned.ppUsageCounts == nullptr && owned.pUsageCounts != nullptr
                && owned.pUsageCounts != borrowed.pUsageCounts
                && owned.pUsageCounts[0].count == 3 && owned.pUsageCounts[1].count == 5;
        };

        vku::safe_VkAccelerationStructureGeometryMicromapDataKHR safe(&source);
        Check(ownsIndependentCopy(safe, source),
              "KHR micromap safe struct deep-copies the selected valid usage-count representation");

        usage[0].count = 30;
        usage[1].count = 50;
        Check(ownsIndependentCopy(safe, source),
              "KHR micromap safe struct does not alias source usage counts");

        vku::safe_VkAccelerationStructureGeometryMicromapDataKHR copied(safe);
        vku::safe_VkAccelerationStructureGeometryMicromapDataKHR assigned;
        assigned = safe;
        Check(ownsIndependentCopy(copied, safe) && ownsIndependentCopy(assigned, safe)
                  && ownsIndependentCopy(assigned, copied),
              "KHR micromap safe struct copy and assignment own independent usage counts");

        const VkMicromapUsageKHR replacement {9, 10, VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR};
        const VkMicromapUsageKHR* replacementIndirect[1] {&replacement};
        const VkAccelerationStructureGeometryMicromapDataKHR replacementSource {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR,
            nullptr,
            1,
            useIndirect ? &replacement : nullptr,
            useIndirect ? nullptr : replacementIndirect,
            0x3000,
            0x4000,
            32,
        };
        assigned.initialize(&replacementSource);
        const bool replacementCopied = useIndirect
            ? assigned.ppUsageCounts == nullptr && assigned.pUsageCounts != nullptr
                  && assigned.pUsageCounts != &replacement && assigned.pUsageCounts[0].count == 9
            : assigned.pUsageCounts == nullptr && assigned.ppUsageCounts != nullptr
                  && assigned.ppUsageCounts != replacementIndirect && assigned.ppUsageCounts[0] != nullptr
                  && assigned.ppUsageCounts[0] != &replacement && assigned.ppUsageCounts[0]->count == 9;
        Check(assigned.usageCountsCount == 1 && replacementCopied && assigned.data == 0x3000
                  && assigned.triangleArray == 0x4000 && assigned.triangleArrayStride == 32,
              "KHR micromap safe struct reinitialization switches usage-count representations");
        Check(ownsIndependentCopy(safe, source) && ownsIndependentCopy(copied, safe),
              "KHR micromap reinitialization does not change the other copies");
    }
}

void TestNeuralStatisticsDeviceCapabilityPatch()
{
    driver = {};
    Instance instance(InstanceHandle(), FakeGetInstanceProcAddr);
    const std::string extension = VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME;

    {
        driver.neuralStatisticsFeatureSupported = false;
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM requested {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
            nullptr,
            VK_FALSE,
        };
        const VkDeviceCreateInfo createInfo {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            &requested,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
        };
        vku::safe_VkDeviceCreateInfo safeCreateInfo(&createInfo);
        std::vector<std::string> supported {extension};
        Device::createInfoPatches.back()(instance, PhysicalDeviceHandle(), safeCreateInfo, supported);

        const auto* configured =
            vku::FindStructInPNextChain<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(
                safeCreateInfo.pNext);
        Check(driver.neuralStatisticsFeatureQueries == 1 && HasDeviceExtension(safeCreateInfo, extension.c_str())
                  && configured != nullptr && configured->dataGraphNeuralAcceleratorStatistics == VK_TRUE,
              "enumerated statistics extension is enabled despite a falsely advertised feature bit");
        Check(requested.dataGraphNeuralAcceleratorStatistics == VK_FALSE,
              "feature workaround patches the safe copy without modifying the application's structure");
    }

    {
        driver.neuralStatisticsFeatureSupported = true;
        driver.neuralStatisticsFeatureQueries = 0;
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM requested {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
            nullptr,
            VK_FALSE,
        };
        const VkDeviceCreateInfo createInfo {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            &requested,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
        };
        vku::safe_VkDeviceCreateInfo safeCreateInfo(&createInfo);
        std::vector<std::string> supported {extension};
        Device::createInfoPatches.back()(instance, PhysicalDeviceHandle(), safeCreateInfo, supported);

        const auto* configured =
            vku::FindStructInPNextChain<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(
                safeCreateInfo.pNext);
        Check(driver.neuralStatisticsFeatureQueries == 1 && HasDeviceExtension(safeCreateInfo, extension.c_str())
                  && configured != nullptr && configured->dataGraphNeuralAcceleratorStatistics == VK_TRUE,
              "advertised supported statistics feature is enabled in the device create info");
    }

    {
        driver.neuralStatisticsFeatureSupported = false;
        driver.neuralStatisticsFeatureQueries = 0;
        const VkDeviceCreateInfo createInfo {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        vku::safe_VkDeviceCreateInfo safeCreateInfo(&createInfo);
        std::vector<std::string> supported {extension};
        Device::createInfoPatches.back()(instance, PhysicalDeviceHandle(), safeCreateInfo, supported);

        const auto* configured =
            vku::FindStructInPNextChain<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(
                safeCreateInfo.pNext);
        Check(driver.neuralStatisticsFeatureQueries == 1 && HasDeviceExtension(safeCreateInfo, extension.c_str())
                  && configured != nullptr && configured->dataGraphNeuralAcceleratorStatistics == VK_TRUE,
              "feature workaround inserts an enabled structure when the application provides none");
        Check(createInfo.pNext == nullptr && createInfo.enabledExtensionCount == 0,
              "feature workaround leaves the application's original create info unchanged");
    }

    {
        driver.neuralStatisticsFeatureQueries = 0;
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM requested {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM,
            nullptr,
            VK_FALSE,
        };
        const VkDeviceCreateInfo createInfo {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            &requested,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
        };
        vku::safe_VkDeviceCreateInfo safeCreateInfo(&createInfo);
        std::vector<std::string> supported;
        Device::createInfoPatches.back()(instance, PhysicalDeviceHandle(), safeCreateInfo, supported);

        const auto* configured =
            vku::FindStructInPNextChain<VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(
                safeCreateInfo.pNext);
        Check(driver.neuralStatisticsFeatureQueries == 0 && !HasDeviceExtension(safeCreateInfo, extension.c_str())
                  && configured != nullptr && configured->dataGraphNeuralAcceleratorStatistics == VK_FALSE,
              "unadvertised statistics extension is not queried, enabled, or forced");
    }
}

void TestNeuralStatisticsDeviceCreation()
{
    namespace fs = std::filesystem;
    for (const auto result : {VK_SUCCESS, VK_ERROR_FEATURE_NOT_PRESENT})
    {
        driver = {};
        driver.neuralStatisticsFeatureSupported = false;
        driver.deviceCreateResult = result;
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-device-feature-" + UniqueTestSuffix());
        expectedInstanceRoot = root;
        const std::string rootText = root.string();
        const char* rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo {};
        std::array<VkLayerSettingEXT, 3> settings {};
        VkLayerSettingsCreateInfoEXT settingsInfo {};
        VkLayerInstanceLink instanceLink {};
        VkLayerInstanceCreateInfo instanceLoaderInfo {};
        auto instanceInfo = MakeInstanceCreateInfo(rootValue, applicationInfo, settings, settingsInfo,
                                                   instanceLink, instanceLoaderInfo);
        VkInstance instance = VK_NULL_HANDLE;
        Check(layer_vkCreateInstance<user_tag>(&instanceInfo, nullptr, &instance) == VK_SUCCESS,
              "feature-advertisement test creates the tracked instance");

        VkLayerDeviceLink deviceLink {};
        VkLayerDeviceCreateInfo deviceLoaderInfo {};
        VkDeviceQueueCreateInfo queueInfo {};
        float priority = 0.0f;
        auto deviceInfo = MakeDeviceCreateInfo(deviceLink, deviceLoaderInfo, queueInfo, priority);
        VkDevice device = VK_NULL_HANDLE;
        Check(layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &deviceInfo, nullptr, &device) == result,
              "driver result from attempted statistics enablement is returned unchanged");
        Check(driver.deviceCreates == 1 && driver.neuralStatisticsExtensionEnabled
                  && driver.neuralStatisticsFeatureEnabled,
              "feature workaround attempts one downstream creation with the extension and feature enabled");
        if (result == VK_SUCCESS)
        {
            Check(device == DeviceHandle() && Device::retrieve(device)->neuralStatisticsEnabled,
                  "accepted enablement publishes a capturable logical device despite the false feature query");
            Check(Instance::retrieve(instance)->captureMetadata.warnings.empty(),
                  "accepted feature workaround does not report the device as noncapturable");
            layer_vkDestroyDevice<user_tag>(device, nullptr);
        }
        else
        {
            Check(device == VK_NULL_HANDLE && Instance::retrieve(instance)->captureMetadata.devices.empty(),
                  "rejected enablement does not fabricate or publish a logical device");
        }
        layer_vkDestroyInstance<user_tag>(instance, nullptr);
        std::error_code error;
        fs::remove_all(root, error);
    }
}

void TestInstanceCreateObservations()
{
    namespace fs = std::filesystem;
    const fs::path root =
        fs::temp_directory_path() / ("neural-statistics-instance-observations-" + UniqueTestSuffix());
    expectedInstanceRoot = root;
    VkInstanceCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance application = reinterpret_cast<VkInstance>(uintptr_t {0xBEEF});
    std::error_code error;
    fs::create_directories(root);

    for (const uint32_t probeCount : {0u, 2u, 7u})
    {
        driver = {};
        driver.instanceCreateResult = VK_ERROR_INCOMPATIBLE_DRIVER;
        driver.applicationInstanceOutput = &application;
        for (uint32_t i = 0; i < probeCount; ++i)
        {
            VkInstance temporary = VK_NULL_HANDLE;
            FakeCreateInstance(&info, nullptr, &temporary);
        }
        FakeCreateInstance(&info, nullptr, &application);
        Check(driver.instanceCreates == probeCount + 1 && driver.applicationInstanceCreates == 1
                  && driver.auxiliaryInstanceCreates == probeCount,
              "instance observations distinguish one application attempt from a variable number of probes");
        Check(driver.writerRootObservedBeforeEveryInstanceCreate,
              "instance observations accept a root present before every call");
    }

    driver = {};
    driver.instanceCreateResult = VK_ERROR_INCOMPATIBLE_DRIVER;
    driver.applicationInstanceOutput = &application;
    fs::remove_all(root, error);
    VkInstance temporary = VK_NULL_HANDLE;
    FakeCreateInstance(&info, nullptr, &temporary);
    fs::create_directories(root);
    FakeCreateInstance(&info, nullptr, &application);
    Check(!driver.writerRootObservedBeforeEveryInstanceCreate,
          "a later root claim cannot hide an earlier downstream call before writer startup");
    FakeCreateInstance(&info, nullptr, &application);
    Check(driver.applicationInstanceCreates == 2 && driver.auxiliaryInstanceCreates == 1,
          "a duplicate application creation cannot be counted as a version probe");

    fs::remove_all(root, error);
    expectedInstanceRoot.clear();
    driver = {};
}

void TestInstanceWriterAndMultipleDevices()
{
    namespace fs = std::filesystem;

    {
        driver = {};
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-instance-collision-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        fs::create_directories(root, error);
        const std::string rootText = root.string();
        const char *rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo{};
        std::array<VkLayerSettingEXT, 3> settings{};
        VkLayerSettingsCreateInfoEXT settingsInfo{};
        VkLayerInstanceLink link{};
        VkLayerInstanceCreateInfo loaderInfo{};
        auto createInfo = MakeInstanceCreateInfo(rootValue, applicationInfo, settings, settingsInfo, link, loaderInfo);
        Check(LayerOptions(&createInfo).getCaptureRoot() == root,
              "instance harness API settings resolve the requested capture root");
        const VkInstance provisionalInstance = reinterpret_cast<VkInstance>(uintptr_t {0xDEAD});
        VkInstance output = provisionalInstance;
        Check(layer_vkCreateInstance<user_tag>(&createInfo, nullptr, &output) == VK_ERROR_INITIALIZATION_FAILED
                  && driver.instanceCreates == 0,
              "writer startup/root-claim failure rejects before downstream vkCreateInstance");
        Check(output == provisionalInstance,
              "pre-forward instance failure preserves the loader's provisional output handle");
        fs::remove_all(root, error);
    }

    {
        driver = {};
        driver.instanceCreateResult = VK_ERROR_INCOMPATIBLE_DRIVER;
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-instance-downstream-fail-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        expectedInstanceRoot = root;
        const std::string rootText = root.string();
        const char *rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo{};
        std::array<VkLayerSettingEXT, 3> settings{};
        VkLayerSettingsCreateInfoEXT settingsInfo{};
        VkLayerInstanceLink link{};
        VkLayerInstanceCreateInfo loaderInfo{};
        auto createInfo = MakeInstanceCreateInfo(rootValue, applicationInfo, settings, settingsInfo, link, loaderInfo);
        const VkInstance provisionalInstance = reinterpret_cast<VkInstance>(uintptr_t {0xBEEF});
        VkInstance output = provisionalInstance;
        driver.applicationInstanceOutput = &output;
        Check(layer_vkCreateInstance<user_tag>(&createInfo, nullptr, &output) == VK_ERROR_INCOMPATIBLE_DRIVER,
              "downstream instance failure is preserved exactly");
        Check(driver.applicationInstanceCreates == 1,
              "downstream failure attempts application instance creation exactly once");
        Check(driver.writerRootObservedBeforeEveryInstanceCreate,
              "capture writer claims its root before every downstream instance creation, including probes");
        Check(output == provisionalInstance,
              "downstream instance failure preserves the loader's provisional output handle");
        std::ifstream captureFile(root / "capture.json");
        const auto document = nlohmann::json::parse(captureFile, nullptr, false);
        Check(document.is_object() && document.value("status", "") == "error",
              "downstream instance failure drains a coherent error capture");
        fs::remove_all(root, error);
    }

    for (const bool omitApplicationInfo : {true, false})
    {
        driver = {};
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-default-application-info-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        expectedInstanceRoot = root;
        const std::string rootText = root.string();
        const char* rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo {};
        std::array<VkLayerSettingEXT, 3> settings {};
        VkLayerSettingsCreateInfoEXT settingsInfo {};
        VkLayerInstanceLink link {};
        VkLayerInstanceCreateInfo loaderInfo {};
        auto createInfo = MakeInstanceCreateInfo(rootValue, applicationInfo, settings, settingsInfo, link, loaderInfo);
        applicationInfo.apiVersion = 0;
        createInfo.pApplicationInfo = omitApplicationInfo ? nullptr : &applicationInfo;
        Check(getApplicationAPIVersion(&createInfo) == APIVersion{1, 0},
              "missing application info or a zero API version defaults to Vulkan 1.0");

        VkInstance instance = VK_NULL_HANDLE;
        Check(layer_vkCreateInstance<user_tag>(&createInfo, nullptr, &instance) == VK_SUCCESS
                  && instance == InstanceHandle(),
              "instance creation accepts a missing application info or zero API version");
        Check(driver.lastInstanceCreateHadApplicationInfo && driver.lastInstanceCreateApiVersion == VK_API_VERSION_1_1,
              "common create path provides valid promoted application info to the driver");
        Check(createInfo.pApplicationInfo == (omitApplicationInfo ? nullptr : &applicationInfo)
                  && applicationInfo.apiVersion == 0,
              "API version promotion leaves the application's create info unchanged");

        layer_vkDestroyInstance<user_tag>(instance, nullptr);
        fs::remove_all(root, error);
    }

    {
        driver = {};
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-instance-device-policy-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        expectedInstanceRoot = root;
        const std::string rootText = root.string();
        const char *rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo{};
        std::array<VkLayerSettingEXT, 3> settings{};
        VkLayerSettingsCreateInfoEXT settingsInfo{};
        VkLayerInstanceLink instanceLink{};
        VkLayerInstanceCreateInfo instanceLoaderInfo{};
        auto instanceInfo = MakeInstanceCreateInfo(rootValue,
                                                   applicationInfo,
                                                   settings,
                                                   settingsInfo,
                                                   instanceLink,
                                                   instanceLoaderInfo);
        VkInstance instance = VK_NULL_HANDLE;
        Check(layer_vkCreateInstance<user_tag>(&instanceInfo, nullptr, &instance) == VK_SUCCESS
                  && instance == InstanceHandle(),
              "actual instance user hook creates and publishes the tracked instance");

        VkLayerDeviceLink deviceLink{};
        VkLayerDeviceCreateInfo deviceLoaderInfo{};
        VkDeviceQueueCreateInfo queueInfo{};
        float priority = 0.0f;
        auto deviceInfo = MakeDeviceCreateInfo(deviceLink, deviceLoaderInfo, queueInfo, priority);

        driver.neuralStatisticsFeatureSupported = false;
        VkDevice firstDevice = VK_NULL_HANDLE;
        Check(layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &deviceInfo, nullptr, &firstDevice) == VK_SUCCESS
                  && firstDevice == DeviceHandle(),
              "first logical device is created and published");
        Check(driver.neuralStatisticsExtensionEnabled && driver.neuralStatisticsFeatureEnabled
                  && Device::retrieve(firstDevice)->neuralStatisticsEnabled,
              "falsely unadvertised statistics feature reaches device creation and is tracked as enabled");

        VkLayerDeviceLink secondDeviceLink{};
        VkLayerDeviceCreateInfo secondDeviceLoaderInfo{};
        VkDeviceQueueCreateInfo secondQueueInfo{};
        float secondPriority = 0.0f;
        auto secondDeviceInfo =
            MakeDeviceCreateInfo(secondDeviceLink, secondDeviceLoaderInfo, secondQueueInfo, secondPriority);
        driver.exposeNeuralStatisticsExtension = false;
        VkDevice secondDevice = VK_NULL_HANDLE;
        Check(layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &secondDeviceInfo, nullptr, &secondDevice)
                      == VK_SUCCESS
                  && secondDevice == DeviceHandle1() && driver.deviceCreates == 2,
              "second simultaneously-live logical device is created and published");
        Check(Device::retrieve(firstDevice)->captureDeviceId == capture::LogicalDeviceId(0)
                  && Device::retrieve(secondDevice)->captureDeviceId == capture::LogicalDeviceId(1),
              "logical devices receive distinct capture-local identities");
        Check(driver.neuralStatisticsFeatureQueries == 1 && !Device::retrieve(secondDevice)->neuralStatisticsEnabled,
              "missing statistics extension remains noncapturable independently per logical device");
        Check(driver.requirements2ExtensionEnabled,
              "actual Vulkan-1.0 device capability enumeration enables KHR requirements2");

        layer_vkDestroyDevice<user_tag>(secondDevice, nullptr);
        layer_vkDestroyDevice<user_tag>(firstDevice, nullptr);
        layer_vkDestroyInstance<user_tag>(instance, nullptr);

        std::ifstream captureFile(root / "capture.json");
        const auto document = nlohmann::json::parse(captureFile, nullptr, false);
        Check(document.is_object() && document.value("status", "") == "complete"
                  && document.value("schema_version", 0) == 2 && document.at("devices").size() == 2
                  && document.at("devices").at(0).at("id") == 0 && document.at("devices").at(1).at("id") == 1
                  && document.at("warnings").size() == 1
                  && document.at("warnings").at(0) == "neural statistics capture is unavailable for logical device 1",
              "complete capture reports a logical device whose statistics capability is unavailable");
        fs::remove_all(root, error);
    }

    for (const auto [point, failure] : {
             std::pair{capture::fault::Point::DeviceAfterDownstreamCreateBeforePublication,
                       capture::fault::Failure::BadAllocation},
             std::pair{capture::fault::Point::DeviceAfterDownstreamCreateBeforePublication,
                       capture::fault::Failure::Unexpected},
             std::pair{capture::fault::Point::DeviceBeforeStore, capture::fault::Failure::BadAllocation},
             std::pair{capture::fault::Point::DeviceBeforeStore, capture::fault::Failure::Unexpected}})
    {
        driver = {};
        driver.probeDeviceDestroyLock = true;
        const VkResult expectedResult = failure == capture::fault::Failure::BadAllocation
            ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_ERROR_INITIALIZATION_FAILED;
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-device-publication-fail-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        expectedInstanceRoot = root;
        const std::string rootText = root.string();
        const char* rootValue = rootText.c_str();
        VkApplicationInfo applicationInfo {};
        std::array<VkLayerSettingEXT, 3> settings {};
        VkLayerSettingsCreateInfoEXT settingsInfo {};
        VkLayerInstanceLink instanceLink {};
        VkLayerInstanceCreateInfo instanceLoaderInfo {};
        auto instanceInfo = MakeInstanceCreateInfo(rootValue,
                                                   applicationInfo,
                                                   settings,
                                                   settingsInfo,
                                                   instanceLink,
                                                   instanceLoaderInfo);
        VkInstance instance = VK_NULL_HANDLE;
        Check(layer_vkCreateInstance<user_tag>(&instanceInfo, nullptr, &instance) == VK_SUCCESS,
              "device-publication rollback setup creates the tracked instance");

        VkLayerDeviceLink deviceLink {};
        VkLayerDeviceCreateInfo deviceLoaderInfo {};
        VkDeviceQueueCreateInfo queueInfo {};
        float priority = 0.0f;
        auto deviceInfo = MakeDeviceCreateInfo(deviceLink, deviceLoaderInfo, queueInfo, priority);
        VkDevice device = reinterpret_cast<VkDevice>(uintptr_t {0xD00D});
        {
            capture::fault::ScopedInjection injection(point, failure);
            Check(layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &deviceInfo, nullptr, &device)
                      == expectedResult,
                  "device construction or publication failure is translated at the ABI boundary");
        }
        Check(device == VK_NULL_HANDLE && driver.deviceCreates == 1 && driver.deviceDestroys == 1
                  && driver.deviceDestroyLockAvailable
                  && Instance::retrieve(instance)->captureMetadata.devices.empty()
                  && Instance::retrieve(instance)->isCaptureEnabled(),
              "failed device publication destroys the downstream device, clears output, and leaves capture coherent");

        layer_vkDestroyInstance<user_tag>(instance, nullptr);
        std::ifstream captureFile(root / "capture.json");
        const auto document = nlohmann::json::parse(captureFile, nullptr, false);
        Check(document.is_object() && document.value("status", "") == "complete" && document.at("devices").empty(),
              "device publication rollback leaves a complete zero-device capture");
        fs::remove_all(root, error);
    }

    expectedInstanceRoot.clear();
}

VkShaderModuleCreateInfo ShaderInfo()
{
    static const uint32_t code[]{0x07230203, 0x00010000, 0, 1};
    return {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, sizeof(code), code};
}

VkDataGraphPipelineCreateInfoARM PipelineInfo(const void *next = nullptr)
{
    return {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM, next, 0, VK_NULL_HANDLE, 0, nullptr};
}

void TestProtectedPipelinePassThrough()
{
    LiveFixture fixture;
    auto info = PipelineInfo();
    info.flags = VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT_EXT;
    driver.pipelineOutputs = {1};

    VkPipeline pipeline = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                        VK_NULL_HANDLE,
                                                        VK_NULL_HANDLE,
                                                        1,
                                                        &info,
                                                        nullptr,
                                                        &pipeline)
                  == VK_SUCCESS
              && pipeline != VK_NULL_HANDLE,
          "protected-only pipeline creation is preserved");
    const auto pipelineRecord = fixture.device()->resourceManager.getPipelineRecord(pipeline);
    Check(driver.lastPipelineCreateInfoObserved && driver.lastPipelineCreateInfo.flags == info.flags
              && driver.lastPipelineCreateInfo.pNext == info.pNext && !driver.lastPipelineCreateHadNeuralStatistics,
          "protected-only pipeline is forwarded without the forbidden neural-statistics pNext");
    Check(pipelineRecord != nullptr && !pipelineRecord->capturable && !pipelineRecord->metadata.statisticsEnabled
              && pipelineRecord->unsupportedReason == "selected capture rejects protected data-graph pipelines",
          "protected-only pipeline is tracked as explicitly noncapturable");

    const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
        nullptr,
        0,
        pipeline,
    };
    VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS
              && session != VK_NULL_HANDLE && driver.originalSessionCreates == 1
              && driver.instrumentedSessionCreates == 0,
          "protected-only pipeline session is forwarded without statistics instrumentation");
    const auto sessionRecord = fixture.device()->resourceManager.getSessionRecord(session);
    Check(sessionRecord != nullptr && !sessionRecord->capturable && !sessionRecord->statisticsCaptureActive,
          "protected-only session retains a noncapturable placeholder");

    fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
    const auto command =
        fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
    const VkDataGraphPipelineDispatchInfoARM dispatchInfo {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM,
        nullptr,
        0,
    };
    layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
    Check(driver.cmdDispatchDataGraphCalls == 1 && driver.cmdFillBufferCalls == 0 && driver.cmdPipelineBarrierCalls == 0
              && driver.cmdCopyBufferCalls == 0 && command->dispatches.size() == 1
              && !command->dispatches.front().unsupportedReason.empty(),
          "protected-only dispatch records countable noncapture metadata without injected GPU commands");
    const VkCommandBuffer submittedCommandBuffer = CommandBufferHandle();
    const VkSubmitInfo
        submit {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &submittedCommandBuffer, 0, nullptr};
    Check(layer_vkQueueSubmit<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS
              && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
              && !fixture.instance()->isCaptureEnabled() && sessionRecord->nextExecutionIndex == 0,
          "selected protected workload terminalizes capture but forwards the original submit unchanged");

    layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
}

void TestUnsupportedNeuralStatisticsPassThrough()
{
    LiveFixture
        fixture(true, true, true, true, 64, std::vector<VkQueueFamilyProperties> {}, std::vector<uint32_t> {}, false);
    Check(!fixture.device()->neuralStatisticsEnabled,
          "device without enabled statistics feature retains an unavailable capability gate");

    VkPipelineCreationFeedback feedback {};
    VkPipelineCreationFeedbackCreateInfo feedbackInfo {
        VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
        nullptr,
        &feedback,
        0,
        nullptr,
    };
    const VkDataGraphPipelineCreateInfoARM info = PipelineInfo(&feedbackInfo);
    const uint64_t nextPipelineId = fixture.instance()->pipelineIds.nextValue();
    driver.pipelineOutputs = {1};
    VkPipeline pipeline = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                        VK_NULL_HANDLE,
                                                        VK_NULL_HANDLE,
                                                        1,
                                                        &info,
                                                        nullptr,
                                                        &pipeline)
                  == VK_SUCCESS
              && pipeline != VK_NULL_HANDLE,
          "unsupported statistics device preserves downstream pipeline creation");
    Check(driver.lastPipelineCreateInfoObserved && std::memcmp(&driver.lastPipelineCreateInfo, &info, sizeof(info)) == 0
              && driver.lastPipelineCreateHadPNext && !driver.lastPipelineCreateHadNeuralStatistics,
          "unsupported statistics device forwards the application pipeline create info byte-for-byte");
    Check(fixture.device()->resourceManager.getPipelineRecord(pipeline) == nullptr
              && fixture.instance()->pipelineIds.nextValue() == nextPipelineId,
          "unsupported statistics pipeline is not tracked or assigned a capture identity");

    const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
        nullptr,
        0,
        pipeline,
    };
    VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS
              && session != VK_NULL_HANDLE && driver.originalSessionCreates == 1
              && driver.instrumentedSessionCreates == 0
              && fixture.device()->resourceManager.getSessionRecord(session) == nullptr,
          "session for an untracked pipeline is forwarded without statistics instrumentation");

    layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
}

void TestShaderTransactions()
{
    LiveFixture fixture;
    const auto info = ShaderInfo();

    for (const auto point : {
             capture::fault::Point::ShaderRecordBeforeInsertion,
             capture::fault::Point::ShaderMapInsertion,
             capture::fault::Point::ShaderRecordAfterInsertionBeforeCommit,
         })
    {
        const uint64_t nextId = fixture.instance()->shaderModuleIds.nextValue();
        const uint32_t destroys = driver.shaderDestroys;
        VkShaderModule shader = reinterpret_cast<VkShaderModule>(uintptr_t{0xDEAD});
        capture::fault::ScopedInjection injection(point);
        const VkResult result = layer_vkCreateShaderModule<user_tag>(DeviceHandle(), &info, nullptr, &shader);
        Check(result == VK_ERROR_OUT_OF_HOST_MEMORY,
              "shader allocation/map/post-insertion failure translates to host-memory error");
        Check(shader == VK_NULL_HANDLE && driver.shaderDestroys == destroys + 1,
              "failed shader publication destroys downstream object and nulls output");
        Check(fixture.instance()->shaderModuleIds.nextValue() == nextId,
              "failed shader publication does not commit its ID");
    }

    driver.throwShaderCreate = true;
    VkShaderModule shader = VK_NULL_HANDLE;
    Check(layer_vkCreateShaderModule<user_tag>(DeviceHandle(), &info, nullptr, &shader) ==
              VK_ERROR_UNKNOWN,
          "shader hook translates an unexpected downstream exception at the ABI boundary");
    driver.throwShaderCreate = false;

    Check(layer_vkCreateShaderModule<user_tag>(DeviceHandle(), &info, nullptr, &shader) ==
              VK_SUCCESS,
          "shader hook success remains application-observable");
    Check(fixture.device()->resourceManager.getShaderModuleRecord(shader) != nullptr,
          "successful shader is tracked by the production hook");
    layer_vkDestroyShaderModule<user_tag>(DeviceHandle(), shader, nullptr);
}

void TestCommandBufferPublicationTransactions()
{
    LiveFixture fixture;
    const VkCommandPoolCreateInfo poolInfo {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0,
        0,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    Check(layer_vkCreateCommandPool<user_tag>(DeviceHandle(), &poolInfo, nullptr, &pool) == VK_SUCCESS
              && pool != VK_NULL_HANDLE,
          "command-buffer publication test creates a tracked pool");

    const VkCommandBufferAllocateInfo allocateInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    const uint32_t freesBefore = driver.commandBufferFrees;
    VkCommandBuffer commandBuffer = reinterpret_cast<VkCommandBuffer>(uintptr_t {0xDEAD});
    {
        capture::fault::ScopedInjection injection(capture::fault::Point::CommandBufferMapInsertion);
        Check(layer_vkAllocateCommandBuffers<user_tag>(DeviceHandle(), &allocateInfo, &commandBuffer)
                  == VK_ERROR_OUT_OF_HOST_MEMORY,
              "command-buffer map allocation failure translates to host-memory error");
    }

    const auto poolRecord = fixture.device()->resourceManager.getCommandPoolRecord(pool);
    Check(commandBuffer == VK_NULL_HANDLE && driver.commandBufferFrees == freesBefore + 1,
          "failed command-buffer publication frees and nulls the downstream handle");
    Check(poolRecord != nullptr && poolRecord->cbufferMap.empty(),
          "failed command-buffer publication rolls back provisional pool bookkeeping");
    layer_vkDestroyCommandPool<user_tag>(DeviceHandle(), pool, nullptr);
}

void TestTerminalCaptureObjectPassThrough()
{
    LiveFixture fixture;
    Check(fixture.instance()->disableCapture("intentional terminal object pass-through test"),
          "terminal object pass-through setup disables capture");

    const auto shaderInfo = ShaderInfo();
    const uint64_t nextShaderId = fixture.instance()->shaderModuleIds.nextValue();
    const uint32_t shaderDestroysBefore = driver.shaderDestroys;
    VkShaderModule shader = VK_NULL_HANDLE;
    {
        capture::fault::ScopedInjection injection(capture::fault::Point::ShaderMapInsertion);
        Check(layer_vkCreateShaderModule<user_tag>(DeviceHandle(), &shaderInfo, nullptr, &shader) == VK_SUCCESS,
              "terminal shader creation bypasses capture-only publication faults");
    }
    Check(shader != VK_NULL_HANDLE && driver.shaderDestroys == shaderDestroysBefore
              && fixture.device()->resourceManager.getShaderModuleRecord(shader) == nullptr
              && fixture.instance()->shaderModuleIds.nextValue() == nextShaderId,
          "terminal shader creation preserves the downstream handle without capture bookkeeping");
    layer_vkDestroyShaderModule<user_tag>(DeviceHandle(), shader, nullptr);

    const VkCommandPoolCreateInfo poolInfo {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0,
        0,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    Check(layer_vkCreateCommandPool<user_tag>(DeviceHandle(), &poolInfo, nullptr, &pool) == VK_SUCCESS
              && pool != VK_NULL_HANDLE && fixture.device()->resourceManager.getCommandPoolRecord(pool) == nullptr,
          "terminal command-pool creation forwards untracked");
    const VkCommandBufferAllocateInfo allocateInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    VkCommandBuffer allocated = VK_NULL_HANDLE;
    Check(layer_vkAllocateCommandBuffers<user_tag>(DeviceHandle(), &allocateInfo, &allocated) == VK_SUCCESS
              && allocated != VK_NULL_HANDLE
              && fixture.device()->resourceManager.getCommandBufferRecord(allocated) == nullptr,
          "terminal command-buffer allocation forwards untracked");
    const VkCommandBufferBeginInfo beginInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        0,
        nullptr,
    };
    Check(layer_vkBeginCommandBuffer<user_tag>(CommandBufferHandle(), &beginInfo) == VK_SUCCESS
              && layer_vkResetCommandBuffer<user_tag>(CommandBufferHandle(), 0) == VK_SUCCESS,
          "terminal command-buffer begin/reset bypass missing capture records");
    layer_vkDestroyCommandPool<user_tag>(DeviceHandle(), pool, nullptr);

    VkLayerDeviceLink deviceLink {};
    VkLayerDeviceCreateInfo deviceLoaderInfo {};
    VkDeviceQueueCreateInfo queueInfo {};
    float priority = 0.0f;
    auto deviceInfo = MakeDeviceCreateInfo(deviceLink, deviceLoaderInfo, queueInfo, priority);
    VkPhysicalDeviceSynchronization2Features applicationSynchronization2 {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        nullptr,
        VK_FALSE,
    };
    deviceLoaderInfo.pNext = &applicationSynchronization2;
    const uint32_t extensionQueriesBefore = driver.deviceExtensionQueries;
    const uint32_t featureQueriesBefore = driver.physicalDeviceFeatureQueries;
    const uint32_t propertyQueriesBefore = driver.physicalDevicePropertyQueries;
    const uint32_t queueQueriesBefore = driver.queueFamilyPropertyQueries;
    driver.deviceCreates = 1;
    VkDevice secondDevice = VK_NULL_HANDLE;
    VkResult createResult = VK_ERROR_UNKNOWN;
    {
        capture::fault::ScopedInjection injection(capture::fault::Point::DeviceAfterDownstreamCreateBeforePublication);
        createResult = layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &deviceInfo, nullptr, &secondDevice);
    }
    Check(createResult == VK_SUCCESS && secondDevice == DeviceHandle1() && driver.deviceCreates == 2,
          "terminal capture forwards a later valid logical-device creation");
    // The framework still enumerates extensions and queries the device API version.
    Check(driver.deviceExtensionQueries == extensionQueriesBefore + 2
              && driver.physicalDeviceFeatureQueries == featureQueriesBefore
              && driver.physicalDevicePropertyQueries == propertyQueriesBefore + 1
              && driver.queueFamilyPropertyQueries == queueQueriesBefore,
          "terminal logical-device creation performs only framework-required capability queries");
    Check(driver.lastDeviceCreateEnabledExtensionCount == 0 && !driver.requirements2ExtensionEnabled
              && !driver.neuralStatisticsExtensionEnabled && !driver.neuralStatisticsFeatureEnabled
              && !driver.synchronization2FeatureEnabled && applicationSynchronization2.synchronization2 == VK_FALSE,
          "terminal logical-device creation forwards application extensions and features unchanged");
    Device* terminalDevice = Device::retrieve(secondDevice);
    Check(terminalDevice != nullptr && terminalDevice->collector == nullptr
              && terminalDevice->queueFamilyProperties.empty() && !terminalDevice->neuralStatisticsEnabled
              && !terminalDevice->synchronization2Enabled && fixture.instance()->captureMetadata.devices.empty(),
          "terminal logical device retains dispatch tracking without capture-only state or metadata");
    layer_vkDestroyDevice<user_tag>(secondDevice, nullptr);
}

void TestPipelineTransactionsAndPartialResults()
{
    LiveFixture fixture;
    std::array<VkDataGraphPipelineCreateInfoARM, 3> infos {PipelineInfo(), PipelineInfo(), PipelineInfo()};

    driver.pipelineOutputs = {1, 1, 1};
    for (const auto point : {
             capture::fault::Point::PipelineRecordBeforeInsertion,
             capture::fault::Point::PipelineMapInsertion,
             capture::fault::Point::PipelineRecordAfterInsertionBeforeCommit,
         })
    {
        const uint64_t nextId = fixture.instance()->pipelineIds.nextValue();
        const uint32_t destroys = driver.pipelineDestroys;
        std::array<VkPipeline, 3> outputs{};
        capture::fault::ScopedInjection injection(point, capture::fault::Failure::BadAllocation, 2);
        const VkResult result = layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                                              VK_NULL_HANDLE,
                                                                              VK_NULL_HANDLE,
                                                                              3,
                                                                              infos.data(),
                                                                              nullptr,
            outputs.data());
        Check(result == VK_ERROR_OUT_OF_HOST_MEMORY,
              "second/later pipeline record failure translates to host-memory error");
        Check(outputs[0] == VK_NULL_HANDLE && outputs[1] == VK_NULL_HANDLE && outputs[2] == VK_NULL_HANDLE
                  && driver.pipelineDestroys == destroys + 3,
              "multi-pipeline rollback destroys every downstream handle and nulls every output");
        Check(fixture.instance()->pipelineIds.nextValue() == nextId, "multi-pipeline rollback commits no partial IDs");
    }

    driver.pipelineResult = VK_PIPELINE_COMPILE_REQUIRED_EXT;
    driver.pipelineOutputs = {1, 0, 1};
    std::array<VkPipeline, 3> compileRequired{};
    const uint64_t firstId = fixture.instance()->pipelineIds.nextValue();
    Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                        VK_NULL_HANDLE,
                                                        VK_NULL_HANDLE,
                                                        3,
                                                        infos.data(),
                                                        nullptr,
                                                        compileRequired.data())
              == VK_PIPELINE_COMPILE_REQUIRED_EXT,
          "compile-required result is preserved exactly");
    Check(compileRequired[0] != VK_NULL_HANDLE && compileRequired[1] == VK_NULL_HANDLE
              && compileRequired[2] != VK_NULL_HANDLE,
          "compile-required legal sparse output pattern is preserved for the application");
    Check(fixture.device()->resourceManager.getPipelineRecord(compileRequired[0])->id == capture::PipelineId(firstId)
              && fixture.device()->resourceManager.getPipelineRecord(compileRequired[2])->id
                     == capture::PipelineId(firstId + 1),
          "only published compile-required handles receive stable contiguous IDs");
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), compileRequired[0], nullptr);
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), compileRequired[2], nullptr);

    driver.pipelineResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    driver.pipelineOutputs = {1, 0, 0};
    std::array<VkPipeline, 3> partialError{};
    Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                        VK_NULL_HANDLE,
                                                        VK_NULL_HANDLE,
                                                        3,
                                                        infos.data(),
                                                        nullptr,
                                                        partialError.data())
              == VK_ERROR_OUT_OF_DEVICE_MEMORY,
          "downstream pipeline error result is preserved when an earlier handle is valid");
    Check(partialError[0] != VK_NULL_HANDLE
              && fixture.device()->resourceManager.getPipelineRecord(partialError[0]) != nullptr,
          "valid partial-error pipeline output is application-observable and tracked");
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), partialError[0], nullptr);
}

void TestPostIdentityPipelinePublicationFailure()
{
    namespace fs = std::filesystem;
    const fs::path root =
        fs::temp_directory_path() / ("neural-statistics-pipeline-publication-failure-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        driver.exposeStaticPipelineProperties = true;
        driver.pipelineOutputs = {1, 1};
        std::array<VkDataGraphPipelineCreateInfoARM, 2> infos{PipelineInfo(), PipelineInfo()};
        std::array<VkPipeline, 2> outputs{};
        const uint64_t nextId = fixture.instance()->pipelineIds.nextValue();
        const uint32_t destroys = driver.pipelineDestroys;

        capture::fault::ScopedInjection injection(capture::fault::Point::WriterPipelineArtifactSubmission,
                                                  capture::fault::Failure::Unexpected,
                                                  2);
        const VkResult result = layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                                              VK_NULL_HANDLE,
                                                                              VK_NULL_HANDLE,
                                                                              static_cast<uint32_t>(infos.size()),
                                                                              infos.data(),
                                                                              nullptr,
                                                                              outputs.data());

        Check(result == VK_ERROR_UNKNOWN,
              "post-identity static-artifact rejection is contained at the pipeline ABI boundary");
        Check(outputs[0] == VK_NULL_HANDLE && outputs[1] == VK_NULL_HANDLE && driver.pipelineDestroys == destroys + 2,
              "post-identity failure destroys and nulls every downstream batch handle");
        Check(driver.lastPipelineHandles.size() == 2
                  && fixture.device()->resourceManager.getPipelineRecord(driver.lastPipelineHandles[0]) == nullptr
                  && fixture.device()->resourceManager.getPipelineRecord(driver.lastPipelineHandles[1]) == nullptr,
              "post-identity failure removes every pipeline mapping");
        Check(fixture.instance()->pipelineIds.nextValue() == nextId,
              "post-identity failure leaves pipeline IDs uncommitted");

        const auto state = fixture.instance()->captureStateSnapshot();
        Check(!state.enabled && state.terminalRequestIssued && state.terminalTransitionCount == 1
                  && state.failure == "static pipeline artifact writer job was rejected during object creation",
              "post-identity failure records exactly one coherent first terminal diagnostic");
        Check(fixture.instance()->captureWriter->snapshot().acceptedJobs >= 3,
              "pipeline identity and one static artifact are accepted before terminal publication");

        const uint32_t creates = driver.pipelineCreates;
        VkPipeline later = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                            VK_NULL_HANDLE,
                                                            VK_NULL_HANDLE,
                                                            1,
                                                            infos.data(),
                                                            nullptr,
                                                            &later)
                      == VK_SUCCESS
                  && later != VK_NULL_HANDLE && driver.pipelineCreates == creates + 1
                  && !driver.lastPipelineCreateHadPNext
                  && fixture.device()->resourceManager.getPipelineRecord(later) == nullptr
                  && fixture.instance()->pipelineIds.nextValue() == nextId,
              "terminal capture forwards later pipeline creation without capture instrumentation");
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), later, nullptr);
    }

    std::ifstream captureFile(root / "capture.json");
    const auto document = nlohmann::json::parse(captureFile, nullptr, false);
    Check(document.is_object() && document.value("status", "") == "error",
          "post-identity pipeline failure drains a terminal error capture");
    fs::remove_all(root, cleanupError);
}

VkPipeline CreateTrackedPipeline(LiveFixture &fixture, bool capturable)
{
    VkDataGraphPipelineNeuralStatisticsCreateInfoARM applicationStatistics{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_NEURAL_STATISTICS_CREATE_INFO_ARM,
        nullptr,
        VK_TRUE,
    };
    const auto info = PipelineInfo(capturable ? nullptr : &applicationStatistics);
    driver.pipelineResult = VK_SUCCESS;
    driver.pipelineOutputs = {1};
    VkPipeline pipeline = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                        VK_NULL_HANDLE,
                                                        VK_NULL_HANDLE,
                                                        1,
                                                        &info,
                                                        nullptr,
                                                        &pipeline)
              == VK_SUCCESS,
          "pipeline setup succeeds");
    return pipeline;
}

void TestMidCallTerminalCaptureObjectPassThrough()
{
    {
        LiveFixture fixture;
        const auto info = PipelineInfo();
        driver.pipelineOutputs = {1};
        const uint64_t nextPipelineId = fixture.instance()->pipelineIds.nextValue();
        const uint32_t destroysBefore = driver.pipelineDestroys;
        terminalizeCaptureDuringPipelineCreate = fixture.instance();

        VkPipeline pipeline = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelinesARM<user_tag>(DeviceHandle(),
                                                            VK_NULL_HANDLE,
                                                            VK_NULL_HANDLE,
                                                            1,
                                                            &info,
                                                            nullptr,
                                                            &pipeline)
                      == VK_SUCCESS
                  && pipeline != VK_NULL_HANDLE,
              "capture shutdown inside downstream pipeline creation preserves the published handle");
        Check(!fixture.instance()->isCaptureEnabled() && driver.lastPipelineCreateHadNeuralStatistics
                  && driver.pipelineDestroys == destroysBefore
                  && fixture.device()->resourceManager.getPipelineRecord(pipeline) == nullptr
                  && fixture.instance()->pipelineIds.nextValue() == nextPipelineId,
              "mid-call pipeline shutdown abandons capture bookkeeping without rolling back application work");
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM info {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        const uint64_t nextSessionId = fixture.instance()->sessionIds.nextValue();
        const uint32_t instrumentedCreates = driver.instrumentedSessionCreates;
        const uint32_t originalCreates = driver.originalSessionCreates;
        const uint32_t destroysBefore = driver.sessionDestroys;
        const uint32_t buffersBefore = driver.bufferCreates;
        terminalizeCaptureDuringSessionCreate = fixture.instance();

        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &info, nullptr, &session)
                      == VK_SUCCESS
                  && session != VK_NULL_HANDLE,
              "capture shutdown inside downstream session creation falls back to the original session");
        Check(!fixture.instance()->isCaptureEnabled() && driver.instrumentedSessionCreates == instrumentedCreates + 1
                  && driver.originalSessionCreates == originalCreates + 1
                  && driver.sessionDestroys == destroysBefore + 1 && driver.bufferCreates == buffersBefore
                  && fixture.device()->resourceManager.getSessionRecord(session) == nullptr
                  && fixture.instance()->sessionIds.nextValue() == nextSessionId,
              "mid-call session shutdown destroys capture-only state and publishes no hidden bookkeeping");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }
}
void TestSessionFallbackAndRequirements2Route()
{
    {
        LiveFixture fixture(false, true);
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM info{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };

        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        capture::fault::ScopedInjection injection(capture::fault::Point::SessionMapInsertion);
        const VkResult result =
            layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &info, nullptr, &session);
        Check(result == VK_SUCCESS && session != VK_NULL_HANDLE,
              "capture-only session allocation failure falls back to an original legal session");
        const auto record = fixture.device()->resourceManager.getSessionRecord(session);
        Check(record != nullptr && !record->statisticsCaptureActive && !record->capturable,
              "fallback session is represented by a noncapturable placeholder");
        Check(driver.instrumentedSessionCreates == 1 && driver.originalSessionCreates == 1
                  && driver.sessionDestroys == 1,
            "failed instrumented session is destroyed before exactly one original fallback create");
        Check(driver.requirements2CoreCalls == 0 && driver.requirements2KhrCalls == 1,
              "a Vulkan-1.0/r56-style profile selects the exact enabled KHR requirements2 route");
        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, false);
        const VkDataGraphPipelineSessionCreateInfoARM info{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &info, nullptr, &session)
                  == VK_SUCCESS,
              "noncapturable pipeline creates the original application session");
        const auto placeholder = fixture.device()->resourceManager.getSessionRecord(session);
        Check(placeholder != nullptr && !placeholder->capturable && !placeholder->unsupportedReason.empty(),
              "noncapturable original session retains an owned rejection reason");
        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }
}

std::shared_ptr<SessionRecord> AddSessionOccurrence(LiveFixture& fixture,
                                                    std::string reason,
                                                    bool countable = true,
                                                    bool selectedSnapshot = false)
{
    capture::PipelineMetadata metadata;
    auto pipeline = fixture.device()->resourceManager.addPipelineRecord(NewHandle<VkPipeline>(),
                                                                        capture::PipelineId(900),
                                                                        std::move(metadata),
                                                                        nullptr);
    auto session = fixture.device()->resourceManager.addSessionRecord(
        NewHandle<VkDataGraphPipelineSessionARM>(),
        capture::SessionId(901),
        pipeline,
        {},
        selectedSnapshot ? StatsMemory {NewHandle<VkDeviceMemory>(), VK_NULL_HANDLE, 0, 0, 0} : StatsMemory {});
    auto pool = fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
    (void) pool;
    auto command = fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
    if (!countable)
    {
        command->snapshotError = std::move(reason);
    }
    else
    {
        CommandBufferRecord::DispatchOccurrence occurrence;
        occurrence.session = session;
        occurrence.unsupportedReason = std::move(reason);
        if (selectedSnapshot)
        {
            occurrence.snapshot = std::make_shared<StatsSnapshot>(DeviceHandle(), FakeDestroyBuffer, FakeFreeMemory);
            occurrence.snapshot->session = session;
            occurrence.snapshot->queueFamilyIndex = 0;
        }
        command->dispatches.emplace_back(std::move(occurrence));
    }
    return session;
}

struct SelectedOccurrence
{
    std::shared_ptr<SessionRecord> session;
    std::shared_ptr<StatsSnapshot> snapshot;
};

SelectedOccurrence AddSelectedOccurrence(LiveFixture &fixture, VkCommandPool commandPool,
                                         VkCommandBuffer commandBuffer, uint64_t pipelineId,
                                         uint64_t sessionId)
{
    capture::PipelineMetadata metadata;
    auto pipeline = fixture.device()->resourceManager.addPipelineRecord(NewHandle<VkPipeline>(),
                                                                        capture::PipelineId(pipelineId),
                                                                        std::move(metadata),
                                                                        nullptr);
    auto session = fixture.device()->resourceManager.addSessionRecord(
        NewHandle<VkDataGraphPipelineSessionARM>(),
        capture::SessionId(sessionId),
        pipeline,
        {},
        StatsMemory {NewHandle<VkDeviceMemory>(), VK_NULL_HANDLE, 64, 64, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT});
    fixture.device()->resourceManager.addCommandPoolRecord(commandPool, 0, 0);
    auto command = fixture.device()->resourceManager.addCommandBufferRecord(commandBuffer, commandPool);
    auto snapshot = std::make_shared<StatsSnapshot>(DeviceHandle(), FakeDestroyBuffer, FakeFreeMemory);
    snapshot->session = session;
    snapshot->buffer = NewHandle<VkBuffer>();
    snapshot->memory = NewHandle<VkDeviceMemory>();
    snapshot->queueFamilyIndex = 0;
    snapshot->dataSize = 64;
    snapshot->allocationSize = 64;
    snapshot->memoryProperties = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    command->dispatches.push_back({session, snapshot, {}});
    return {std::move(session), std::move(snapshot)};
}

SelectedOccurrence AddSelectedSnapshotForSession(LiveFixture& fixture,
                                                 VkCommandPool commandPool,
                                                 VkCommandBuffer commandBuffer,
                                                 const std::shared_ptr<SessionRecord> &session)
{
    fixture.device()->resourceManager.addCommandPoolRecord(commandPool, 0, 0);
    auto command = fixture.device()->resourceManager.addCommandBufferRecord(commandBuffer, commandPool);
    auto snapshot = std::make_shared<StatsSnapshot>(DeviceHandle(), FakeDestroyBuffer, FakeFreeMemory);
    snapshot->session = session;
    snapshot->buffer = NewHandle<VkBuffer>();
    snapshot->memory = NewHandle<VkDeviceMemory>();
    snapshot->queueFamilyIndex = 0;
    snapshot->dataSize = 64;
    snapshot->allocationSize = 64;
    snapshot->memoryProperties = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    command->dispatches.push_back({session, snapshot, {}});
    return {session, std::move(snapshot)};
}

VkResult SubmitLegacy(VkQueue queue = QueueHandle(), VkCommandBuffer commandBuffer = CommandBufferHandle())
{
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr,
    };
    return layer_vkQueueSubmit<user_tag>(queue, 1, &submit, VK_NULL_HANDLE);
}

VkResult SubmitLegacyBatch(VkQueue queue, const std::vector<VkCommandBuffer> &commandBuffers)
{
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        static_cast<uint32_t>(commandBuffers.size()),
        commandBuffers.data(),
        0,
        nullptr,
    };
    return layer_vkQueueSubmit<user_tag>(queue, 1, &submit, VK_NULL_HANDLE);
}

void TestCaptureAllocationsReleaseGlobalLock()
{
    const VkDataGraphPipelineDispatchInfoARM dispatchInfo {
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM,
        nullptr,
        0,
    };

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS,
              "direct snapshot lock probe creates an instrumented session");
        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        const auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());

        const uint32_t buffersBefore = driver.bufferCreates;
        const uint32_t bufferDestroysBefore = driver.bufferDestroys;
        const uint32_t allocationsBefore = driver.memoryAllocations;
        const uint32_t freesBefore = driver.memoryFrees;
        ArmBufferCreatePause();
        std::thread producer(
            [&] { layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo); });
        const bool lockAvailable = FinishBufferCreatePause(producer, fixture.instance());

        Check(lockAvailable, "direct snapshot allocation calls the downstream driver without the global Vulkan lock");
        Check(!fixture.instance()->isCaptureEnabled() && driver.cmdDispatchDataGraphCalls == 1
                  && driver.cmdFillBufferCalls == 0 && driver.cmdCopyBufferCalls == 0 && command->dispatches.empty()
                  && driver.bufferCreates == buffersBefore + 1 && driver.bufferDestroys == bufferDestroysBefore + 1
                  && driver.memoryAllocations == allocationsBefore + 1 && driver.memoryFrees == freesBefore + 1,
              "terminalization during unlocked direct allocation forwards only the application dispatch and "
              "reclaims the staged snapshot");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }

    {
        LiveFixture fixture;
        auto source = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1400, 1401);
        (void) source;
        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        const auto primary =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        const VkCommandBuffer secondary = CommandBufferHandle1();

        const uint32_t buffersBefore = driver.bufferCreates;
        const uint32_t bufferDestroysBefore = driver.bufferDestroys;
        const uint32_t allocationsBefore = driver.memoryAllocations;
        const uint32_t freesBefore = driver.memoryFrees;
        ArmBufferCreatePause();
        std::thread producer([&] { layer_vkCmdExecuteCommands<user_tag>(CommandBufferHandle(), 1, &secondary); });
        const bool lockAvailable = FinishBufferCreatePause(producer, fixture.instance());

        Check(lockAvailable,
              "secondary snapshot allocation calls the downstream driver without the global Vulkan lock");
        Check(!fixture.instance()->isCaptureEnabled() && driver.cmdExecuteCommandsCalls == 1
                  && driver.cmdExecuteCommandBuffers.size() == 1
                  && driver.cmdExecuteCommandBuffers.front() == std::vector<VkCommandBuffer> {secondary}
                  && driver.cmdCopyBufferCalls == 0 && primary->dispatches.empty()
                  && driver.bufferCreates == buffersBefore + 1 && driver.bufferDestroys == bufferDestroysBefore + 1
                  && driver.memoryAllocations == allocationsBefore + 1 && driver.memoryFrees == freesBefore + 1,
              "terminalization during unlocked secondary allocation commits no partial primary metadata or copies");
    }

    {
        LiveFixture fixture;
        AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1410, 1411);
        VkResult submitResult = VK_ERROR_UNKNOWN;
        ArmBufferCreatePause();
        std::thread producer([&] { submitResult = SubmitLegacy(); });
        const bool lockAvailable = FinishBufferCreatePause(producer, nullptr);
        fixture.instance()->waitForCollectorJobs();

        Check(lockAvailable, "legacy submission archive materialization releases the global Vulkan lock");
        Check(submitResult == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 1,
              "legacy submission remains successful after unlocked archive materialization");
    }

    {
        LiveFixture fixture;
        AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1420, 1421);
        const VkCommandBufferSubmitInfo commandInfo {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            nullptr,
            CommandBufferHandle(),
            0,
        };
        const VkSubmitInfo2 submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            nullptr,
            0,
            0,
            nullptr,
            1,
            &commandInfo,
            0,
            nullptr,
        };
        VkResult submitResult = VK_ERROR_UNKNOWN;
        ArmBufferCreatePause();
        std::thread producer(
            [&] { submitResult = layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE); });
        const bool lockAvailable = FinishBufferCreatePause(producer, nullptr);
        fixture.instance()->waitForCollectorJobs();

        Check(lockAvailable, "submit2 archive materialization releases the global Vulkan lock");
        Check(submitResult == VK_SUCCESS && driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 1,
              "submit2 remains successful after unlocked archive materialization");
    }
}
void TestRecordingFailureTerminalBypass()
{
    const VkDataGraphPipelineDispatchInfoARM dispatchInfo {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM,
                                                           nullptr,
                                                           0};

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS,
              "recording-failure setup creates an instrumented session");
        const auto sessionRecord = fixture.device()->resourceManager.getSessionRecord(session);
        Check(sessionRecord != nullptr && sessionRecord->statisticsCaptureActive,
              "recording-failure setup session has active statistics capture");

        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        const auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        const uint32_t buffersBefore = driver.bufferCreates;
        {
            capture::fault::ScopedInjection injection(capture::fault::Point::DispatchMetadataRecording,
                                                      capture::fault::Failure::Unexpected);
            layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
        }
        Check(!fixture.instance()->isCaptureEnabled() && driver.cmdDispatchDataGraphCalls == 1
                  && driver.bufferCreates == buffersBefore && driver.cmdCopyBufferCalls == 0
                  && command->dispatches.empty(),
              "dispatch bookkeeping exception terminalizes capture and forwards only the application command");

        layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
        Check(driver.cmdDispatchDataGraphCalls == 2 && driver.bufferCreates == buffersBefore
                  && driver.cmdCopyBufferCalls == 0 && command->dispatches.empty(),
              "later dispatch after terminal failure bypasses all capture instrumentation");

        const uint32_t originalSessions = driver.originalSessionCreates;
        const uint32_t instrumentedSessions = driver.instrumentedSessionCreates;
        VkDataGraphPipelineSessionARM bypassSession = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &bypassSession)
                      == VK_SUCCESS
                  && bypassSession != VK_NULL_HANDLE && driver.originalSessionCreates == originalSessions + 1
                  && driver.instrumentedSessionCreates == instrumentedSessions
                  && fixture.device()->resourceManager.getSessionRecord(bypassSession) == nullptr,
              "session creation after terminal failure forwards the original create info untracked");

        const VkCommandBuffer secondary = CommandBufferHandle1();
        layer_vkCmdExecuteCommands<user_tag>(CommandBufferHandle(), 1, &secondary);
        Check(driver.cmdExecuteCommandsCalls == 1 && driver.cmdExecuteCommandBuffers.size() == 1
                  && driver.cmdExecuteCommandBuffers.back() == std::vector<VkCommandBuffer> {secondary}
                  && driver.cmdCopyBufferCalls == 0,
              "execute-commands after terminal failure forwards the original array without archiving");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0,
              "submit after recording failure forwards without capture rejection or marker work");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), bypassSession, nullptr);
        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS,
              "post-snapshot publication failure setup creates an instrumented session");

        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        const auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        const uint32_t buffersBefore = driver.bufferCreates;
        const uint32_t bufferDestroysBefore = driver.bufferDestroys;
        const uint32_t allocationsBefore = driver.memoryAllocations;
        const uint32_t freesBefore = driver.memoryFrees;
        {
            capture::fault::ScopedInjection injection(capture::fault::Point::DispatchMetadataPublication,
                                                      capture::fault::Failure::Unexpected);
            layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
        }
        Check(!fixture.instance()->isCaptureEnabled() && driver.cmdDispatchDataGraphCalls == 1
                  && driver.bufferCreates == buffersBefore + 1 && driver.bufferDestroys == bufferDestroysBefore + 1
                  && driver.memoryAllocations == allocationsBefore + 1 && driver.memoryFrees == freesBefore + 1
                  && driver.cmdFillBufferCalls == 0 && driver.cmdPipelineBarrierCalls == 0
                  && driver.cmdCopyBufferCalls == 0 && command->dispatches.empty(),
              "metadata append failure after snapshot allocation forwards only the application dispatch and releases "
              "the unrecorded snapshot without recording commands against it");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0,
              "post-snapshot metadata failure leaves the current command buffer submission transparent");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }

    {
        LiveFixture fixture;
        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM sessionInfo {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS,
              "snapshot-allocation failure setup creates an instrumented session");

        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        const auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        const uint32_t buffersBefore = driver.bufferCreates;
        driver.bufferCreateResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
        Check(!fixture.instance()->isCaptureEnabled() && driver.cmdDispatchDataGraphCalls == 1
                  && driver.bufferCreates == buffersBefore + 1 && driver.cmdCopyBufferCalls == 0
                  && command->dispatches.size() == 1 && command->dispatches.front().snapshot == nullptr,
              "snapshot-allocation failure terminalizes capture while preserving the application dispatch");

        layer_vkCmdDispatchDataGraphARM<user_tag>(CommandBufferHandle(), session, &dispatchInfo);
        Check(driver.cmdDispatchDataGraphCalls == 2 && driver.bufferCreates == buffersBefore + 1
                  && command->dispatches.size() == 1 && driver.cmdCopyBufferCalls == 0,
              "later dispatch bypasses snapshot allocation after a Vulkan snapshot failure");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0,
              "snapshot-allocation failure leaves later submission transparent");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
    }
}

void TestStatisticsAliasQueueFamilySharing()
{
    std::vector<VkQueueFamilyProperties> families(4);
    families[0].queueFlags = VK_QUEUE_DATA_GRAPH_BIT_ARM | VK_QUEUE_TRANSFER_BIT;
    families[0].queueCount = 1;
    families[1].queueFlags = VK_QUEUE_DATA_GRAPH_BIT_ARM | VK_QUEUE_COMPUTE_BIT;
    families[1].queueCount = 2;
    families[2].queueFlags = VK_QUEUE_DATA_GRAPH_BIT_ARM;
    families[2].queueCount = 1;
    families[3].queueFlags = VK_QUEUE_TRANSFER_BIT;
    families[3].queueCount = 1;

    LiveFixture fixture(true, true, true, true, 64, std::move(families), {0, 1, 2, 3});
    VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
    const VkDataGraphPipelineSessionCreateInfoARM info{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
        nullptr,
        0,
        pipeline,
    };
    VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
    Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &info, nullptr, &session) == VK_SUCCESS,
          "statistics session creation succeeds with multiple compatible queue families");
    const auto record = fixture.device()->resourceManager.getSessionRecord(session);
    Check(record != nullptr && record->statisticsCaptureActive, "multi-family statistics session remains capturable");
    Check(driver.bufferCreateObservations.size() == 1,
          "statistics session creates one hidden alias buffer before command recording");
    if (driver.bufferCreateObservations.size() == 1)
    {
        const auto &source = driver.bufferCreateObservations[0];
        Check(source.usage == (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                  && source.sharingMode == VK_SHARING_MODE_CONCURRENT
                  && source.queueFamilyIndices == std::vector<uint32_t>({0, 1}),
              "statistics alias uses concurrent sharing across every enabled data-graph transfer-capable family");
    }
    layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
    layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
}

void TestFilterBeforeLegalityAndSubmissionTransactions()
{
    const std::array<const char *, 6> reasons{
        "selected capture rejects protected data-graph execution",
        "selected capture rejects foreign processing engines",
        "selected capture cannot override application-provided pipeline statistics state",
        "selected capture found malformed neural-statistics bind-point requirements",
        "selected capture has no acceptable statistics memory type",
        "selected capture failed to create a host-visible snapshot destination",
    };
    for (const char *reason : reasons)
    {
        {
            LiveFixture fixture;
            fixture.instance()->dispatchFilter = DispatchFilter::Single(99);
            auto session = AddSessionOccurrence(fixture, reason);
            Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1
                      && session->nextExecutionIndex == 1,
                  "known noncapturable occurrence is counted and transparent when unselected");
        }
        {
            LiveFixture fixture;
            fixture.instance()->dispatchFilter = DispatchFilter::All();
            auto session = AddSessionOccurrence(fixture, reason);
            Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1
                      && session->nextExecutionIndex == 0 && !fixture.instance()->isCaptureEnabled(),
                  "selected noncapturable occurrence terminalizes capture and forwards unchanged");
        }
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::Single(99);
        AddSessionOccurrence(fixture, "secondary command metadata propagation failed", false);
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1
                  && !fixture.instance()->isCaptureEnabled(),
              "uncountable recording corruption terminalizes capture and forwards the current submit");
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        auto session = AddSessionOccurrence(fixture, {}, true, true);
        driver.applicationSubmitResult = VK_ERROR_DEVICE_LOST;
        Check(SubmitLegacy() == VK_ERROR_DEVICE_LOST && session->nextExecutionIndex == 0
                  && fixture.instance()->dispatchIds.nextValue() == 0,
              "downstream queue failure cancels occurrence and ID reservations");
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        auto session = AddSessionOccurrence(fixture, {}, true, true);
        Check(SubmitLegacy() == VK_SUCCESS && session->nextExecutionIndex == 1
                  && fixture.instance()->dispatchIds.nextValue() == 1,
              "accepted queue success commits executed index and selected dispatch ID");
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        auto session = AddSessionOccurrence(fixture, {}, true, true);
        const std::vector<VkCommandBuffer> duplicates{CommandBufferHandle(), CommandBufferHandle()};
        Check(SubmitLegacyBatch(QueueHandle(), duplicates) == VK_SUCCESS && driver.legacyApplicationSubmits == 1
                  && session->nextExecutionIndex == 2 && fixture.instance()->dispatchIds.nextValue() == 2,
              "same primary command buffer twice receives unique submission snapshots");
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        auto session = AddSessionOccurrence(fixture, {}, true, true);
        driver.createFenceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
        Check(SubmitLegacy() == VK_SUCCESS && !fixture.instance()->isCaptureEnabled()
                  && fixture.instance()->collectorJobCount() == 0,
              "post-forward collection failure preserves the accepted Vulkan result and disables capture");
        CheckFenceAccounting("marker creation failure leaves no fake-fence ownership or residue");
    }

    {
        LiveFixture fixture;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        auto session = AddSessionOccurrence(fixture, {}, true, true);
        driver.markerSubmitResult = VK_ERROR_DEVICE_LOST;
        Check(SubmitLegacy() == VK_SUCCESS && !fixture.instance()->isCaptureEnabled()
                  && fixture.instance()->collectorJobCount() == 0,
              "marker submission failure preserves the application result and retires capacity");
        CheckFenceAccounting("marker submission failure destroys its created fence exactly once");
    }
}

void TestLegacyDeviceGroupSubmitRewrite()
{
    LiveFixture fixture;
    fixture.instance()->dispatchFilter = DispatchFilter::All();
    AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1010, 1011);
    AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1020, 1021);

    const std::array<VkCommandBuffer, 2> commandBuffers {CommandBufferHandle(), CommandBufferHandle1()};
    const std::array<uint32_t, 2> commandBufferDeviceMasks{0x1, 0x2};
    const std::array<uint32_t, 1> waitSemaphoreDeviceIndices{1};
    const std::array<uint32_t, 1> signalSemaphoreDeviceIndices{0};
    const VkSemaphore waitSemaphore = NewHandle<VkSemaphore>();
    const VkSemaphore signalSemaphore = NewHandle<VkSemaphore>();
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkDeviceGroupSubmitInfo deviceGroup{
        VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO,
        nullptr,
        static_cast<uint32_t>(waitSemaphoreDeviceIndices.size()),
        waitSemaphoreDeviceIndices.data(),
        static_cast<uint32_t>(commandBufferDeviceMasks.size()),
        commandBufferDeviceMasks.data(),
        static_cast<uint32_t>(signalSemaphoreDeviceIndices.size()),
        signalSemaphoreDeviceIndices.data(),
    };
    const VkProtectedSubmitInfo protectedSubmit{
        VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
        &deviceGroup,
        VK_FALSE,
    };
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        &protectedSubmit,
        1,
        &waitSemaphore,
        &waitStage,
        static_cast<uint32_t>(commandBuffers.size()),
        commandBuffers.data(),
        1,
        &signalSemaphore,
    };

    Check(layer_vkQueueSubmit<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS,
          "legacy device-group submission with selected capture succeeds");
    fixture.instance()->waitForCollectorJobs();

    Check(driver.lastLegacyCommandBuffers.size() == 4 && driver.lastLegacyCommandBuffers[0] == CommandBufferHandle()
              && driver.lastLegacyCommandBuffers[2] == CommandBufferHandle1()
              && driver.lastLegacyCommandBuffers[1] != VK_NULL_HANDLE
              && driver.lastLegacyCommandBuffers[3] != VK_NULL_HANDLE
              && driver.lastLegacyCommandBuffers[1] != CommandBufferHandle()
              && driver.lastLegacyCommandBuffers[3] != CommandBufferHandle1(),
          "legacy device-group rewrite interleaves archive command buffers");
    Check(driver.lastLegacyCommandBufferDeviceMasks == std::vector<uint32_t>({0x1, 0x1, 0x2, 0x2}),
          "legacy device-group rewrite duplicates each application device mask for its archive");
    Check(driver.lastLegacyWaitSemaphoreDeviceIndices == std::vector<uint32_t>({1})
              && driver.lastLegacySignalSemaphoreDeviceIndices == std::vector<uint32_t>({0}),
          "legacy device-group rewrite preserves semaphore device indices");
    Check(
        driver.lastLegacyDeviceGroupHadProtectedPrefix,
        "legacy device-group rewrite preserves structures preceding the device-group information");
    Check(deviceGroup.commandBufferCount == commandBufferDeviceMasks.size()
              && deviceGroup.pCommandBufferDeviceMasks == commandBufferDeviceMasks.data()
              && commandBufferDeviceMasks == std::array<uint32_t, 2>({0x1, 0x2})
              && protectedSubmit.pNext == &deviceGroup,
          "legacy device-group rewrite does not modify application-owned structures");
}

void TestSecondaryOccurrenceArchiving()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-secondary-occurrences-" + UniqueTestSuffix());
    std::error_code error;
    fs::remove_all(root, error);
    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto secondaryOccurrence =
            AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1030, 1031);
        auto secondary = fixture.device()->resourceManager.getCommandBufferRecord(CommandBufferHandle1());
        secondary->usageFlags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        auto primary =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());

        const std::array<VkCommandBuffer, 2> executions {CommandBufferHandle1(), CommandBufferHandle1()};
        layer_vkCmdExecuteCommands<user_tag>(CommandBufferHandle(),
                                             static_cast<uint32_t>(executions.size()),
                                             executions.data());

        Check(driver.cmdExecuteCommandsCalls == 2 && driver.cmdCopyBufferCalls == 2,
              "secondary arrays are split and archived after every occurrence");
        Check(primary->dispatches.size() == 2 && primary->dispatches[0].snapshot != secondaryOccurrence.snapshot
                  && primary->dispatches[1].snapshot != secondaryOccurrence.snapshot
                  && primary->dispatches[0].snapshot != primary->dispatches[1].snapshot,
              "repeated secondary executions receive unique primary occurrence snapshots");
        Check(SubmitLegacy() == VK_SUCCESS, "primary containing repeated secondary executions submits successfully");
        fixture.instance()->waitForCollectorJobs();
        Check(secondaryOccurrence.session->nextExecutionIndex == 2 && fixture.instance()->dispatchIds.nextValue() == 2,
              "repeated secondary executions materialize two selected captures");
    }

    {
        LiveFixture fixture;
        auto secondaryOccurrence =
            AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1032, 1033);
        (void) secondaryOccurrence;
        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        auto primary =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());

        const std::array<VkCommandBuffer, 3> executions {CommandBufferHandle1(),
                                                         CommandBufferHandle1(),
                                                         CommandBufferHandle1()};
        capture::fault::ScopedInjection injection(capture::fault::Point::SecondaryExecutionMetadataPropagation,
                                                  capture::fault::Failure::Unexpected,
                                                  2);
        layer_vkCmdExecuteCommands<user_tag>(CommandBufferHandle(),
                                             static_cast<uint32_t>(executions.size()),
                                             executions.data());

        Check(driver.cmdExecuteCommandsCalls == 1 && driver.cmdExecuteCommandBuffers.size() == 1
                  && driver.cmdExecuteCommandBuffers[0]
                         == std::vector<VkCommandBuffer>(executions.begin(), executions.end()),
              "secondary metadata failure forwards the complete original array exactly once");
        Check(driver.cmdCopyBufferCalls == 0 && !primary->snapshotError.empty(),
              "secondary metadata failure abandons partial archiving and marks capture metadata unusable");
        Check(!fixture.instance()->isCaptureEnabled(),
              "secondary metadata exception terminalizes capture before later submission");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers.size() == 1
                  && driver.lastLegacyCommandBuffers[0] == CommandBufferHandle() && driver.cmdCopyBufferCalls == 0,
              "secondary metadata failure forwards the later original primary submit without a marker");
        Check(secondaryOccurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && secondaryOccurrence.snapshot->dispatchId == std::nullopt,
              "secondary metadata failure consumes no execution or capture identity bookkeeping");
    }
    fs::remove_all(root, error);
}

std::atomic<bool> snapshotRetirementEntered {false};
std::atomic<bool> releaseSnapshotRetirement {false};
std::atomic<uint32_t> secondarySnapshotCreateCalls {0};

VKAPI_ATTR void VKAPI_CALL PausingSnapshotDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*)
{
    snapshotRetirementEntered.store(true, std::memory_order_release);
    snapshotRetirementEntered.notify_all();
    while (!releaseSnapshotRetirement.load(std::memory_order_acquire))
    {
        releaseSnapshotRetirement.wait(false, std::memory_order_relaxed);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ThrowOnSecondSnapshotCreateBuffer(VkDevice device,
                                                                 const VkBufferCreateInfo* createInfo,
                                                                 const VkAllocationCallbacks* allocator,
                                                                 VkBuffer* buffer)
{
    if (secondarySnapshotCreateCalls.fetch_add(1, std::memory_order_relaxed) + 1 == 2)
    {
        throw std::runtime_error("injected secondary snapshot creation failure");
    }
    return FakeCreateBuffer(device, createInfo, allocator, buffer);
}

void ArmSnapshotRetirementProbe() noexcept
{
    snapshotRetirementEntered.store(false, std::memory_order_relaxed);
    releaseSnapshotRetirement.store(false, std::memory_order_relaxed);
}

bool FinishSnapshotRetirementProbe(std::thread& producer)
{
    const auto enterDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!snapshotRetirementEntered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < enterDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const bool entered = snapshotRetirementEntered.load(std::memory_order_acquire);
    bool lockAvailable = false;
    if (entered)
    {
        const auto lockDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        do
        {
            if (g_vulkanLock.try_lock())
            {
                lockAvailable = true;
                g_vulkanLock.unlock();
                break;
            }
            std::this_thread::yield();
        }
        while (std::chrono::steady_clock::now() < lockDeadline);
    }

    releaseSnapshotRetirement.store(true, std::memory_order_release);
    releaseSnapshotRetirement.notify_all();
    producer.join();
    return entered && lockAvailable;
}

void TestSnapshotOwnerLockBoundaries()
{
    {
        LiveFixture fixture;
        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        command->usageFlags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        command->snapshotError = "stale snapshot error";

        auto snapshot = std::make_shared<StatsSnapshot>(DeviceHandle(), PausingSnapshotDestroyBuffer, FakeFreeMemory);
        snapshot->buffer = NewHandle<VkBuffer>();
        command->dispatches.push_back({nullptr, snapshot, {}});
        snapshot.reset();

        ArmSnapshotRetirementProbe();
        VkResult resetResult = VK_ERROR_UNKNOWN;
        std::thread producer([&] { resetResult = layer_vkResetCommandBuffer<user_tag>(CommandBufferHandle(), 0); });
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);

        Check(retiredUnlocked, "final command-buffer snapshot owner retires after the global lock is released");
        Check(resetResult == VK_SUCCESS && command->dispatches.empty() && command->snapshotError.empty()
                  && command->usageFlags == 0,
              "deferred snapshot retirement preserves command-buffer reset metadata semantics");
    }

    {
        LiveFixture fixture;
        auto secondaryOccurrence =
            AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1034, 1035);
        auto secondary = fixture.device()->resourceManager.getCommandBufferRecord(CommandBufferHandle1());
        auto secondSource = std::make_shared<StatsSnapshot>(DeviceHandle(), FakeDestroyBuffer, FakeFreeMemory);
        secondSource->session = secondaryOccurrence.session;
        secondary->dispatches.push_back({secondaryOccurrence.session, secondSource, {}});

        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        auto primary =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        fixture.device()->driver.vkCreateBuffer = ThrowOnSecondSnapshotCreateBuffer;
        fixture.device()->driver.vkDestroyBuffer = PausingSnapshotDestroyBuffer;
        secondarySnapshotCreateCalls.store(0, std::memory_order_relaxed);
        ArmSnapshotRetirementProbe();

        const VkCommandBuffer secondaryHandle = CommandBufferHandle1();
        std::thread producer([&] { layer_vkCmdExecuteCommands<user_tag>(CommandBufferHandle(), 1, &secondaryHandle); });
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);
        fixture.device()->driver.vkCreateBuffer = FakeCreateBuffer;
        fixture.device()->driver.vkDestroyBuffer = FakeDestroyBuffer;

        Check(retiredUnlocked,
              "secondary staging exception retires an earlier destination after the global lock is released");
        Check(secondarySnapshotCreateCalls.load(std::memory_order_relaxed) == 2 && driver.cmdExecuteCommandsCalls == 1
                  && !primary->snapshotError.empty() && !fixture.instance()->isCaptureEnabled(),
              "secondary staging exception follows the terminal transparent-forwarding path");
    }
}

void TestSnapshotReusePolicies()
{
    namespace fs = std::filesystem;

    {
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-simultaneous-reuse-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1040, 1041);
            auto command = fixture.device()->resourceManager.getCommandBufferRecord(CommandBufferHandle());
            command->usageFlags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }
            Check(SubmitLegacy() == VK_SUCCESS && SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 2
                      && fixture.instance()->collectorJobCount() == 2 && occurrence.snapshot->inFlightUses == 2,
                  "overlapping simultaneous-use submissions receive independent destinations");
            SignalMarkerFence(0);
            Check(WaitForCollectorCount(*fixture.instance(), 1) && occurrence.snapshot->inFlightUses == 1,
                  "first simultaneous execution retires independently");
            SignalMarkerFence(1);
            Check(WaitForCollectorCount(*fixture.instance(), 0) && !occurrence.snapshot->inFlight,
                  "all simultaneous execution destinations drain independently");
        }
        fs::remove_all(root, error);
    }

    {
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-distinct-same-session-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1042, 1043);
            auto second =
                AddSelectedSnapshotForSession(fixture, CommandPoolHandle1(), CommandBufferHandle1(), first.session);
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }
            Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS
                      && SubmitLegacy(QueueHandle(), CommandBufferHandle1()) == VK_SUCCESS,
                  "distinct snapshots for one synchronized session do not wait for CPU collection");
            Check(fixture.instance()->collectorJobCount() == 2 && first.snapshot->inFlight && second.snapshot->inFlight
                      && first.snapshot->dispatchId == capture::DispatchId(0)
                      && second.snapshot->dispatchId == capture::DispatchId(1)
                      && first.session->nextExecutionIndex == 2,
                  "distinct same-session submissions retain independent snapshots and ordered identities");
            SignalMarkerFence(0);
            SignalMarkerFence(1);
            Check(WaitForCollectorCount(*fixture.instance(), 0) && !first.snapshot->inFlight
                      && !second.snapshot->inFlight,
                  "distinct same-session snapshots drain independently");
        }
        fs::remove_all(root, error);
    }
}

void TestAsynchronousCollectorOrderingAndReuse()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-async-collector-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    ResetFileSystemObserverState();
    ResetCollectorObserverState();
    capture::SetCaptureFileSystemObserver(ObserveFileSystem);
    capture::SetCollectorObserver(ObserveCollector);
    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto reused = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1050, 1051);
        auto independent = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1060, 1061);
        {
            std::lock_guard lock(fakeFenceMutex);
            autoSignalMarkerFences = false;
        }

        observerProducer0.store(CurrentThreadHash(), std::memory_order_relaxed);
        Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS
                  && SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS
                  && SubmitLegacy(QueueHandle1(), CommandBufferHandle1()) == VK_SUCCESS,
              "overlapping reuse and independent queue work forward without retirement waits");
        Check(driver.legacyApplicationSubmits == 3 && MarkerFenceCount() == 3
                  && fixture.instance()->collectorJobCount() == 3 && reused.snapshot->inFlightUses == 2
                  && independent.snapshot->inFlightUses == 1 && reused.snapshot->dispatchId == capture::DispatchId(1)
                  && independent.snapshot->dispatchId == capture::DispatchId(2),
              "submission-owned snapshots retain admission-ordered identities");

        SignalMarkerFence(2);
        Check(WaitForCollectorCount(*fixture.instance(), 2) && reused.snapshot->inFlightUses == 2
                  && !independent.snapshot->inFlight,
              "independent later queue completion retires out of order");
        SignalMarkerFence(0);
        Check(WaitForCollectorCount(*fixture.instance(), 1) && reused.snapshot->inFlightUses == 1,
              "first reused execution retires without affecting the second");
        SignalMarkerFence(1);
        Check(WaitForCollectorCount(*fixture.instance(), 0) && !reused.snapshot->inFlight,
              "all unique submission snapshots drain independently");
        Check(observerCollectorCopies.load(std::memory_order_relaxed) == 3,
              "all successful selected submissions are copied on the collector thread");
    }
    capture::SetCollectorObserver(nullptr);
    capture::SetCaptureFileSystemObserver(nullptr);

    Check(!observerCollectorRanOnProducer.load(std::memory_order_relaxed),
          "fence polling and mapped-memory collection never run on intercepted producers");
    Check(!observerRanOnProducer.load(std::memory_order_relaxed),
          "asynchronous collection performs no filesystem operation on intercepted producers");
    std::ifstream captureFile(root / "capture.json");
    const auto document = nlohmann::json::parse(captureFile, nullptr, false);
    Check(document.is_object() && document.value("status", "") == "complete",
          "normal completion publishes only after collector and writer drain");
    fs::remove_all(root, cleanupError);
}

void TestCollectorCapacityTurnover()
{
    namespace fs = std::filesystem;

    {
        const fs::path root = fs::temp_directory_path() / ("neural-statistics-capacity-turnover-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture(true, true, true, true, 1);
            StartCaptureWriter(fixture, root);
            auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1062, 1063);
            auto second = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1064, 1065);
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }

            Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS,
                  "completion-turnover first selected submission fills capacity one");
            std::atomic<bool> producerFinished{false};
            VkResult secondResult = VK_ERROR_UNKNOWN;
            std::thread producer(
                [&]
                {
                    secondResult = SubmitLegacy(QueueHandle1(), CommandBufferHandle1());
                    producerFinished.store(true, std::memory_order_release);
                });

            const bool blocked = fixture.device()->collector->waitForBlockedProducers(1);
            Check(blocked, "completion-turnover second production hook waits without owning capacity");
            SignalMarkerFence(0);
            const bool completedWithoutClose = WaitForFlag(producerFinished);
            if (!completedWithoutClose)
            {
                fixture.device()->collector->closeAdmission();
            }
            producer.join();

            Check(completedWithoutClose && secondResult == VK_SUCCESS,
                  "collector completion releases capacity without closeAdmission rescue");
            Check(driver.legacyApplicationSubmits == 2 && driver.markerSubmits == 2 && MarkerFenceCount() == 2,
                  "capacity waiter rebuilds and forwards exactly one application/marker pair");
            Check(first.snapshot->dispatchId == capture::DispatchId(0)
                      && second.snapshot->dispatchId == capture::DispatchId(1) && first.snapshot->executionIndex == 0
                      && second.snapshot->executionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 2,
                  "failed capacity attempt consumes no IDs or execution indices");

            SignalMarkerFence(1);
            Check(WaitForCollectorCount(*fixture.instance(), 0) && fixture.device()->collector->pendingCount() == 0
                      && !first.snapshot->inFlight && !second.snapshot->inFlight,
                  "completion turnover drains all jobs and tentative/accepted reservations");
            CheckFenceAccounting("completion turnover destroys every marker fence exactly once");
        }
        fs::remove_all(root, error);
    }

    {
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-capacity-failure-turnover-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture(true, true, true, true, 1);
            StartCaptureWriter(fixture, root);
            auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1066, 1067);
            auto second = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1068, 1069);
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }

            capture::fault::ScopedGlobalInjection injection(capture::fault::Point::WriterDispatchArtifactSubmission,
                capture::fault::Failure::Unexpected);
            Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS,
                  "failure-turnover first selected submission fills capacity one");
            std::atomic<bool> producerFinished{false};
            VkResult secondResult = VK_SUCCESS;
            std::thread producer(
                [&]
                {
                    secondResult = SubmitLegacy(QueueHandle1(), CommandBufferHandle1());
                    producerFinished.store(true, std::memory_order_release);
                });

            const bool blocked = fixture.device()->collector->waitForBlockedProducers(1);
            Check(blocked, "failure-turnover second production hook waits on full capacity");
            SignalMarkerFence(0);
            const bool completedWithoutClose = WaitForFlag(producerFinished);
            if (!completedWithoutClose)
            {
                fixture.device()->collector->closeAdmission();
            }
            producer.join();

            Check(completedWithoutClose && secondResult == VK_SUCCESS,
                  "terminal completion wakes the waiter to forward without capture");
            Check(driver.legacyApplicationSubmits == 2 && driver.markerSubmits == 1 && MarkerFenceCount() == 1,
                  "failure turnover forwards the original second submit without a capture marker");
            Check(first.snapshot->dispatchId == capture::DispatchId(0) && second.snapshot->dispatchId == std::nullopt
                      && first.session->nextExecutionIndex == 1 && second.session->nextExecutionIndex == 0
                      && fixture.instance()->dispatchIds.nextValue() == 1,
                  "failure turnover retires only the accepted first identity");
            Check(WaitForCollectorCount(*fixture.instance(), 0) && fixture.device()->collector->pendingCount() == 0
                      && !first.snapshot->inFlight && !second.snapshot->inFlight,
                  "failure turnover cancels tentative reservations and retires accepted work");
            const auto state = fixture.instance()->captureStateSnapshot();
            Check(!state.enabled && state.terminalRequestIssued && state.terminalTransitionCount == 1
                      && state.failure == "raw-statistics writer job was rejected by the GPU collector",
                  "failure turnover publishes exactly one first terminal transition");
            CheckFenceAccounting("failure turnover destroys the accepted marker fence exactly once");
        }
        fs::remove_all(root, error);
    }
}

void TestDeviceWaitFailureShutdown()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-device-wait-failure-" + UniqueTestSuffix());
    std::error_code error;
    fs::remove_all(root, error);

    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1070, 1071);
        auto second = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1072, 1073);
        {
            std::lock_guard lock(fakeFenceMutex);
            autoSignalMarkerFences = false;
        }

        Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS
                  && SubmitLegacy(QueueHandle1(), CommandBufferHandle1()) == VK_SUCCESS
                  && fixture.instance()->collectorJobCount() == 2 && MarkerFenceCount() == 2,
              "device-wait failure setup owns multiple pending jobs and marker fences");
        driver.deviceWaitIdleResult = VK_ERROR_DEVICE_LOST;
        fixture.destroyDevice();

        Check(driver.deviceWaitIdleCalls == 1 && driver.deviceDestroys == 1 && driver.collectorJobsAtDeviceDestroy == 0
                  && driver.liveFencesAtDeviceDestroy == 0,
              "downstream destroy runs only after failed drain joins and retires jobs/fences");
        Check(fixture.instance()->collectorJobCount() == 0 && !first.snapshot->inFlight && !second.snapshot->inFlight,
              "device-wait failure finishes every accepted plan exactly once");
        Check(first.session->nextExecutionIndex == 1 && second.session->nextExecutionIndex == 1
                  && first.snapshot->dispatchId == capture::DispatchId(0)
                  && second.snapshot->dispatchId == capture::DispatchId(1)
                  && fixture.instance()->dispatchIds.nextValue() == 2,
              "shutdown failure preserves accepted deterministic IDs and indices");
        Check(driver.mapMemoryCalls == 0,
              "device-wait failure submits no raw writer work after terminal drain failure");
        const auto state = fixture.instance()->captureStateSnapshot();
        Check(!state.enabled && state.terminalRequestIssued && state.terminalTransitionCount == 1
                  && state.failure == "GPU collector device drain failed during shutdown",
              "device-wait failure records one first terminal transition");
        CheckFenceAccounting("device-wait failure destroys every pending fence exactly once");
    }

    fs::remove_all(root, error);
}

void TestCollectorFailuresBackpressureAndLifetime()
{
    namespace fs = std::filesystem;

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1070, 1071);
        capture::fault::ScopedInjection injection(capture::fault::Point::CollectorJobConstruction,
                                                  capture::fault::Failure::BadAllocation);
        Check(SubmitLegacy() == VK_ERROR_OUT_OF_HOST_MEMORY && driver.legacyApplicationSubmits == 0
                  && !occurrence.snapshot->inFlight,
              "collector job construction failure is pre-forward and releases reservations");
    }

    const auto runCollectorFailure = [&](capture::fault::Point point, const char *name)
    {
        const fs::path root =
            fs::temp_directory_path() / (std::string("neural-statistics-") + name + "-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1080, 1081);
            capture::fault::ScopedGlobalInjection injection(point, capture::fault::Failure::Unexpected);
            Check(SubmitLegacy() == VK_SUCCESS, "post-forward collector fault preserves application success");
            Check(WaitForCollectorCount(*fixture.instance(), 0),
                  "post-forward collector fault retires within the finite timeout");
            Check(!fixture.instance()->isCaptureEnabled() && !occurrence.snapshot->inFlight,
                  "post-forward collector fault terminalizes capture and releases reservations");
            CheckFenceAccounting("post-forward collector fault destroys its marker fence exactly once");
        }
        fs::remove_all(root, error);
    };
    runCollectorFailure(capture::fault::Point::CollectorQueueAdmission, "collector-admission-fault");
    runCollectorFailure(capture::fault::Point::CollectorPollingBookkeeping, "collector-polling-fault");
    runCollectorFailure(capture::fault::Point::CollectorCompletionCopy, "collector-copy-fault");

    {
        const fs::path root = fs::temp_directory_path() / ("neural-statistics-fence-error-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1090, 1091);
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }
            Check(SubmitLegacy() == VK_SUCCESS, "fence-error setup submission succeeds");
            SignalMarkerFence(0, VK_ERROR_DEVICE_LOST);
            Check(WaitForCollectorCount(*fixture.instance(), 0),
                  "device-lost fence status retires within the finite timeout");
            Check(!fixture.instance()->isCaptureEnabled() && !occurrence.snapshot->inFlight,
                  "device-lost fence status terminalizes capture without stranding reservations");
            CheckFenceAccounting("device-lost polling destroys its marker fence exactly once");
        }
        fs::remove_all(root, error);
    }

    for (const bool mapFailure : {false, true})
    {
        const fs::path root = fs::temp_directory_path()
                            / ((mapFailure ? "neural-statistics-map-error-" : "neural-statistics-invalidate-error-")
                               + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1095, 1096);
            occurrence.snapshot->memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            driver.memoryTypeProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            if (mapFailure)
            {
                driver.mapMemoryResult = VK_ERROR_MEMORY_MAP_FAILED;
            }
            else
            {
                driver.invalidateMemoryResult = VK_ERROR_DEVICE_LOST;
            }
            Check(SubmitLegacy() == VK_SUCCESS, "mapped-memory failure setup preserves application success");
            fixture.instance()->waitForCollectorJobs();
            Check(!fixture.instance()->isCaptureEnabled() && !occurrence.snapshot->inFlight,
                  "map/invalidate failure terminalizes capture and releases reservations");
        }
        fs::remove_all(root, error);
    }

    {
        LiveFixture fixture(true, true, true, true, 1);
        auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1097, 1098);
        auto second = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1099, 1100);
        {
            std::lock_guard lock(fakeFenceMutex);
            autoSignalMarkerFences = false;
        }
        Check(SubmitLegacy(QueueHandle(), CommandBufferHandle()) == VK_SUCCESS,
              "first job fills the bounded collector");
        std::atomic<bool> producerFinished{false};
        VkResult blockedResult = VK_SUCCESS;
        std::thread blockedProducer(
            [&]
            {
                blockedResult = SubmitLegacy(QueueHandle1(), CommandBufferHandle1());
                producerFinished.store(true, std::memory_order_release);
            });
        Check(fixture.device()->collector->waitForBlockedProducers(1),
              "full collector applies bounded producer backpressure");
        fixture.device()->collector->closeAdmission();
        const bool closeWokeProducer = WaitForFlag(producerFinished);
        blockedProducer.join();
        Check(closeWokeProducer && blockedResult == VK_SUCCESS && driver.legacyApplicationSubmits == 2
                  && second.snapshot->dispatchId == std::nullopt && !second.snapshot->inFlight,
              "collector close wakes, cancels capture planning, and forwards the blocked application submit");
        Check(WaitForCollectorCount(*fixture.instance(), 0) && !first.snapshot->inFlight,
              "accepted work drains after collector admission closes");
        CheckFenceAccounting("close wakeup destroys accepted fences exactly once without residue");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1102, 1103);
        std::weak_ptr<StatsSnapshot> retained = occurrence.snapshot;
        {
            std::lock_guard lock(fakeFenceMutex);
            autoSignalMarkerFences = false;
        }
        Check(SubmitLegacy() == VK_SUCCESS, "snapshot-lifetime setup submission succeeds");
        Check(layer_vkResetCommandBuffer<user_tag>(CommandBufferHandle(), 0) == VK_SUCCESS,
              "command-buffer reset succeeds while collector work is pending");
        occurrence.snapshot.reset();
        Check(!retained.expired(), "collector shared ownership retains snapshot after command-buffer reset");
        SignalMarkerFence(0);
        fixture.instance()->waitForCollectorJobs();
        Check(retained.expired(), "snapshot allocation retires after collector completion releases final ownership");
    }

    {
        const uint32_t waitsBefore = driver.deviceWaitIdleCalls;
        {
            LiveFixture fixture;
            AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1104, 1105);
            {
                std::lock_guard lock(fakeFenceMutex);
                autoSignalMarkerFences = false;
            }
            Check(SubmitLegacy() == VK_SUCCESS, "device-destruction drain setup submission succeeds");
        }
        Check(driver.deviceWaitIdleCalls == waitsBefore + 1 && driver.deviceDestroys == 1,
              "device destruction asks the collector thread to drain before downstream destroy");
    }
}

void TestRawWriterEnqueueFailure()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-raw-enqueue-failure-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    ResetFileSystemObserverState();
    capture::SetCaptureFileSystemObserver(ObserveFileSystem);
    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1100, 1101);

        capture::fault::ScopedGlobalInjection injection(capture::fault::Point::WriterDispatchArtifactSubmission,
            capture::fault::Failure::Unexpected);
        const VkResult result = SubmitLegacy();
        fixture.instance()->waitForCollectorJobs();
        Check(result == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 1,
              "raw writer enqueue rejection preserves successful application and marker submits");
        Check(occurrence.session->nextExecutionIndex == 1 && occurrence.snapshot->executionIndex == 0
                  && occurrence.snapshot->dispatchId == capture::DispatchId(0),
              "raw enqueue rejection preserves accepted selection identity and order");
        Check(!occurrence.snapshot->inFlight, "raw enqueue rejection releases session and snapshot reservations");

        const auto state = fixture.instance()->captureStateSnapshot();
        Check(!state.enabled && state.terminalRequestIssued && state.terminalTransitionCount == 1
                  && state.failure == "raw-statistics writer job was rejected by the GPU collector",
              "raw enqueue rejection publishes exactly one coherent terminal diagnostic");
    }
    capture::SetCaptureFileSystemObserver(nullptr);

    Check(!observerRanOnProducer.load(std::memory_order_relaxed),
          "raw enqueue failure performs no filesystem publication on the intercepted thread");
    Check(observerWriter.load(std::memory_order_relaxed) != 0
              && !observerSawMultipleWorkers.load(std::memory_order_relaxed),
          "raw enqueue failure publication remains confined to one writer thread");
    std::ifstream captureFile(root / "capture.json");
    const auto document = nlohmann::json::parse(captureFile, nullptr, false);
    Check(document.is_object() && document.value("status", "") == "error",
          "raw enqueue rejection drains a terminal error capture");
    fs::remove_all(root, cleanupError);
}

void TestConcurrentCollectionTerminalTransition()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-concurrent-collection-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    ResetFileSystemObserverState();
    capture::SetCaptureFileSystemObserver(ObserveFileSystem);
    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto first = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1200, 1201);
        auto second = AddSelectedOccurrence(fixture, CommandPoolHandle1(), CommandBufferHandle1(), 1210, 1211);

        concurrentApplicationSubmits.store(0, std::memory_order_relaxed);
        concurrentMarkerSubmits.store(0, std::memory_order_relaxed);
        concurrentCollectionMode.store(true, std::memory_order_release);
        {
            std::lock_guard lock(fakeFenceMutex);
            autoSignalMarkerFences = false;
        }

        VkResult firstResult = VK_ERROR_UNKNOWN;
        VkResult secondResult = VK_ERROR_UNKNOWN;
        capture::fault::ScopedGlobalInjection injection(capture::fault::Point::WriterDispatchArtifactSubmission,
            capture::fault::Failure::Unexpected);
        std::thread firstProducer(
            [&]
            {
                observerProducer0.store(CurrentThreadHash(), std::memory_order_relaxed);
                firstResult = SubmitLegacy(QueueHandle(), CommandBufferHandle());
            });
        std::thread secondProducer(
            [&]
            {
                observerProducer1.store(CurrentThreadHash(), std::memory_order_relaxed);
                secondResult = SubmitLegacy(QueueHandle1(), CommandBufferHandle1());
            });
        firstProducer.join();
        secondProducer.join();
        Check(MarkerFenceCount() == 2, "both concurrent submissions are admitted before either completion");
        SignalMarkerFence(0);
        SignalMarkerFence(1);
        fixture.instance()->waitForCollectorJobs();
        concurrentCollectionMode.store(false, std::memory_order_release);

        Check(firstResult == VK_SUCCESS && secondResult == VK_SUCCESS
                  && concurrentApplicationSubmits.load(std::memory_order_relaxed) == 2
                  && concurrentMarkerSubmits.load(std::memory_order_relaxed) == 2,
              "two genuine concurrent queue producers preserve both downstream successes");
        Check(first.session->nextExecutionIndex == 1 && second.session->nextExecutionIndex == 1
                  && first.snapshot->executionIndex == 0 && second.snapshot->executionIndex == 0,
              "concurrent submissions preserve per-session selected execution indices");
        Check(first.snapshot->dispatchId.has_value() && second.snapshot->dispatchId.has_value()
                  && first.snapshot->dispatchId != second.snapshot->dispatchId
                  && ((first.snapshot->dispatchId == capture::DispatchId(0)
                       && second.snapshot->dispatchId == capture::DispatchId(1))
                      || (first.snapshot->dispatchId == capture::DispatchId(1)
                          && second.snapshot->dispatchId == capture::DispatchId(0))),
              "concurrent submissions assign unique contiguous IDs in admission order");
        Check(!first.snapshot->inFlight && !second.snapshot->inFlight,
              "collector releases every reservation after terminal publication");

        const auto state = fixture.instance()->captureStateSnapshot();
        Check(!state.enabled && state.terminalRequestIssued && state.terminalTransitionCount == 1
                  && state.failure == "raw-statistics writer job was rejected by the GPU collector",
              "racing completions publish exactly one first terminal transition and diagnostic");
    }
    capture::SetCaptureFileSystemObserver(nullptr);

    Check(!observerRanOnProducer.load(std::memory_order_relaxed),
          "concurrent intercepted producers perform no filesystem publication");
    Check(observerWriter.load(std::memory_order_relaxed) != 0
              && !observerSawMultipleWorkers.load(std::memory_order_relaxed),
          "concurrent collection still confines filesystem publication to one writer thread");
    std::ifstream captureFile(root / "capture.json");
    const auto document = nlohmann::json::parse(captureFile, nullptr, false);
    Check(document.is_object() && document.value("status", "") == "error",
          "concurrent terminal transition drains a coherent error capture");
    fs::remove_all(root, cleanupError);
}

void TestLiveBackpressureAndFileSystemThreadBoundary()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("neural-statistics-live-hook-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    ResetFileSystemObserverState();
    capture::SetCaptureFileSystemObserver(ObserveFileSystem);
    {
        LiveFixture fixture;
        capture::CaptureMetadata metadata;
        metadata.layerName = "live-hook-test";
        metadata.layerVersion = "1.0.0";
        metadata.commitIdentity = "test";
        metadata.layerImplementationVersion = 1;
        metadata.statisticsMode = 0;
        metadata.dispatchFilter = "all";
        metadata.devices.push_back(capture::DeviceMetadata {capture::LogicalDeviceId(0),
                                                            "fake-device",
                                                            std::nullopt,
                                                            std::nullopt,
                                                            std::nullopt,
                                                            VK_API_VERSION_1_0});
        fixture.instance()->captureMetadata = metadata;
        fixture.instance()->dispatchFilter = DispatchFilter::All();
        fixture.instance()->captureWriter = std::make_unique<capture::CaptureWriter>(
            capture::CaptureWriterOptions{root, metadata, DispatchFilter::All(), 1});
        Check(fixture.instance()->captureWriter->start().ok(), "live-hook writer starts before producer work");

        VkPipeline pipeline = CreateTrackedPipeline(fixture, true);
        const VkDataGraphPipelineSessionCreateInfoARM sessionInfo{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
            nullptr,
            0,
            pipeline,
        };
        VkDataGraphPipelineSessionARM session = VK_NULL_HANDLE;
        Check(layer_vkCreateDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), &sessionInfo, nullptr, &session)
                  == VK_SUCCESS,
              "live-hook captured session setup succeeds");
        auto sessionRecord = fixture.device()->resourceManager.getSessionRecord(session);
        Check(sessionRecord != nullptr && sessionRecord->statisticsCaptureActive,
              "live-hook captured session has active statistics state");

        auto drained = fixture.instance()->captureWriter->publishDocuments();
        Check(drained.accepted && drained.wait().ok(), "setup writer work drains before controlled backpressure");

        fixture.device()->resourceManager.addCommandPoolRecord(CommandPoolHandle(), 0, 0);
        auto command =
            fixture.device()->resourceManager.addCommandBufferRecord(CommandBufferHandle(), CommandPoolHandle());
        for (int i = 0; i < 3; ++i)
        {
            auto snapshot = std::make_shared<StatsSnapshot>(DeviceHandle(), FakeDestroyBuffer, FakeFreeMemory);
            snapshot->session = sessionRecord;
            snapshot->queueFamilyIndex = 0;
            snapshot->dataSize = 0;
            snapshot->allocationSize = 0;
            snapshot->memoryProperties = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            command->dispatches.push_back({sessionRecord, std::move(snapshot), {}});
        }

        capture::fault::ScopedGlobalPause pause(capture::fault::Point::WriterWriteTemporaryFile);
        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer(
            [&]
            {
                observerProducer1.store(CurrentThreadHash(), std::memory_order_relaxed);
                submitResult = SubmitLegacy();
            });
        pause.waitUntilReached();
        Check(fixture.instance()->captureWriter->waitForBlockedSubmitters(1),
              "tiny live writer queue applies deterministic hook-thread backpressure");
        pause.release();
        producer.join();
        Check(submitResult == VK_SUCCESS, "accepted live submission returns success after backpressure clears");

        layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(DeviceHandle(), session, nullptr);
        layer_vkDestroyPipeline<user_tag>(DeviceHandle(), pipeline, nullptr);
        // Fixture destruction now exercises device/instance destruction while
        // raw and metadata work may still be queued; instance shutdown drains it.
    }
    capture::SetCaptureFileSystemObserver(nullptr);

    Check(!observerRanOnProducer.load(std::memory_order_relaxed),
          "intercepted Vulkan producer threads execute no CaptureFileSystem operation");
    Check(observerWriter.load(std::memory_order_relaxed) != 0
              && !observerSawMultipleWorkers.load(std::memory_order_relaxed),
          "all observed capture filesystem operations execute on one writer thread");

    bool hasInternalFile = false;
    for (const auto &entry : fs::recursive_directory_iterator(root))
    {
        const std::string name = entry.path().filename().string();
        hasInternalFile |= name.rfind(".capture-internal-", 0) == 0 || (name.size() >= 4 && name.ends_with(".tmp"));
    }
    Check(!hasInternalFile, "drained live capture leaves no temporary or internal publication files");
    std::ifstream captureFile(root / "capture.json");
    const auto document = nlohmann::json::parse(captureFile, nullptr, false);
    Check(document.is_object() && document.value("status", "") == "complete",
          "destruction with queued writer work publishes complete capture status");
    fs::remove_all(root, cleanupError);
}

void TestCaptureCommitLinearization()
{
    {
        LiveFixture fixture;
        std::latch callbackEntered {1};
        std::latch releaseCallback {1};
        std::latch disableStarted {1};
        std::promise<bool> disablePromise;
        auto disableResult = disablePromise.get_future();
        std::atomic<uint32_t> callbackCalls {0};
        bool committed = false;

        std::thread committer(
            [&]
            {
                std::unique_lock admission {fixture.instance()->submissionAdmissionMutex};
                std::unique_lock vulkan {g_vulkanLock};
                committed = fixture.instance()->commitCaptureIfEnabled(
                    [&]() noexcept
                    {
                        callbackCalls.fetch_add(1, std::memory_order_relaxed);
                        callbackEntered.count_down();
                        releaseCallback.wait();
                        fixture.instance()->dispatchIds.commit();
                    });
            });
        callbackEntered.wait();

        std::thread disabler(
            [&]
            {
                disableStarted.count_down();
                disablePromise.set_value(
                    fixture.instance()->disableCapture("capture commit linearization test terminal transition"));
            });
        disableStarted.wait();
        const bool disableBlocked =
            disableResult.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout;
        releaseCallback.count_down();
        committer.join();
        const bool terminalized = disableResult.get();
        disabler.join();

        const auto state = fixture.instance()->captureStateSnapshot();
        Check(disableBlocked && committed && callbackCalls.load(std::memory_order_relaxed) == 1
                  && fixture.instance()->dispatchIds.nextValue() == 1 && terminalized && !state.enabled
                  && state.terminalTransitionCount == 1,
              "capture commit linearizes before a terminal transition blocked on the submission gate");
    }

    {
        LiveFixture fixture;
        Check(fixture.instance()->disableCapture("capture commit disable-first test terminal transition"),
              "disable-first capture commit setup terminalizes capture");
        std::atomic<bool> callbackRan {false};
        const bool committed = fixture.instance()->commitCaptureIfEnabled(
            [&]() noexcept
            {
                callbackRan.store(true, std::memory_order_relaxed);
                fixture.instance()->dispatchIds.commit();
            });
        Check(!committed && !callbackRan.load(std::memory_order_relaxed)
                  && fixture.instance()->dispatchIds.nextValue() == 0,
              "capture terminal transition linearizes before and suppresses a later identity commit");
    }

    {
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-writer-commit-first-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            std::latch callbackEntered {1};
            std::latch releaseCallback {1};
            std::latch terminalStarted {1};
            std::promise<bool> terminalAcceptedPromise;
            auto terminalAcceptedResult = terminalAcceptedPromise.get_future();
            std::atomic<bool> terminalCompleted {false};
            std::atomic<uint32_t> callbackCalls {0};
            bool committed = false;

            std::thread committer(
                [&]
                {
                    std::unique_lock admission {fixture.instance()->submissionAdmissionMutex};
                    std::unique_lock vulkan {g_vulkanLock};
                    committed = fixture.instance()->commitCaptureIfEnabled(
                        [&]() noexcept
                        {
                            callbackCalls.fetch_add(1, std::memory_order_relaxed);
                            callbackEntered.count_down();
                            releaseCallback.wait();
                            fixture.instance()->dispatchIds.commit();
                        });
                });
            callbackEntered.wait();

            std::thread terminator(
                [&]
                {
                    terminalStarted.count_down();
                    auto terminal = fixture.instance()->captureWriter->finishError(
                        "writer terminal transition during capture identity commit");
                    terminalAcceptedPromise.set_value(terminal.accepted);
                    if (terminal.accepted)
                    {
                        terminalCompleted.store(terminal.wait().ok(), std::memory_order_relaxed);
                    }
                });
            terminalStarted.wait();
            const bool terminalBlocked =
                terminalAcceptedResult.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout;
            releaseCallback.count_down();
            committer.join();
            const bool terminalAccepted = terminalAcceptedResult.get();
            terminator.join();

            const auto writerSnapshot = fixture.instance()->captureWriter->snapshot();
            Check(terminalBlocked && committed && callbackCalls.load(std::memory_order_relaxed) == 1
                      && fixture.instance()->dispatchIds.nextValue() == 1 && terminalAccepted
                      && terminalCompleted.load(std::memory_order_relaxed)
                      && writerSnapshot.state == capture::WriterState::Error && !writerSnapshot.accepting,
                  "identity commit linearizes before an asynchronous writer terminal transition");
        }
        fs::remove_all(root, error);
    }

    {
        const fs::path root =
            fs::temp_directory_path() / ("neural-statistics-writer-terminal-first-" + UniqueTestSuffix());
        std::error_code error;
        fs::remove_all(root, error);
        {
            LiveFixture fixture;
            StartCaptureWriter(fixture, root);
            auto terminal =
                fixture.instance()->captureWriter->finishError("writer terminal-first identity commit test");
            Check(terminal.accepted, "writer terminal-first setup reserves the terminal job");

            std::atomic<bool> callbackRan {false};
            const bool committed = fixture.instance()->commitCaptureIfEnabled(
                [&]() noexcept
                {
                    callbackRan.store(true, std::memory_order_relaxed);
                    fixture.instance()->dispatchIds.commit();
                });
            const bool terminalCompleted = terminal.accepted && terminal.wait().ok();
            const auto writerSnapshot = fixture.instance()->captureWriter->snapshot();
            Check(!committed && !callbackRan.load(std::memory_order_relaxed)
                      && fixture.instance()->dispatchIds.nextValue() == 0 && terminalCompleted
                      && writerSnapshot.state == capture::WriterState::Error && !writerSnapshot.accepting,
                  "an accepted writer terminal job suppresses a later identity commit");
        }
        fs::remove_all(root, error);
    }
}

void TestMidCallTerminalCaptureSubmissionBypass()
{
    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1220, 1221);
        occurrence.snapshot->executionIndex = 77;
        ArmApplicationSubmitPause();

        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer([&] { submitResult = SubmitLegacy(); });
        const bool terminalizedWhileUnlocked = FinishApplicationSubmitPause(producer, fixture.instance());

        Check(terminalizedWhileUnlocked,
              "legacy application submit releases the global lock during the deterministic terminal transition");
        Check(submitResult == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers.size() == 2,
              "legacy mid-call terminal transition preserves the accepted rewritten downstream submit");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->executionIndex == 77
                  && occurrence.snapshot->dispatchId == std::nullopt,
              "legacy mid-call terminal transition commits no capture identity or session index");
        Check(!occurrence.snapshot->inFlight && fixture.instance()->collectorJobCount() == 0
                  && fixture.device()->deferredSubmissionArchives.size() == 1
                  && fixture.device()->deferredSubmissionSnapshots.size() == 2,
              "legacy mid-call terminal transition cancels reservations and defers GPU-used resources");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1230, 1231);
        occurrence.snapshot->executionIndex = 88;
        const VkCommandBufferSubmitInfo commandInfo {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            nullptr,
            CommandBufferHandle(),
            0,
        };
        const VkSubmitInfo2 submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            nullptr,
            0,
            0,
            nullptr,
            1,
            &commandInfo,
            0,
            nullptr,
        };
        ArmApplicationSubmitPause();

        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer(
            [&] { submitResult = layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE); });
        const bool terminalizedWhileUnlocked = FinishApplicationSubmitPause(producer, fixture.instance());

        Check(terminalizedWhileUnlocked,
              "submit2 application submit releases the global lock during the deterministic terminal transition");
        Check(submitResult == VK_SUCCESS && driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 0,
              "submit2 mid-call terminal transition preserves the accepted rewritten downstream submit");
        Check(driver.lastSubmit2CommandBuffers.size() == 2
                  && driver.lastSubmit2CommandBuffers[0] == CommandBufferHandle(),
              "submit2 mid-call terminal transition retains the rewritten archive submission already accepted");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->executionIndex == 88
                  && occurrence.snapshot->dispatchId == std::nullopt,
              "submit2 mid-call terminal transition commits no capture identity or session index");
        Check(!occurrence.snapshot->inFlight && fixture.instance()->collectorJobCount() == 0
                  && fixture.device()->deferredSubmissionArchives.size() == 1
                  && fixture.device()->deferredSubmissionSnapshots.size() == 2,
              "submit2 mid-call terminal transition cancels reservations and defers GPU-used resources");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1240, 1241);
        occurrence.snapshot->executionIndex = 99;
        fixture.device()->driver.vkDestroyBuffer = PausingSnapshotDestroyBuffer;
        ArmBufferCreatePause();
        ArmSnapshotRetirementProbe();

        VkResult submitResult = VK_ERROR_UNKNOWN;
        bool collectorConstructionFaultRemainedArmed = false;
        std::thread producer(
            [&]
            {
                capture::fault::ScopedInjection injection(capture::fault::Point::CollectorJobConstruction,
                                                          capture::fault::Failure::BadAllocation);
                submitResult = SubmitLegacy();
                try
                {
                    capture::fault::Checkpoint(capture::fault::Point::CollectorJobConstruction);
                }
                catch (const std::bad_alloc&)
                {
                    collectorConstructionFaultRemainedArmed = true;
                }
            });
        const bool creationUnlocked = ReleaseBufferCreatePause(fixture.instance());
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);
        fixture.device()->driver.vkDestroyBuffer = FakeDestroyBuffer;

        Check(creationUnlocked && retiredUnlocked && collectorConstructionFaultRemainedArmed,
              "legacy terminal-during-materialization retires staged resources outside the global lock");
        Check(submitResult == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "legacy terminal-during-materialization forwards only the original application submit");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->executionIndex == 99
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight
                  && fixture.device()->deferredSubmissionArchives.empty()
                  && fixture.device()->deferredSubmissionSnapshots.empty(),
              "legacy terminal-during-materialization cancels staged capture state without deferral");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1250, 1251);
        occurrence.snapshot->executionIndex = 111;
        const VkCommandBufferSubmitInfo commandInfo {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            nullptr,
            CommandBufferHandle(),
            0,
        };
        const VkSubmitInfo2 submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            nullptr,
            0,
            0,
            nullptr,
            1,
            &commandInfo,
            0,
            nullptr,
        };
        fixture.device()->driver.vkDestroyBuffer = PausingSnapshotDestroyBuffer;
        ArmBufferCreatePause();
        ArmSnapshotRetirementProbe();

        VkResult submitResult = VK_ERROR_UNKNOWN;
        bool collectorConstructionFaultRemainedArmed = false;
        std::thread producer(
            [&]
            {
                capture::fault::ScopedInjection injection(capture::fault::Point::CollectorJobConstruction,
                                                          capture::fault::Failure::BadAllocation);
                submitResult = layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE);
                try
                {
                    capture::fault::Checkpoint(capture::fault::Point::CollectorJobConstruction);
                }
                catch (const std::bad_alloc&)
                {
                    collectorConstructionFaultRemainedArmed = true;
                }
            });
        const bool creationUnlocked = ReleaseBufferCreatePause(fixture.instance());
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);
        fixture.device()->driver.vkDestroyBuffer = FakeDestroyBuffer;

        Check(creationUnlocked && retiredUnlocked && collectorConstructionFaultRemainedArmed,
              "submit2 terminal-during-materialization retires staged resources outside the global lock");
        Check(submitResult == VK_SUCCESS && driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 0
                  && driver.lastSubmit2CommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "submit2 terminal-during-materialization forwards only the original application submit");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->executionIndex == 111
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight
                  && fixture.device()->deferredSubmissionArchives.empty()
                  && fixture.device()->deferredSubmissionSnapshots.empty(),
              "submit2 terminal-during-materialization cancels staged capture state without deferral");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1260, 1261);
        driver.bufferCreateResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        ArmBufferCreatePause();

        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer([&] { submitResult = SubmitLegacy(); });
        const bool creationUnlocked = FinishBufferCreatePause(producer, fixture.instance());

        Check(creationUnlocked && submitResult == VK_SUCCESS && driver.legacyApplicationSubmits == 1
                  && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "terminal capture takes precedence over a concurrent materialization device-allocation failure");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight
                  && fixture.device()->deferredSubmissionArchives.empty()
                  && fixture.device()->deferredSubmissionSnapshots.empty(),
              "terminal materialization failure forwards exactly without capture bookkeeping or deferral");
    }
}

void TestTerminalDuringSubmitRewriteFailureBypass()
{
    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1270, 1271);
        fixture.device()->driver.vkDestroyBuffer = PausingSnapshotDestroyBuffer;
        ArmSnapshotRetirementProbe();
        capture::fault::ScopedGlobalPause pause(capture::fault::Point::LegacySubmitRewrite);
        capture::fault::ScopedGlobalInjection injection(capture::fault::Point::LegacySubmitRewrite,
                                                        capture::fault::Failure::BadAllocation);

        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer([&] { submitResult = SubmitLegacy(); });
        pause.waitUntilReached();
        const bool terminalized =
            fixture.instance()->disableCapture("capture terminalized during legacy submit rewrite");
        pause.release();
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);
        fixture.device()->driver.vkDestroyBuffer = FakeDestroyBuffer;

        Check(terminalized && retiredUnlocked && submitResult == VK_SUCCESS
                  && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "legacy rewrite bad-allocation after terminal capture exact-forwards the original submit");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight
                  && fixture.device()->deferredSubmissionArchives.empty()
                  && fixture.device()->deferredSubmissionSnapshots.empty(),
              "legacy terminal rewrite failure retires staged owners unlocked without capture bookkeeping");
    }

    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1280, 1281);
        const VkCommandBufferSubmitInfo commandInfo {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            nullptr,
            CommandBufferHandle(),
            0,
        };
        const VkSubmitInfo2 submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            nullptr,
            0,
            0,
            nullptr,
            1,
            &commandInfo,
            0,
            nullptr,
        };
        fixture.device()->driver.vkDestroyBuffer = PausingSnapshotDestroyBuffer;
        ArmSnapshotRetirementProbe();
        capture::fault::ScopedGlobalPause pause(capture::fault::Point::Submit2Rewrite);
        capture::fault::ScopedGlobalInjection injection(capture::fault::Point::Submit2Rewrite,
                                                        capture::fault::Failure::Unexpected);

        VkResult submitResult = VK_ERROR_UNKNOWN;
        std::thread producer(
            [&] { submitResult = layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE); });
        pause.waitUntilReached();
        const bool terminalized =
            fixture.instance()->disableCapture("capture terminalized during submit2 rewrite");
        pause.release();
        const bool retiredUnlocked = FinishSnapshotRetirementProbe(producer);
        fixture.device()->driver.vkDestroyBuffer = FakeDestroyBuffer;

        Check(terminalized && retiredUnlocked && submitResult == VK_SUCCESS && driver.coreSubmit2Calls == 1
                  && driver.coreMarkerSubmit2Calls == 0
                  && driver.lastSubmit2CommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "submit2 rewrite unexpected exception after terminal capture exact-forwards the original submit");
        Check(occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight
                  && fixture.device()->deferredSubmissionArchives.empty()
                  && fixture.device()->deferredSubmissionSnapshots.empty(),
              "submit2 terminal rewrite failure retires staged owners unlocked without capture bookkeeping");
    }
}

void TestWriterTerminalSubmissionBypass()
{
    namespace fs = std::filesystem;
    const fs::path root =
        fs::temp_directory_path() / ("neural-statistics-writer-terminal-submit-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1290, 1291);
        auto terminal = fixture.instance()->captureWriter->finishError("writer terminal before application submit");
        Check(terminal.accepted && terminal.wait().ok(),
              "writer-terminal submit setup publishes the independent writer failure");

        const auto before = fixture.instance()->captureStateSnapshot();
        Check(before.enabled && !fixture.instance()->isCaptureEnabled(),
              "writer terminal state disables operational capture while preserving the raw lifecycle diagnostic");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers == std::vector<VkCommandBuffer> {CommandBufferHandle()},
              "writer terminal before submit exact-forwards the original legacy application work");
        const auto after = fixture.instance()->captureStateSnapshot();
        Check(after.enabled && after.terminalTransitionCount == 0
                  && occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt && !occurrence.snapshot->inFlight,
              "writer-terminal submit bypass preserves raw diagnostics and commits no capture bookkeeping");
    }

    fs::remove_all(root, cleanupError);
}
void TestWriterTerminalDeviceBypass()
{
    namespace fs = std::filesystem;
    const fs::path root =
        fs::temp_directory_path() / ("neural-statistics-writer-terminal-device-" + UniqueTestSuffix());
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);

    {
        LiveFixture fixture;
        StartCaptureWriter(fixture, root);
        auto terminal = fixture.instance()->captureWriter->finishError("writer terminal before device creation");
        Check(terminal.accepted && terminal.wait().ok(),
              "writer-terminal device setup publishes the independent writer failure");
        const auto rawBefore = fixture.instance()->captureStateSnapshot();
        Check(rawBefore.enabled && rawBefore.terminalTransitionCount == 0 && !fixture.instance()->isCaptureEnabled(),
              "writer terminal is operationally disabled while retaining raw lifecycle diagnostics");

        VkLayerDeviceLink deviceLink {};
        VkLayerDeviceCreateInfo deviceLoaderInfo {};
        VkDeviceQueueCreateInfo queueInfo {};
        float priority = 0.0f;
        auto deviceInfo = MakeDeviceCreateInfo(deviceLink, deviceLoaderInfo, queueInfo, priority);
        VkPhysicalDeviceSynchronization2Features applicationSynchronization2 {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            nullptr,
            VK_FALSE,
        };
        deviceLoaderInfo.pNext = &applicationSynchronization2;
        const uint32_t extensionQueriesBefore = driver.deviceExtensionQueries;
        const uint32_t featureQueriesBefore = driver.physicalDeviceFeatureQueries;
        const uint32_t propertyQueriesBefore = driver.physicalDevicePropertyQueries;
        const uint32_t queueQueriesBefore = driver.queueFamilyPropertyQueries;
        driver.deviceCreates = 1;

        VkDevice secondDevice = VK_NULL_HANDLE;
        VkResult createResult = VK_ERROR_UNKNOWN;
        {
            capture::fault::ScopedInjection injection(
                capture::fault::Point::DeviceAfterDownstreamCreateBeforePublication);
            createResult = layer_vkCreateDevice<user_tag>(PhysicalDeviceHandle(), &deviceInfo, nullptr, &secondDevice);
        }
        Check(createResult == VK_SUCCESS && secondDevice == DeviceHandle1() && driver.deviceCreates == 2,
              "writer-terminal capture forwards a later valid logical-device creation");
        // The framework still enumerates extensions and queries the device API version.
        Check(driver.deviceExtensionQueries == extensionQueriesBefore + 2
                  && driver.physicalDeviceFeatureQueries == featureQueriesBefore
                  && driver.physicalDevicePropertyQueries == propertyQueriesBefore + 1
                  && driver.queueFamilyPropertyQueries == queueQueriesBefore,
              "writer-terminal device creation performs only framework-required capability queries");
        Check(driver.lastDeviceCreateEnabledExtensionCount == 0 && !driver.requirements2ExtensionEnabled
                  && !driver.neuralStatisticsExtensionEnabled && !driver.neuralStatisticsFeatureEnabled
                  && !driver.synchronization2FeatureEnabled && applicationSynchronization2.synchronization2 == VK_FALSE,
              "writer-terminal device creation forwards application extensions and features unchanged");
        Device* terminalDevice = Device::retrieve(secondDevice);
        const auto rawAfter = fixture.instance()->captureStateSnapshot();
        Check(terminalDevice != nullptr && terminalDevice->collector == nullptr
                  && terminalDevice->queueFamilyProperties.empty() && !terminalDevice->neuralStatisticsEnabled
                  && !terminalDevice->synchronization2Enabled && rawAfter.enabled
                  && rawAfter.terminalTransitionCount == 0 && !fixture.instance()->isCaptureEnabled(),
              "writer-terminal logical device retains dispatch tracking without capture-only state");
        layer_vkDestroyDevice<user_tag>(secondDevice, nullptr);
    }

    fs::remove_all(root, cleanupError);
}

void TestTerminalCaptureSubmissionBypass()
{
    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1200, 1201);
        auto command = fixture.device()->resourceManager.getCommandBufferRecord(CommandBufferHandle());
        command->snapshotError = "stale capture metadata must be ignored after terminal failure";

        Check(fixture.instance()->disableCapture("intentional terminal test failure"),
              "legacy bypass setup terminalizes capture");
        Check(SubmitLegacy() == VK_SUCCESS && driver.legacyApplicationSubmits == 1 && driver.markerSubmits == 0
                  && driver.lastLegacyCommandBuffers.size() == 1
                  && driver.lastLegacyCommandBuffers[0] == CommandBufferHandle()
                  && occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt,
              "terminal capture forwards the original legacy submit without capture bookkeeping");
    }

    const VkCommandBufferSubmitInfo commandInfo {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 nullptr,
                                                 CommandBufferHandle(),
                                                 0};
    const VkSubmitInfo2 submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        nullptr,
        0,
        0,
        nullptr,
        1,
        &commandInfo,
        0,
        nullptr,
    };
    {
        LiveFixture fixture;
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1210, 1211);
        Check(fixture.instance()->disableCapture("intentional terminal test failure"),
              "submit2 bypass setup terminalizes capture");
        Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS
                  && driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 0
                  && occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                  && occurrence.snapshot->dispatchId == std::nullopt,
              "terminal capture forwards the original submit2 call without capture bookkeeping");
    }
}

void TestSubmit2Routes()
{
    const VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 nullptr,
                                                 CommandBufferHandle(),
                                                 0};
    const VkSubmitInfo2 submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &commandInfo, 0, nullptr,
    };

    {
        LiveFixture fixture(true, true, true, true);
        fixture.instance()->dispatchFilter = DispatchFilter::Single(99);
        AddSessionOccurrence(fixture, "route-test");
        Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS
                  && driver.coreSubmit2Calls == 1 && driver.khrSubmit2Calls == 0,
              "core submit2 entrypoint selects only the exact core downstream function");
    }
    {
        LiveFixture fixture(true, true, true, true);
        fixture.instance()->dispatchFilter = DispatchFilter::Single(99);
        AddSessionOccurrence(fixture, "route-test");
        Check(layer_vkQueueSubmit2KHR<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS
                  && driver.coreSubmit2Calls == 0 && driver.khrSubmit2Calls == 1,
              "KHR submit2 entrypoint selects only the exact KHR downstream function");
    }
    {
        LiveFixture fixture(true, true, true, true);
        AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1300, 1301);
        Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS,
              "selected core submit2 application submission succeeds");
        fixture.instance()->waitForCollectorJobs();
        Check(driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 1 && driver.khrSubmit2Calls == 0
                  && driver.khrMarkerSubmit2Calls == 0,
              "selected core submit2 uses the core route for application and marker submissions");
    }

    {
        LiveFixture fixture(true, true, true, true);
        auto occurrence = AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1305, 1306);
        {
            capture::fault::ScopedInjection injection(capture::fault::Point::Submit2Rewrite,
                                                      capture::fault::Failure::BadAllocation);
            Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE)
                          == VK_ERROR_OUT_OF_HOST_MEMORY
                      && driver.coreSubmit2Calls == 0 && driver.coreMarkerSubmit2Calls == 0
                      && occurrence.session->nextExecutionIndex == 0 && fixture.instance()->dispatchIds.nextValue() == 0
                      && occurrence.snapshot->dispatchId == std::nullopt,
                  "submit2 rewrite allocation failure is translated and rolls back the plan before forwarding");
        }
        Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS,
              "submit2 remains reusable after rewrite-allocation rollback");
        fixture.instance()->waitForCollectorJobs();
        Check(driver.coreSubmit2Calls == 1 && driver.coreMarkerSubmit2Calls == 1
                  && occurrence.session->nextExecutionIndex == 1 && fixture.instance()->dispatchIds.nextValue() == 1,
              "retry after submit2 rewrite failure commits exactly one capture identity");
    }

    {
        LiveFixture fixture(true, true, true, true);
        AddSelectedOccurrence(fixture, CommandPoolHandle(), CommandBufferHandle(), 1310, 1311);
        Check(layer_vkQueueSubmit2KHR<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS,
              "selected KHR submit2 application submission succeeds");
        fixture.instance()->waitForCollectorJobs();
        Check(driver.khrSubmit2Calls == 1 && driver.khrMarkerSubmit2Calls == 1 && driver.coreSubmit2Calls == 0
                  && driver.coreMarkerSubmit2Calls == 0,
              "selected KHR submit2 uses the KHR route for application and marker submissions");
    }
    {
        LiveFixture fixture(true, true, false, true);
        fixture.instance()->dispatchFilter = DispatchFilter::Single(99);
        AddSessionOccurrence(fixture, "route-test");
        Check(layer_vkQueueSubmit2<user_tag>(QueueHandle(), 1, &submit, VK_NULL_HANDLE) == VK_ERROR_UNKNOWN
                  && driver.coreSubmit2Calls == 0 && driver.khrSubmit2Calls == 0,
              "missing core route returns an error without falling through to KHR");
    }
}
} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--device-capabilities") == 0)
    {
        TestNeuralStatisticsDeviceCapabilityPatch();
        TestNeuralStatisticsDeviceCreation();
        if (failures != 0)
        {
            std::cerr << failures << " device-capability test(s) failed\n";
            return 1;
        }
        std::cout << "All device-capability tests passed\n";
        return 0;
    }

    if (argc == 3 && std::strcmp(argv[1], "--turnover-stress") == 0)
    {
        uint32_t repetitions = 0;
        try
        {
            repetitions = static_cast<uint32_t>(std::stoul(argv[2]));
        }
        catch (...)
        {
            std::cerr << "invalid turnover stress repetition count\n";
            return 2;
        }
        if (repetitions == 0)
        {
            std::cerr << "turnover stress repetition count must be nonzero\n";
            return 2;
        }
        for (uint32_t i = 0; i < repetitions; ++i)
        {
            TestCollectorCapacityTurnover();
        }
        if (failures != 0)
        {
            std::cerr << failures << " turnover stress test(s) failed\n";
            return 1;
        }
        std::cout << "Collector turnover stress passed " << repetitions << " repetitions\n";
        return 0;
    }

    TestSafeMicromapUsageDeepCopy();
    TestNeuralStatisticsDeviceCapabilityPatch();
    TestNeuralStatisticsDeviceCreation();
    TestInstanceCreateObservations();
    TestInstanceWriterAndMultipleDevices();
    TestProtectedPipelinePassThrough();
    TestUnsupportedNeuralStatisticsPassThrough();
    TestShaderTransactions();
    TestCommandBufferPublicationTransactions();
    TestTerminalCaptureObjectPassThrough();
    TestMidCallTerminalCaptureObjectPassThrough();
    TestPipelineTransactionsAndPartialResults();
    TestPostIdentityPipelinePublicationFailure();
    TestSessionFallbackAndRequirements2Route();
    TestCaptureAllocationsReleaseGlobalLock();
    TestRecordingFailureTerminalBypass();
    TestStatisticsAliasQueueFamilySharing();
    TestFilterBeforeLegalityAndSubmissionTransactions();
    TestLegacyDeviceGroupSubmitRewrite();
    TestSecondaryOccurrenceArchiving();
    TestSnapshotOwnerLockBoundaries();
    TestSnapshotReusePolicies();
    TestAsynchronousCollectorOrderingAndReuse();
    TestCollectorCapacityTurnover();
    TestDeviceWaitFailureShutdown();
    TestCollectorFailuresBackpressureAndLifetime();
    TestRawWriterEnqueueFailure();
    TestConcurrentCollectionTerminalTransition();
    TestLiveBackpressureAndFileSystemThreadBoundary();
    TestCaptureCommitLinearization();
    TestMidCallTerminalCaptureSubmissionBypass();
    TestTerminalDuringSubmitRewriteFailureBypass();
    TestWriterTerminalSubmissionBypass();
    TestWriterTerminalDeviceBypass();
    TestTerminalCaptureSubmissionBypass();
    TestSubmit2Routes();

    if (failures != 0)
    {
        std::cerr << failures << " live-hook test(s) failed\n";
        return 1;
    }
    std::cout << "All live-hook tests passed\n";
    return 0;
}
