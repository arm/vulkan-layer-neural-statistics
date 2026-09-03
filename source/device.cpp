/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "device.hpp"

#include "fault_injection.hpp"
#include "framework/utils.hpp"
#include "instance.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

namespace
{
bool IsDeviceExtensionEnabled(const VkDeviceCreateInfo &createInfo, const char *extension)
{
    for (uint32_t i = 0; i < createInfo.enabledExtensionCount; ++i)
    {
        if (std::strcmp(createInfo.ppEnabledExtensionNames[i], extension) == 0)
        {
            return true;
        }
    }
    return false;
}
} // namespace

void enableDeviceSynchronization2(Instance &instance, VkPhysicalDevice physicalDevice,
                                  vku::safe_VkDeviceCreateInfo &createInfo,
                                  std::vector<std::string> &supported)
{
    if (!instance.isCaptureEnabled())
    {
        return;
    }

    VkPhysicalDeviceSynchronization2Features available{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        nullptr,
        VK_FALSE,
    };
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &available,
        {},
    };
    instance.driver.vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    if (available.synchronization2 != VK_TRUE)
    {
        LAYER_ERR("Structured capture disabled: synchronization2 is not supported");
        return;
    }

    static const std::string extension{VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME};
    if (isIn(extension, supported) && vku::AddExtension(createInfo, extension.c_str()))
    {
        LAYER_LOG("Device extension added: %s", extension.c_str());
    }

    void *pNextBase = const_cast<void *>(createInfo.pNext);
    if (auto *vulkan13 = vku::FindStructInPNextChain<VkPhysicalDeviceVulkan13Features>(pNextBase))
    {
        vulkan13->synchronization2 = VK_TRUE;
        LAYER_LOG(
            "Structured capture synchronization2 enabled through VkPhysicalDeviceVulkan13Features");
        return;
    }

    if (auto *config =
            vku::FindStructInPNextChain<VkPhysicalDeviceSynchronization2Features>(pNextBase))
    {
        config->synchronization2 = VK_TRUE;
        LAYER_LOG("Structured capture synchronization2 enabled through existing feature structure");
        return;
    }

    VkPhysicalDeviceSynchronization2Features newFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        nullptr,
        VK_TRUE,
    };
    vku::AddToPnext(createInfo, newFeatures);
    LAYER_LOG("Structured capture synchronization2 feature structure added and enabled");
}

// cppcheck-suppress constParameterCallback
void enableDeviceVkKhrGetMemoryRequirements2(Instance &instance, VkPhysicalDevice physicalDevice,
                                             vku::safe_VkDeviceCreateInfo &createInfo,
                                             std::vector<std::string> &supported)
{
    UNUSED(physicalDevice);
    if (!instance.isCaptureEnabled())
    {
        return;
    }

    // VK_KHR_get_memory_requirements2 is the specification-defined route for
    // VkMemoryDedicatedRequirements on Vulkan 1.0 implementations. Enabling
    // the extension from the driver's enumerated capability list avoids an
    // unsafe legacy-requirements assumption and leaves drivers that expose
    // neither core nor KHR requirements2 conservatively noncapturable.
    static const std::string extension{VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME};
    if (!isIn(extension, supported))
    {
        LAYER_LOG("Device extension not available: %s", extension.c_str());
        return;
    }
    if (vku::AddExtension(createInfo, extension.c_str()))
    {
        LAYER_LOG("Device extension added: %s", extension.c_str());
    }
}

void enableDeviceVkArmDataGraphNeuralAcceleratorStatistics(Instance &instance,
                                                           VkPhysicalDevice physicalDevice,
                                                           vku::safe_VkDeviceCreateInfo &createInfo,
                                                           std::vector<std::string> &supported)
{
    if (!instance.isCaptureEnabled())
    {
        return;
    }

    static const std::string target{VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME};

    // Test if the desired extension is supported
    if (!isIn(target, supported))
    {
        LAYER_LOG("Device extension not available: %s", target.c_str());
        return;
    }

    VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM available =
        vku::InitStructHelper();
    VkPhysicalDeviceFeatures2 availableFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &available,
        {},
    };
    instance.driver.vkGetPhysicalDeviceFeatures2(physicalDevice, &availableFeatures);
    if (available.dataGraphNeuralAcceleratorStatistics != VK_TRUE)
    {
        LAYER_LOG("Device feature not available: dataGraphNeuralAcceleratorStatistics");
        return;
    }

    // We know we can const-cast here because createInfo is a safe-struct clone.
    void *pNextBase = const_cast<void *>(createInfo.pNext);
    VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM newFeatures =
        vku::InitStructHelper();

    // Enable the extension - this will skip adding if already enabled
    if (vku::AddExtension(createInfo, target.c_str()))
    {
        LAYER_LOG("Device extension added: %s", target.c_str());
    }

    // Check if the application already provided the neural accelerator statistics feature config.
    auto *config = vku::FindStructInPNextChain<
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(pNextBase);
    if (config)
    {
        if (!config->dataGraphNeuralAcceleratorStatistics)
        {
            LAYER_LOG("Device extension force enabled: %s", target.c_str());
            config->dataGraphNeuralAcceleratorStatistics = true;
        }
        else
        {
            LAYER_LOG("Device extension already enabled: %s", target.c_str());
        }
    }

    // Add a config if not already configured by the application
    if (!config)
    {
        newFeatures.dataGraphNeuralAcceleratorStatistics = true;
        vku::AddToPnext(createInfo, newFeatures);
        LAYER_LOG("Device extension config added: %s", target.c_str());
    }
}

/**
 * @brief The dispatch lookup for all of the created Vulkan instances.
 */
static std::unordered_map<void *, std::unique_ptr<Device>> g_devices;

/* See header for documentation. */
const std::vector<DeviceCreatePatchPtr> Device::createInfoPatches{
    enableDeviceSynchronization2, enableDeviceVkKhrGetMemoryRequirements2,
    enableDeviceVkArmDataGraphNeuralAcceleratorStatistics};

/* See header for documentation. */
void Device::store(VkDevice handle, std::unique_ptr<Device> device)
{
    void *key = getDispatchKey(handle);
    g_devices.insert({key, std::move(device)});
}

/* See header for documentation. */
Device *Device::retrieve(VkDevice handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_devices));
    return g_devices.at(key).get();
}

/* See header for documentation. */
Device *Device::retrieve(VkQueue handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_devices));
    return g_devices.at(key).get();
}

/* See header for documentation. */
Device *Device::retrieve(VkCommandBuffer handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_devices));
    return g_devices.at(key).get();
}

/* See header for documentation. */
std::unique_ptr<Device> Device::destroy(VkDevice handle)
{
    void *key = getDispatchKey(handle);
    assert(isInMap(key, g_devices));

    auto device = std::move(g_devices.at(key));
    g_devices.erase(key);
    return device;
}

/* See header for documentation. */
Device::Device(Instance *_instance, VkPhysicalDevice _physicalDevice, VkDevice _device,
               PFN_vkGetDeviceProcAddr nlayerGetProcAddress, const VkDeviceCreateInfo &createInfo,
               capture::GpuCollectorOptions collectorOptions)
    : instance(_instance), physicalDevice(_physicalDevice), device(_device)
{
    initDriverDeviceDispatchTable(device, nlayerGetProcAddress, driver);
    if (!instance->isCaptureEnabled())
    {
        return;
    }
    capture::fault::Checkpoint(capture::fault::Point::DeviceAfterDownstreamCreateBeforePublication);

    const auto *vulkan13 =
        vku::FindStructInPNextChain<VkPhysicalDeviceVulkan13Features>(createInfo.pNext);
    const auto *synchronization2 =
        vku::FindStructInPNextChain<VkPhysicalDeviceSynchronization2Features>(createInfo.pNext);
    const auto *neuralStatistics = vku::FindStructInPNextChain<
        VkPhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>(createInfo.pNext);
    synchronization2Enabled =
        (vulkan13 != nullptr && vulkan13->synchronization2 == VK_TRUE) ||
        (synchronization2 != nullptr && synchronization2->synchronization2 == VK_TRUE);
    neuralStatisticsEnabled =
        IsDeviceExtensionEnabled(createInfo,
                                 VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME) &&
        neuralStatistics != nullptr &&
        neuralStatistics->dataGraphNeuralAcceleratorStatistics == VK_TRUE;
    synchronization2Route = capture_policy::SelectSynchronization2Route(
        driver.vkCmdPipelineBarrier2, driver.vkCmdPipelineBarrier2KHR);
    const char *synchronization2RouteName = "unavailable";
    if (synchronization2Route == capture_policy::Synchronization2Route::Core)
    {
        synchronization2RouteName = "core";
    }
    else if (synchronization2Route == capture_policy::Synchronization2Route::Khr)
    {
        synchronization2RouteName = "KHR";
    }
    LAYER_LOG("Structured capture synchronization2 enabled=%u command-route=%s",
              synchronization2Enabled, synchronization2RouteName);
    LAYER_LOG("Structured capture neural statistics enabled=%u", neuralStatisticsEnabled);
    LAYER_LOG("Structured capture submit2 downstream availability: core=%u KHR=%u",
              driver.vkQueueSubmit2 != nullptr, driver.vkQueueSubmit2KHR != nullptr);
    LAYER_LOG(
        "Structured capture memory-requirements2 downstream availability: core=%u KHR=%u legacy=%u",
        driver.vkGetBufferMemoryRequirements2 != nullptr,
        driver.vkGetBufferMemoryRequirements2KHR != nullptr,
        driver.vkGetBufferMemoryRequirements != nullptr);

    uint32_t queueFamilyCount = 0;
    instance->driver.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                                              nullptr);
    queueFamilyProperties.resize(queueFamilyCount);
    instance->driver.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                                              queueFamilyProperties.data());
    queueFamilyProperties.resize(queueFamilyCount);

    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        const auto &props = queueFamilyProperties[i];
        LAYER_LOG("Structured capture queue family %u: flags=0x%08X count=%u graphics=%u "
                  "compute=%u transfer=%u dataGraph=%u",
                  i, static_cast<unsigned>(props.queueFlags), props.queueCount,
                  (props.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0,
                  (props.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0,
                  (props.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0,
                  (props.queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM) != 0);
    }

    constexpr VkQueueFlags transferCapable =
        VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < createInfo.queueCreateInfoCount; ++i)
    {
        const auto &queueInfo = createInfo.pQueueCreateInfos[i];
        LAYER_LOG("Structured capture device requested queue family %u with %u queue(s)",
                  queueInfo.queueFamilyIndex, queueInfo.queueCount);
        if (queueInfo.queueCount == 0 || queueInfo.queueFamilyIndex >= queueFamilyProperties.size())
        {
            continue;
        }
        const auto flags = queueFamilyProperties[queueInfo.queueFamilyIndex].queueFlags;
        if ((flags & VK_QUEUE_DATA_GRAPH_BIT_ARM) != 0 && (flags & transferCapable) != 0)
        {
            captureQueueFamilyIndices.emplace_back(queueInfo.queueFamilyIndex);
        }
    }
    LAYER_LOG("Structured capture enabled data-graph/transfer queue-family count=%u",
              static_cast<uint32_t>(captureQueueFamilyIndices.size()));

    collector = std::make_unique<capture::GpuCollector>(*const_cast<Instance *>(instance), device,
                                                        driver, collectorOptions);
}
