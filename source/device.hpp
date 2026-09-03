/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

/**
 * @file Declares the root class for layer management of VkDevice objects.
 *
 * Role summary
 * ============
 *
 * Devices represent the core context used by the application to connect to the
 * underlying graphics driver. A device object is the dispatch root for the
 * Vulkan driver, so device commands all take some form of dispatchable handle
 * that can be resolved into a unique per-device key. For the driver this key
 * would simply be a pointer directly to the driver-internal device object, but
 * for our layer we use a device dispatch key as an index in to the map to find
 * the layer's driver object.
 *
 * Key properties
 * ==============
 *
 * Vulkan devices are designed to be used concurrently by multiple application
 * threads. An application can have multiple concurrent devices, and use each
 * device from multiple threads.
 *
 * Access to the layer driver structures must therefore be kept thread-safe.
 * For sake of simplicity, we generally implement this by:
 *   - Holding a global lock whenever any thread is inside layer code.
 *   - Releasing the global lock whenever the layer calls a driver function.
 */

#pragma once

#include "capture_policy.hpp"
#include "framework/device_dispatch_table.hpp"
#include "framework/manual_functions.hpp"
#include "gpu_collector.hpp"
#include "instance.hpp"
#include "resource_tracking.hpp"

#include <string>
#include <vector>

#include <vulkan/utility/vk_safe_struct.hpp>
#include <vulkan/vk_layer.h>

/**
 * @brief Function pointer type for patching VkDeviceCreateInfo.
 */
using DeviceCreatePatchPtr = void (*)(Instance &instance, VkPhysicalDevice physicalDevice,
                                      vku::safe_VkDeviceCreateInfo &createInfo,
                                      std::vector<std::string> &supported);

/**
 * @brief This class implements the layer state tracker for a single device.
 */
class Device
{
  public:
    /**
     * @brief Store a new device into the global store of dispatchable devices.
     *
     * @param handle   The dispatchable device handle to use as an indirect key.
     * @param device   The @c Device object to store.
     */
    static void store(VkDevice handle, std::unique_ptr<Device> device);

    /**
     * @brief Fetch a device from the global store of dispatchable devices.
     *
     * @param handle   The dispatchable device handle to use as an indirect lookup.
     *
     * @return The layer device context.
     */
    static Device *retrieve(VkDevice handle);

    /**
     * @brief Fetch a device from the global store of dispatchable devices.
     *
     * @param handle   The dispatchable queue handle to use as an indirect lookup.
     *
     * @return The layer device context.
     */
    static Device *retrieve(VkQueue handle);

    /**
     * @brief Fetch a device from the global store of dispatchable devices.
     *
     * @param handle   The dispatchable command buffer handle to use as an indirect lookup.
     *
     * @return The layer device context.
     */
    static Device *retrieve(VkCommandBuffer handle);

    /**
     * @brief Drop a device from the global store of dispatchable devices.
     *
     * This must be called before the driver VkDevice has been destroyed, as
     * we deference the native device handle to get the dispatch key.
     *
     * @param handle   The dispatchable device handle to use as an indirect lookup.
     *
     * @return Returns the ownership of the Device object to the caller.
     */
    static std::unique_ptr<Device> destroy(VkDevice handle);

    /**
     * @brief Create a new layer device object.
     *
     * Create info is transient, so the constructor must copy what it needs.
     *
     * @param instance               The layer instance object this device is created with.
     * @param physicalDevice         The physical device this logical device is for.
     * @param device                 The device handle this device is created with.
     * @param nlayerGetProcAddress   The vkGetProcAddress function in the driver/next layer down.
     * @param createInfo             The create info used to create the device.
     */
    Device(Instance *instance, VkPhysicalDevice physicalDevice, VkDevice device,
           PFN_vkGetDeviceProcAddr nlayerGetProcAddress, const VkDeviceCreateInfo &createInfo,
           capture::GpuCollectorOptions collectorOptions = {});

    /**
     * @brief Destroy this layer device object.
     */
    ~Device() = default;

  public:
    /**
     * @brief The driver function dispatch table.
     */
    DeviceDispatchTable driver{};

    /**
     * @brief The minimum set of device extensions needed by this layer.
     */
    static const std::vector<DeviceCreatePatchPtr> createInfoPatches;

    /**
     * @brief The instance this device is created with.
     */
    const Instance *instance;

    /**
     * @brief The physical device this device is created with.
     */
    const VkPhysicalDevice physicalDevice;

    /**
     * @brief The device handle this device is created with.
     */
    const VkDevice device;

    /** Capture-local identity assigned before vkCreateDevice returns. */
    capture::LogicalDeviceId captureDeviceId;

    /**
     * @brief Physical queue family capabilities observed for this device.
     */
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;

    /**
     * @brief Enabled queue families that can execute data graphs and transfer commands.
     */
    std::vector<uint32_t> captureQueueFamilyIndices;

    /**
     * @brief Whether synchronization2 was enabled on this logical device.
     */
    bool synchronization2Enabled{false};

    /**
     * @brief Whether data-graph neural accelerator statistics were enabled on this device.
     */
    bool neuralStatisticsEnabled{false};

    /**
     * @brief Available ABI route for injected synchronization2 commands.
     */
    capture_policy::Synchronization2Route synchronization2Route{
        capture_policy::Synchronization2Route::Unavailable};

    /**
     * @brief Tracks active data graph pipelines and sessions.
     */
    DeviceStatsManager resourceManager;

    /** Per-device asynchronous collector bounded by admitted job count. */
    std::unique_ptr<capture::GpuCollector> collector;

    // Resources from accepted submissions whose completion marker could not be
    // created or submitted are retained until device teardown.
    std::vector<std::shared_ptr<SubmissionArchive>> deferredSubmissionArchives;
    std::vector<std::shared_ptr<StatsSnapshot>> deferredSubmissionSnapshots;
};
