/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "device.hpp"
#include "framework/instance_functions.hpp"
#include "framework/utils.hpp"
#include "instance.hpp"
#include "instance_creation_transaction.hpp"
#include "layer_options.hpp"
#include "version.hpp"
#include "vulkan_exception_policy.hpp"

#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

#include <vulkan/utility/vk_struct_helper.hpp>

extern std::mutex g_vulkanLock;

namespace
{
static_assert(std::is_nothrow_move_assignable_v<LayerOptionsPtr>);
static_assert(std::is_nothrow_move_assignable_v<std::unique_ptr<capture::CaptureWriter>>);
static_assert(std::is_nothrow_move_assignable_v<capture::CaptureMetadata>);

capture::CaptureMetadata MakeInitialMetadata(const LayerOptions &options)
{
    capture::CaptureMetadata metadata;
    metadata.layerName = LGL_LAYER_NAME;
    metadata.layerVersion = std::to_string(LGL_VER_MAJOR) + "." + std::to_string(LGL_VER_MINOR) +
                            "." + std::to_string(LGL_VER_PATCH);
#ifdef LGL_COMMIT_ID
    metadata.commitIdentity = LGL_COMMIT_ID;
#endif
    metadata.layerImplementationVersion = 1;
    metadata.statisticsMode = options.getStatsModeIndex();
    metadata.dispatchFilter = options.getDispatchFilterText();
    return metadata;
}

VkResult WriterStartFailureResult(const capture::WriterError &error) noexcept
{
    return error.code == capture::WriterErrorCode::OutOfHostMemory ? VK_ERROR_OUT_OF_HOST_MEMORY
                                                                   : VK_ERROR_INITIALIZATION_FAILED;
}

void FinalizeLocalWriterError(capture::CaptureWriter &writer, const char *message) noexcept
{
    auto terminal = writer.finishError(message);
    if (terminal.accepted)
    {
        (void)terminal.wait();
    }
    try
    {
        (void)writer.shutdown();
    }
    catch (...)
    {
    }
}

void DisableCaptureAfterForward(Instance &instance, const char *message) noexcept
{
    (void)instance.disableCapture(message);
}

void PublishLayerOptions(VkInstance instance, LayerOptionsPtr &&layerOptions)
{
    std::lock_guard<std::mutex> lock{g_vulkanLock};
    Instance::retrieve(instance)->layerOptions = std::move(layerOptions);
}

void DestroyTrackedInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator,
                            PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr) noexcept
{
    try
    {
        layer_vkDestroyInstance<default_tag>(instance, pAllocator);
    }
    catch (...)
    {
        LAYER_ERR("Tracked instance rollback failed; destroying the downstream instance directly");
        DestroyUntrackedCreatedInstance(instance, pAllocator, nextGetInstanceProcAddr);
    }
}

constexpr InstancePublicationOperations kInstancePublicationOperations{
    PublishLayerOptions,
    DestroyTrackedInstance,
};
} // namespace

/* See Vulkan API for documentation. */
template <>
VKAPI_ATTR VkResult VKAPI_CALL
layer_vkCreateInstance<user_tag>(const VkInstanceCreateInfo *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator, VkInstance *pInstance)
{
    const VkInstance loaderProvisionalInstance = *pInstance;
    VkInstance createdInstance = VK_NULL_HANDLE;
    std::unique_ptr<capture::CaptureWriter> writer;
    try
    {
        auto layerOptions = std::make_unique<LayerOptions>(pCreateInfo);
        if (!layerOptions->isValid())
        {
            LAYER_ERR("Neural statistics capture configuration error: %s",
                      layerOptions->getError().c_str());
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        capture::CaptureMetadata metadata = MakeInitialMetadata(*layerOptions);
        writer = std::make_unique<capture::CaptureWriter>(capture::CaptureWriterOptions{
            layerOptions->getCaptureRoot(), metadata, layerOptions->getDispatchFilter(), 64});
        const auto writerStart = writer->start();
        if (!writerStart.ok())
        {
            LAYER_ERR("Structured capture writer failed to start: %s",
                      writerStart.error().message.c_str());
            return WriterStartFailureResult(writerStart.error());
        }

        const VkResult result =
            layer_vkCreateInstance<default_tag>(pCreateInfo, pAllocator, pInstance);
        if (result != VK_SUCCESS)
        {
            FinalizeLocalWriterError(*writer, "downstream vkCreateInstance failed");
            return result;
        }
        createdInstance = *pInstance;
        auto *chainInfo = getChainInfo(pCreateInfo);
        const PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr =
            chainInfo != nullptr && chainInfo->u.pLayerInfo != nullptr
                ? chainInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr
                : nullptr;

        const VkResult publication = PublishCreatedInstanceTransaction(
            createdInstance, pAllocator, pInstance, loaderProvisionalInstance,
            nextGetInstanceProcAddr, std::move(layerOptions), kInstancePublicationOperations);
        if (publication != VK_SUCCESS)
        {
            FinalizeLocalWriterError(*writer, "instance publication failed");
            return publication;
        }

        {
            std::lock_guard<std::mutex> lock{g_vulkanLock};
            auto *instance = Instance::retrieve(createdInstance);
            instance->captureMetadata = std::move(metadata);
            instance->dispatchFilter = instance->layerOptions->getDispatchFilter();
            instance->captureWriter = std::move(writer);
        }
        return VK_SUCCESS;
    }
    catch (...)
    {
        if (writer != nullptr)
        {
            FinalizeLocalWriterError(*writer, "unexpected instance creation failure");
        }
        createdInstance = *pInstance != loaderProvisionalInstance ? *pInstance : VK_NULL_HANDLE;
        if (createdInstance != VK_NULL_HANDLE)
        {
            auto *chainInfo = getChainInfo(pCreateInfo);
            const PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr =
                chainInfo != nullptr && chainInfo->u.pLayerInfo != nullptr
                    ? chainInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr
                    : nullptr;
            DestroyUntrackedCreatedInstance(createdInstance, pAllocator, nextGetInstanceProcAddr);
            *pInstance = loaderProvisionalInstance;
        }
        const auto translation = capture::abi::TranslateException(
            std::current_exception(), capture::abi::BoundaryPhase::Configuration);
        LAYER_ERR("Neural statistics capture configuration failed unexpectedly");
        return translation.result;
    }
}

template <>
VKAPI_ATTR void VKAPI_CALL
layer_vkDestroyInstance<user_tag>(VkInstance instance, const VkAllocationCallbacks *pAllocator)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto layer = Instance::destroy(instance);
    lock.unlock();

    layer->waitForCollectorJobs();
    if (layer->captureWriter != nullptr)
    {
        if (auto *completionWriter = layer->beginCaptureCompletion();
            completionWriter != nullptr &&
            completionWriter->snapshot().state == capture::WriterState::Running)
        {
            auto terminal = completionWriter->finishComplete();
            if (terminal.accepted)
            {
                (void)terminal.wait();
            }
        }
        (void)layer->captureWriter->shutdown();
    }
    layer->driver.vkDestroyInstance(instance, pAllocator);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDevice<user_tag>(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
    Instance *instance = nullptr;
    bool instrumentCaptureDevice = false;
    bool publishCaptureMetadata = false;
    VkPhysicalDeviceProperties properties{};
    {
        std::lock_guard<std::mutex> lock{g_vulkanLock};
        instance = Instance::retrieve(physicalDevice);
        instrumentCaptureDevice = instance->isCaptureEnabled();
        publishCaptureMetadata = instrumentCaptureDevice && instance->isCaptureWriterRunning();
    }

    if (publishCaptureMetadata)
    {
        instance->driver.vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    }
    VkResult result = VK_ERROR_UNKNOWN;
    try
    {
        result =
            layer_vkCreateDevice<default_tag>(physicalDevice, pCreateInfo, pAllocator, pDevice);
    }
    catch (...)
    {
        result = VK_ERROR_UNKNOWN;
    }
    if (result != VK_SUCCESS)
    {
        return result;
    }

    std::lock_guard<std::mutex> lock{g_vulkanLock};
    // Downstream device creation has succeeded. Unlike instance/shader/pipeline
    // publication, the current device policy does not destroy the logical device
    // when subsequent capture metadata admission fails.
    if (!publishCaptureMetadata)
    {
        return result;
    }
    if (!instance->isCaptureWriterRunning())
    {
        return result;
    }
    try
    {
        const capture::LogicalDeviceId id = instance->logicalDeviceIds.peek();
        Device *device = Device::retrieve(*pDevice);
        capture::CaptureMetadata prepared = instance->captureMetadata;
        prepared.devices.push_back(capture::DeviceMetadata{
            id,
            properties.deviceName,
            properties.vendorID,
            properties.deviceID,
            properties.driverVersion,
            properties.apiVersion,
        });
        if (!device->neuralStatisticsEnabled)
        {
            prepared.warnings.emplace_back(
                "neural statistics capture is unavailable for logical device " +
                std::to_string(id.value()));
        }
        auto update = instance->captureWriter->updateMetadata(prepared);
        if (!update.accepted)
        {
            DisableCaptureAfterForward(*instance,
                                       "capture metadata publication failed after downstream "
                                       "vkCreateDevice; preserving VK_SUCCESS");
            return result;
        }
        device->captureDeviceId = id;
        instance->captureMetadata = std::move(prepared);
        instance->logicalDeviceIds.commit();
    }
    catch (...)
    {
        DisableCaptureAfterForward(*instance, "capture metadata bookkeeping failed after "
                                              "downstream vkCreateDevice; preserving VK_SUCCESS");
    }
    return result;
}
