/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "instance_creation_transaction.hpp"

#include "fault_injection.hpp"
#include "vulkan_exception_policy.hpp"

#include <exception>
#include <type_traits>

static_assert(std::is_nothrow_move_assignable_v<LayerOptionsPtr>);

VKAPI_ATTR VkResult VKAPI_CALL PublishCreatedInstanceTransaction(
    VkInstance createdInstance, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance,
    VkInstance failureOutputValue, PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr,
    LayerOptionsPtr &&layerOptions, const InstancePublicationOperations &operations) noexcept
{
    try
    {
        capture::fault::Checkpoint(
            capture::fault::Point::InstanceAfterDownstreamCreateBeforePublication);
        operations.publish(createdInstance, std::move(layerOptions));
        *pInstance = createdInstance;
        return VK_SUCCESS;
    }
    catch (...)
    {
        const auto translation = capture::abi::TranslateException(
            std::current_exception(), capture::abi::BoundaryPhase::ObjectPublication);
        operations.destroyTracked(createdInstance, pAllocator, nextGetInstanceProcAddr);
        *pInstance = failureOutputValue;
        return translation.result;
    }
}

void DestroyUntrackedCreatedInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator,
                                     PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr) noexcept
{
    try
    {
        if (instance == VK_NULL_HANDLE || nextGetInstanceProcAddr == nullptr)
        {
            return;
        }
        const auto rawDestroy = nextGetInstanceProcAddr(instance, "vkDestroyInstance");
        const auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(rawDestroy);
        if (destroy != nullptr)
        {
            destroy(instance, pAllocator);
        }
    }
    catch (...)
    {
    }
}
