/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_policy.hpp"

namespace capture_policy
{
DeferredPipelineDecision ClassifyDeferredPipelineOperation(
    VkDeferredOperationKHR deferredOperation) noexcept
{
    return deferredOperation == VK_NULL_HANDLE ? DeferredPipelineDecision::Capture
                                               : DeferredPipelineDecision::Reject;
}

CommandBufferDecision ClassifyCommandBuffer(bool isProtected, bool hasForeignProcessingEngine,
                                            VkQueueFlags queueFlags) noexcept
{
    if (isProtected)
    {
        return CommandBufferDecision::RejectProtected;
    }
    if (hasForeignProcessingEngine)
    {
        return CommandBufferDecision::RejectForeignProcessingEngine;
    }

    constexpr VkQueueFlags transferCapable =
        VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if ((queueFlags & transferCapable) == 0)
    {
        return CommandBufferDecision::RejectMissingTransferCapability;
    }
    return CommandBufferDecision::Capture;
}

bool IsLegacySubmitProtected(const VkSubmitInfo &submit) noexcept
{
    auto *next = reinterpret_cast<const VkBaseInStructure *>(submit.pNext);
    while (next != nullptr)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO)
        {
            const auto *protectedInfo = reinterpret_cast<const VkProtectedSubmitInfo *>(next);
            return protectedInfo->protectedSubmit == VK_TRUE;
        }
        next = next->pNext;
    }
    return false;
}

bool IsSubmit2Protected(const VkSubmitInfo2 &submit) noexcept
{
    return (submit.flags & VK_SUBMIT_PROTECTED_BIT) != 0;
}

bool RejectProtectedSelectedCapture(bool hasSelectedCapture, bool isProtectedSubmit) noexcept
{
    return hasSelectedCapture && isProtectedSubmit;
}

QueueSubmit2Function SelectQueueSubmit2Function(Submit2Route route, PFN_vkQueueSubmit2 coreFunction,
                                                PFN_vkQueueSubmit2KHR khrFunction) noexcept
{
    return route == Submit2Route::Core ? coreFunction : khrFunction;
}

Synchronization2Route SelectSynchronization2Route(PFN_vkCmdPipelineBarrier2 coreFunction,
                                                  PFN_vkCmdPipelineBarrier2KHR khrFunction) noexcept
{
    if (coreFunction != nullptr)
    {
        return Synchronization2Route::Core;
    }
    if (khrFunction != nullptr)
    {
        return Synchronization2Route::Khr;
    }
    return Synchronization2Route::Unavailable;
}

CmdPipelineBarrier2Function SelectCmdPipelineBarrier2Function(
    Synchronization2Route route, PFN_vkCmdPipelineBarrier2 coreFunction,
    PFN_vkCmdPipelineBarrier2KHR khrFunction) noexcept
{
    switch (route)
    {
    case Synchronization2Route::Core:
        return coreFunction;
    case Synchronization2Route::Khr:
        return khrFunction;
    case Synchronization2Route::Unavailable:
        return nullptr;
    }
    return nullptr;
}
} // namespace capture_policy
