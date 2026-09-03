/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace capture_policy
{
enum class DeferredPipelineDecision
{
    Capture,
    Reject,
};

enum class CommandBufferDecision
{
    Capture,
    RejectProtected,
    RejectForeignProcessingEngine,
    RejectMissingTransferCapability,
};

enum class Submit2Route
{
    Core,
    Khr,
};

enum class Synchronization2Route
{
    Unavailable,
    Core,
    Khr,
};

using QueueSubmit2Function = VkResult(VKAPI_PTR *)(VkQueue, uint32_t, const VkSubmitInfo2 *,
                                                   VkFence);
using CmdPipelineBarrier2Function = void(VKAPI_PTR *)(VkCommandBuffer, const VkDependencyInfo *);

DeferredPipelineDecision ClassifyDeferredPipelineOperation(
    VkDeferredOperationKHR deferredOperation) noexcept;
CommandBufferDecision ClassifyCommandBuffer(bool isProtected, bool hasForeignProcessingEngine,
                                            VkQueueFlags queueFlags) noexcept;
bool IsLegacySubmitProtected(const VkSubmitInfo &submit) noexcept;
bool IsSubmit2Protected(const VkSubmitInfo2 &submit) noexcept;
bool RejectProtectedSelectedCapture(bool hasSelectedCapture, bool isProtectedSubmit) noexcept;
QueueSubmit2Function SelectQueueSubmit2Function(Submit2Route route, PFN_vkQueueSubmit2 coreFunction,
                                                PFN_vkQueueSubmit2KHR khrFunction) noexcept;
Synchronization2Route SelectSynchronization2Route(
    PFN_vkCmdPipelineBarrier2 coreFunction, PFN_vkCmdPipelineBarrier2KHR khrFunction) noexcept;
CmdPipelineBarrier2Function SelectCmdPipelineBarrier2Function(
    Synchronization2Route route, PFN_vkCmdPipelineBarrier2 coreFunction,
    PFN_vkCmdPipelineBarrier2KHR khrFunction) noexcept;
} // namespace capture_policy
