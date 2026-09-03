/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "layer_options.hpp"

#include <memory>

#include <vulkan/vulkan.h>

using LayerOptionsPtr = std::unique_ptr<const LayerOptions>;

struct InstancePublicationOperations
{
    void (*publish)(VkInstance, LayerOptionsPtr &&);
    void (*destroyTracked)(VkInstance, const VkAllocationCallbacks *,
                           PFN_vkGetInstanceProcAddr) noexcept;
};

// Publishes pre-created layer-owned state after the framework has successfully
// created and tracked a downstream VkInstance. Ownership transfer itself is
// non-throwing; the fault checkpoint keeps the rollback boundary injectable.
VKAPI_ATTR VkResult VKAPI_CALL PublishCreatedInstanceTransaction(
    VkInstance createdInstance, const VkAllocationCallbacks *pAllocator, VkInstance *pInstance,
    VkInstance failureOutputValue, PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr,
    LayerOptionsPtr &&layerOptions, const InstancePublicationOperations &operations) noexcept;

// Used only when the framework's create path throws after the downstream layer
// has written a handle but before framework tracking completed.
void DestroyUntrackedCreatedInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator,
                                     PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr) noexcept;
