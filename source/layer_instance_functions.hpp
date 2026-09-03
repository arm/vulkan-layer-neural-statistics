/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vulkan/utility/vk_struct_helper.hpp>
#include <vulkan/vulkan.h>

template <>
VKAPI_ATTR VkResult VKAPI_CALL
layer_vkCreateInstance<user_tag>(const VkInstanceCreateInfo *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator, VkInstance *pInstance);

template <>
VKAPI_ATTR void VKAPI_CALL
layer_vkDestroyInstance<user_tag>(VkInstance instance, const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDevice<user_tag>(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice);
