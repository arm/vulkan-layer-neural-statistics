/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vulkan/utility/vk_struct_helper.hpp>
#include <vulkan/vulkan.h>

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyDevice<user_tag>(VkDevice device,
                                                           const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateShaderModule<user_tag>(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyShaderModule<user_tag>(
    VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDataGraphPipelinesARM<user_tag>(
    VkDevice device, VkDeferredOperationKHR deferredOperation, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkDataGraphPipelineCreateInfoARM *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyPipeline<user_tag>(
    VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDataGraphPipelineSessionARM<user_tag>(
    VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDataGraphPipelineSessionARM *pSession);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(
    VkDevice device, VkDataGraphPipelineSessionARM session,
    const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR VkResult VKAPI_CALL
layer_vkGetDataGraphPipelineSessionBindPointRequirementsARM<user_tag>(
    VkDevice device, const VkDataGraphPipelineSessionBindPointRequirementsInfoARM *pInfo,
    uint32_t *pBindPointRequirementCount,
    VkDataGraphPipelineSessionBindPointRequirementARM *pBindPointRequirements);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkAllocateCommandBuffers<user_tag>(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkBeginCommandBuffer<user_tag>(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkResetCommandBuffer<user_tag>(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkFreeCommandBuffers<user_tag>(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateCommandPool<user_tag>(
    VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkResetCommandPool<user_tag>(VkDevice device,
                                                                  VkCommandPool commandPool,
                                                                  VkCommandPoolResetFlags flags);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyCommandPool<user_tag>(
    VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator);

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkCmdDispatchDataGraphARM<user_tag>(
    VkCommandBuffer commandBuffer, VkDataGraphPipelineSessionARM session,
    const VkDataGraphPipelineDispatchInfoARM *pInfo);

template <>
VKAPI_ATTR void VKAPI_CALL
layer_vkCmdExecuteCommands<user_tag>(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                     const VkCommandBuffer *pCommandBuffers);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit<user_tag>(VkQueue queue, uint32_t submitCount,
                                                             const VkSubmitInfo *pSubmits,
                                                             VkFence fence);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit2<user_tag>(VkQueue queue, uint32_t submitCount,
                                                              const VkSubmitInfo2 *pSubmits,
                                                              VkFence fence);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit2KHR<user_tag>(VkQueue queue,
                                                                 uint32_t submitCount,
                                                                 const VkSubmitInfo2 *pSubmits,
                                                                 VkFence fence);

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkSetDebugUtilsObjectNameEXT<user_tag>(
    VkDevice device, const VkDebugUtilsObjectNameInfoEXT *pNameInfo);
