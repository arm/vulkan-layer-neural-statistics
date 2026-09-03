/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_model.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

// Helper to convert Vulkan handles to a stable integer key for unordered_map.
template <typename VkHandleT> inline static uint64_t as_key(VkHandleT h)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
}

struct StatsMemory
{
    VkDeviceMemory handle{VK_NULL_HANDLE};
    VkBuffer sourceBuffer{VK_NULL_HANDLE};
    VkDeviceSize dataSize{0};
    VkDeviceSize allocationSize{0};
    VkMemoryPropertyFlags memoryProperties{0};
};

struct ResourceRecord
{
    std::string debugName;

    void setDebugName(const std::string &name)
    {
        debugName = name;
    }
};

struct ShaderModuleRecord : public ResourceRecord
{
    ShaderModuleRecord(VkShaderModule shaderModule, capture::ShaderModuleId shaderId,
                       std::vector<uint8_t> ownedSpirv);

    VkShaderModule handle{VK_NULL_HANDLE};
    capture::ShaderModuleId id;
    std::vector<uint8_t> spirv;
};

struct PipelineRecord : public ResourceRecord
{
    PipelineRecord(VkPipeline pipeline, capture::PipelineId pipelineId,
                   capture::PipelineMetadata pipelineMetadata,
                   std::shared_ptr<ShaderModuleRecord> shaderModule);

    VkPipeline handle{VK_NULL_HANDLE};
    capture::PipelineId id;
    capture::PipelineMetadata metadata;
    std::shared_ptr<ShaderModuleRecord> shader;
    bool capturable{true};
    std::string unsupportedReason;
};

struct SessionRecord : public ResourceRecord
{
    SessionRecord(VkDataGraphPipelineSessionARM session,
                  std::shared_ptr<PipelineRecord> pipelineRecord);

    VkDataGraphPipelineSessionARM handle{VK_NULL_HANDLE};
    capture::SessionId id;
    uint64_t flags{0};
    std::shared_ptr<PipelineRecord> pipeline;
    StatsMemory statsMemory;
    bool statisticsCaptureActive{false};
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> filteredBindPointRequirements;
    uint64_t nextExecutionIndex{0};
    bool executionIndexExhausted{false};
    bool capturable{true};
    std::string unsupportedReason;
    std::string captureError;
};

struct StatsSnapshot
{
    StatsSnapshot(VkDevice ownerDevice, PFN_vkDestroyBuffer destroyBufferFunction,
                  PFN_vkFreeMemory freeMemoryFunction) noexcept;
    ~StatsSnapshot() noexcept;

    StatsSnapshot(const StatsSnapshot &) = delete;
    StatsSnapshot &operator=(const StatsSnapshot &) = delete;

    std::shared_ptr<SessionRecord> session;
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize dataSize{0};
    VkDeviceSize allocationSize{0};
    VkMemoryPropertyFlags memoryProperties{0};
    uint64_t executionIndex{0};
    std::optional<capture::DispatchId> dispatchId;
    uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
    uint64_t inFlightUses{0};
    bool inFlight{false};

  private:
    // Snapshot references can outlive command-buffer tracking while a queue-submit
    // collector is pending. The copied downstream dispatch functions let the final
    // shared owner release the Vulkan allocation without consulting layer maps.
    // Vulkan object-lifetime rules must keep ownerDevice alive until all intercepted
    // calls using it have returned. The layer's vkDestroyDevice specialization clears
    // command-buffer ownership before destroying the downstream VkDevice.
    VkDevice ownerDevice{VK_NULL_HANDLE};
    PFN_vkDestroyBuffer destroyBufferFunction{nullptr};
    PFN_vkFreeMemory freeMemoryFunction{nullptr};
};

struct SubmissionArchive
{
    SubmissionArchive(VkDevice ownerDevice,
                      PFN_vkDestroyCommandPool destroyCommandPoolFunction) noexcept;
    ~SubmissionArchive() noexcept;

    SubmissionArchive(const SubmissionArchive &) = delete;
    SubmissionArchive &operator=(const SubmissionArchive &) = delete;

    VkCommandPool commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> commandBuffers;

  private:
    VkDevice ownerDevice{VK_NULL_HANDLE};
    PFN_vkDestroyCommandPool destroyCommandPoolFunction{nullptr};
};

struct CommandBufferRecord : public ResourceRecord
{
    CommandBufferRecord(VkCommandBuffer commandBuffer, VkCommandPool commandPool,
                        uint32_t queueFamilyIndex, bool isProtected);

    VkCommandBuffer handle{VK_NULL_HANDLE};
    VkCommandPool parent{VK_NULL_HANDLE};
    uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
    bool isProtected{false};
    VkCommandBufferUsageFlags usageFlags{0};

    void reset();

    struct DispatchOccurrence
    {
        std::shared_ptr<SessionRecord> session;
        std::shared_ptr<StatsSnapshot> snapshot;
        std::string unsupportedReason;
    };

    std::vector<DispatchOccurrence> dispatches;
    std::string snapshotError;
};

struct CommandPoolRecord : public ResourceRecord
{
    CommandPoolRecord(VkCommandPool commandPool, uint32_t queueFamilyIndex,
                      VkCommandPoolCreateFlags flags);

    void reset();

    VkCommandPool handle{VK_NULL_HANDLE};
    uint32_t queueFamilyIndex{VK_QUEUE_FAMILY_IGNORED};
    bool isProtected{false};
    bool hasForeignProcessingEngine{false};
    std::unordered_map<uint64_t, std::weak_ptr<CommandBufferRecord>> cbufferMap;
};

class DeviceStatsManager
{
  public:
    DeviceStatsManager() = default;
    ~DeviceStatsManager() = default;

    // --- Shader module handling --------------------------------------------

    std::shared_ptr<ShaderModuleRecord> addShaderModuleRecord(VkShaderModule shaderModule,
                                                              capture::ShaderModuleId shaderId,
                                                              std::vector<uint8_t> spirv);
    std::shared_ptr<ShaderModuleRecord> getShaderModuleRecord(VkShaderModule shaderModule);
    void removeShaderModuleRecord(VkShaderModule shaderModule);

    // --- Pipeline handling -------------------------------------------------

    std::shared_ptr<PipelineRecord> addPipelineRecord(
        VkPipeline pipeline, capture::PipelineId pipelineId, capture::PipelineMetadata metadata,
        std::shared_ptr<ShaderModuleRecord> shaderModule);
    std::shared_ptr<PipelineRecord> getPipelineRecord(VkPipeline pipeline);
    void removePipelineRecord(VkPipeline pipeline);
    std::vector<std::shared_ptr<PipelineRecord>> pipelinesUsingShader(
        const std::shared_ptr<ShaderModuleRecord> &shaderModule);

    // --- Session handling --------------------------------------------------

    std::shared_ptr<SessionRecord> addSessionRecord(
        VkDataGraphPipelineSessionARM session, capture::SessionId sessionId,
        std::shared_ptr<PipelineRecord> pipeline,
        std::vector<VkDataGraphPipelineSessionBindPointRequirementARM>
            &&filteredBindPointRequirements,
        const StatsMemory &statsMemory);
    std::shared_ptr<SessionRecord> getSessionRecord(VkDataGraphPipelineSessionARM session);
    void removeSessionRecord(VkDataGraphPipelineSessionARM session);

    // --- Command pool handling --------------------------------------------

    std::shared_ptr<CommandPoolRecord> addCommandPoolRecord(VkCommandPool commandPool,
                                                            uint32_t queueFamilyIndex,
                                                            VkCommandPoolCreateFlags flags);
    std::shared_ptr<CommandPoolRecord> getCommandPoolRecord(VkCommandPool commandPool);
    void removeCommandPoolRecord(VkCommandPool commandPool);

    // --- Command buffer handling ------------------------------------------

    std::shared_ptr<CommandBufferRecord> addCommandBufferRecord(VkCommandBuffer commandBuffer,
                                                                VkCommandPool commandPool);
    std::shared_ptr<CommandBufferRecord> getCommandBufferRecord(VkCommandBuffer commandBuffer);
    void removeCommandBufferRecord(VkCommandBuffer commandBuffer);

    // Must run before the downstream VkDevice is destroyed so final snapshot
    // owners can release their layer-created Vulkan allocations safely.
    void clearCommandBufferRecords() noexcept;

  private:
    std::unordered_map<uint64_t, std::shared_ptr<ShaderModuleRecord>> shaderModules;
    std::unordered_map<uint64_t, std::shared_ptr<PipelineRecord>> pipelines;
    std::unordered_map<uint64_t, std::shared_ptr<SessionRecord>> sessions;
    std::unordered_map<uint64_t, std::shared_ptr<CommandBufferRecord>> commandBuffers;
    std::unordered_map<uint64_t, std::shared_ptr<CommandPoolRecord>> commandPools;
};
