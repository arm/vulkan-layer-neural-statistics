/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "resource_tracking.hpp"

#include "fault_injection.hpp"

ShaderModuleRecord::ShaderModuleRecord(VkShaderModule shaderModule,
                                       capture::ShaderModuleId shaderId,
                                       std::vector<uint8_t> ownedSpirv)
    : handle(shaderModule), id(shaderId), spirv(std::move(ownedSpirv))
{
}

PipelineRecord::PipelineRecord(VkPipeline pipeline, capture::PipelineId pipelineId,
                               capture::PipelineMetadata pipelineMetadata,
                               std::shared_ptr<ShaderModuleRecord> shaderModule)
    : handle(pipeline), id(pipelineId), metadata(std::move(pipelineMetadata)),
      shader(std::move(shaderModule))
{
}

SessionRecord::SessionRecord(VkDataGraphPipelineSessionARM session,
                             std::shared_ptr<PipelineRecord> pipelineRecord)
    : handle(session), pipeline(std::move(pipelineRecord))
{
}

StatsSnapshot::StatsSnapshot(VkDevice device, PFN_vkDestroyBuffer destroyBuffer,
                             PFN_vkFreeMemory freeMemory) noexcept
    : ownerDevice(device), destroyBufferFunction(destroyBuffer), freeMemoryFunction(freeMemory)
{
}

StatsSnapshot::~StatsSnapshot() noexcept
{
    if (buffer != VK_NULL_HANDLE)
    {
        destroyBufferFunction(ownerDevice, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE)
    {
        freeMemoryFunction(ownerDevice, memory, nullptr);
    }
}

SubmissionArchive::SubmissionArchive(VkDevice device,
                                     PFN_vkDestroyCommandPool destroyCommandPool) noexcept
    : ownerDevice(device), destroyCommandPoolFunction(destroyCommandPool)
{
}

SubmissionArchive::~SubmissionArchive() noexcept
{
    if (commandPool != VK_NULL_HANDLE)
    {
        destroyCommandPoolFunction(ownerDevice, commandPool, nullptr);
    }
}

CommandBufferRecord::CommandBufferRecord(VkCommandBuffer commandBuffer, VkCommandPool commandPool,
                                         uint32_t familyIndex, bool protectedCommandBuffer)
    : handle(commandBuffer), parent(commandPool), queueFamilyIndex(familyIndex),
      isProtected(protectedCommandBuffer)
{
}

void CommandBufferRecord::reset()
{
    dispatches.clear();
    snapshotError.clear();
    usageFlags = 0;
}

CommandPoolRecord::CommandPoolRecord(VkCommandPool commandPool, uint32_t familyIndex,
                                     VkCommandPoolCreateFlags flags)
    : handle(commandPool), queueFamilyIndex(familyIndex),
      isProtected((flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT) != 0)
{
}

void CommandPoolRecord::reset()
{
    for (auto it = cbufferMap.begin(); it != cbufferMap.end();)
    {
        if (auto bufferRecord = it->second.lock())
        {
            bufferRecord->reset();
            ++it;
        }
        else
        {
            it = cbufferMap.erase(it);
        }
    }
}

// --- DeviceStatsManager ----------------------------------------------------

std::shared_ptr<ShaderModuleRecord> DeviceStatsManager::addShaderModuleRecord(
    VkShaderModule shaderModule, capture::ShaderModuleId shaderId, std::vector<uint8_t> spirv)
{
    auto record = std::make_shared<ShaderModuleRecord>(shaderModule, shaderId, std::move(spirv));
    capture::fault::Checkpoint(capture::fault::Point::ShaderMapInsertion);
    const auto [_, inserted] = shaderModules.emplace(as_key(shaderModule), record);
    return inserted ? record : nullptr;
}

std::shared_ptr<ShaderModuleRecord> DeviceStatsManager::getShaderModuleRecord(
    VkShaderModule shaderModule)
{
    const auto it = shaderModules.find(as_key(shaderModule));
    return it == shaderModules.end() ? nullptr : it->second;
}

void DeviceStatsManager::removeShaderModuleRecord(VkShaderModule shaderModule)
{
    shaderModules.erase(as_key(shaderModule));
}

std::shared_ptr<PipelineRecord> DeviceStatsManager::addPipelineRecord(
    VkPipeline pipeline, capture::PipelineId pipelineId, capture::PipelineMetadata metadata,
    std::shared_ptr<ShaderModuleRecord> shaderModule)
{
    auto record = std::make_shared<PipelineRecord>(pipeline, pipelineId, std::move(metadata),
                                                   std::move(shaderModule));
    capture::fault::Checkpoint(capture::fault::Point::PipelineMapInsertion);
    const auto [_, inserted] = pipelines.emplace(as_key(pipeline), record);
    return inserted ? record : nullptr;
}

std::shared_ptr<PipelineRecord> DeviceStatsManager::getPipelineRecord(VkPipeline pipeline)
{
    auto it = pipelines.find(as_key(pipeline));
    if (it != pipelines.end())
    {
        return it->second;
    }
    return nullptr;
}

void DeviceStatsManager::removePipelineRecord(VkPipeline pipeline)
{
    pipelines.erase(as_key(pipeline));
}

std::vector<std::shared_ptr<PipelineRecord>> DeviceStatsManager::pipelinesUsingShader(
    const std::shared_ptr<ShaderModuleRecord> &shaderModule)
{
    std::vector<std::shared_ptr<PipelineRecord>> result;
    for (const auto &[_, pipeline] : pipelines)
    {
        if (pipeline->shader == shaderModule)
        {
            result.emplace_back(pipeline);
        }
    }
    return result;
}

std::shared_ptr<SessionRecord> DeviceStatsManager::addSessionRecord(
    VkDataGraphPipelineSessionARM session, capture::SessionId sessionId,
    std::shared_ptr<PipelineRecord> pipeline,
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> &&filteredBindPointRequirements,
    const StatsMemory &statsMemory)
{
    auto record = std::make_shared<SessionRecord>(session, std::move(pipeline));
    record->id = sessionId;
    record->filteredBindPointRequirements = std::move(filteredBindPointRequirements);
    record->statsMemory = statsMemory;
    record->statisticsCaptureActive = statsMemory.handle != VK_NULL_HANDLE;

    capture::fault::Checkpoint(capture::fault::Point::SessionMapInsertion);
    const auto [_, inserted] = sessions.emplace(as_key(session), record);
    return inserted ? record : nullptr;
}

std::shared_ptr<SessionRecord> DeviceStatsManager::getSessionRecord(
    VkDataGraphPipelineSessionARM session)
{
    auto it = sessions.find(as_key(session));
    if (it != sessions.end())
    {
        return it->second;
    }
    return nullptr;
}

void DeviceStatsManager::removeSessionRecord(VkDataGraphPipelineSessionARM session)
{
    sessions.erase(as_key(session));
}

std::shared_ptr<CommandPoolRecord> DeviceStatsManager::addCommandPoolRecord(
    VkCommandPool commandPool, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags)
{
    auto record = std::make_shared<CommandPoolRecord>(commandPool, queueFamilyIndex, flags);
    const auto [_, inserted] = commandPools.emplace(as_key(commandPool), record);
    return inserted ? record : nullptr;
}

std::shared_ptr<CommandPoolRecord> DeviceStatsManager::getCommandPoolRecord(
    VkCommandPool commandPool)
{
    auto it = commandPools.find(as_key(commandPool));
    if (it != commandPools.end())
    {
        return it->second;
    }
    return nullptr;
}

void DeviceStatsManager::removeCommandPoolRecord(VkCommandPool commandPool)
{
    // Remove child command buffers from global map
    const uint64_t poolKey = as_key(commandPool);
    auto it = commandPools.find(poolKey);
    if (it == commandPools.end())
    {
        return;
    }

    auto poolRecord = it->second;
    for (const auto &[cbKey, _] : poolRecord->cbufferMap)
    {
        commandBuffers.erase(cbKey);
    }
    poolRecord->cbufferMap.clear();

    // Remove pool from global map
    commandPools.erase(it);
}

std::shared_ptr<CommandBufferRecord> DeviceStatsManager::addCommandBufferRecord(
    VkCommandBuffer commandBuffer, VkCommandPool commandPool)
{
    const auto poolIt = commandPools.find(as_key(commandPool));
    if (poolIt == commandPools.end())
    {
        return nullptr;
    }

    auto record = std::make_shared<CommandBufferRecord>(
        commandBuffer, commandPool, poolIt->second->queueFamilyIndex, poolIt->second->isProtected);

    const uint64_t commandBufferKey = as_key(commandBuffer);
    const auto [poolRecordIt, poolInserted] = poolIt->second->cbufferMap.emplace(
        commandBufferKey, std::weak_ptr<CommandBufferRecord>(record));
    if (!poolInserted)
    {
        return nullptr;
    }

    try
    {
        capture::fault::Checkpoint(capture::fault::Point::CommandBufferMapInsertion);
        const auto [_, inserted] = commandBuffers.emplace(commandBufferKey, record);
        if (!inserted)
        {
            poolIt->second->cbufferMap.erase(poolRecordIt);
            return nullptr;
        }
    }
    catch (...)
    {
        poolIt->second->cbufferMap.erase(poolRecordIt);
        throw;
    }

    return record;
}

std::shared_ptr<CommandBufferRecord> DeviceStatsManager::getCommandBufferRecord(
    VkCommandBuffer commandBuffer)
{
    auto it = commandBuffers.find(as_key(commandBuffer));
    return (it != commandBuffers.end()) ? it->second : nullptr;
}

void DeviceStatsManager::removeCommandBufferRecord(VkCommandBuffer commandBuffer)
{
    const uint64_t cbKey = as_key(commandBuffer);
    auto it = commandBuffers.find(cbKey);
    if (it == commandBuffers.end())
    {
        return;
    }

    // Remove from the parent pool's buffer map
    const VkCommandPool poolHandle = it->second->parent;
    auto pit = commandPools.find(as_key(poolHandle));
    if (pit != commandPools.end())
    {
        pit->second->cbufferMap.erase(cbKey);
    }

    // Remove from global list
    commandBuffers.erase(it);
}

void DeviceStatsManager::clearCommandBufferRecords() noexcept
{
    commandBuffers.clear();
    commandPools.clear();
}
