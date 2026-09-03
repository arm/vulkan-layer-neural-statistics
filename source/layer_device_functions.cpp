/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "device.hpp"
#include "fault_injection.hpp"
#include "framework/device_dispatch_table.hpp"
#include "submission_planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>

extern std::mutex g_vulkanLock;

namespace
{
class ShaderCreationRollback
{
  public:
    void arm(Device *owner, VkDevice deviceHandle, const VkAllocationCallbacks *allocator,
             VkShaderModule *shaderModule) noexcept
    {
        layer = owner;
        device = deviceHandle;
        pAllocator = allocator;
        pShaderModule = shaderModule;
        active = true;
    }

    void release() noexcept
    {
        active = false;
    }

    ~ShaderCreationRollback() noexcept
    {
        if (active && pShaderModule != nullptr && *pShaderModule != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyShaderModule(device, *pShaderModule, pAllocator);
            *pShaderModule = VK_NULL_HANDLE;
        }
    }

  private:
    Device *layer{nullptr};
    VkDevice device{VK_NULL_HANDLE};
    const VkAllocationCallbacks *pAllocator{nullptr};
    VkShaderModule *pShaderModule{nullptr};
    bool active{false};
};

class PipelineCreationRollback
{
  public:
    void arm(Device *owner, VkDevice deviceHandle, const VkAllocationCallbacks *allocator,
             uint32_t pipelineCount, VkPipeline *pipelines) noexcept
    {
        layer = owner;
        device = deviceHandle;
        pAllocator = allocator;
        count = pipelineCount;
        pPipelines = pipelines;
        active = true;
    }

    void release() noexcept
    {
        active = false;
    }

    ~PipelineCreationRollback() noexcept
    {
        if (!active)
        {
            return;
        }
        for (uint32_t i = count; i > 0; --i)
        {
            VkPipeline &pipeline = pPipelines[i - 1];
            if (pipeline != VK_NULL_HANDLE)
            {
                layer->driver.vkDestroyPipeline(device, pipeline, pAllocator);
                pipeline = VK_NULL_HANDLE;
            }
        }
    }

  private:
    Device *layer{nullptr};
    VkDevice device{VK_NULL_HANDLE};
    const VkAllocationCallbacks *pAllocator{nullptr};
    uint32_t count{0};
    VkPipeline *pPipelines{nullptr};
    bool active{false};
};

class SessionCreationRollback
{
  public:
    void arm(Device *owner, VkDevice deviceHandle, const VkAllocationCallbacks *allocator,
             VkDataGraphPipelineSessionARM *session) noexcept
    {
        layer = owner;
        device = deviceHandle;
        pAllocator = allocator;
        pSession = session;
        active = true;
    }

    void adoptBuffer(VkBuffer value) noexcept
    {
        buffer = value;
    }
    void adoptMemory(VkDeviceMemory value) noexcept
    {
        memory = value;
    }
    void release() noexcept
    {
        active = false;
    }

    void rollbackNow() noexcept
    {
        if (!active)
        {
            return;
        }
        if (buffer != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (pSession != nullptr && *pSession != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyDataGraphPipelineSessionARM(device, *pSession, pAllocator);
            *pSession = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE)
        {
            layer->driver.vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
        active = false;
    }

    ~SessionCreationRollback() noexcept
    {
        rollbackNow();
    }

  private:
    Device *layer{nullptr};
    VkDevice device{VK_NULL_HANDLE};
    const VkAllocationCallbacks *pAllocator{nullptr};
    VkDataGraphPipelineSessionARM *pSession{nullptr};
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    bool active{false};
};

class CommandBuffersRollback
{
  public:
    void arm(Device *owner, VkDevice deviceHandle, VkCommandPool commandPoolHandle,
             uint32_t commandBufferCount, VkCommandBuffer *commandBuffers) noexcept
    {
        layer = owner;
        device = deviceHandle;
        commandPool = commandPoolHandle;
        count = commandBufferCount;
        pCommandBuffers = commandBuffers;
        active = true;
    }

    void release() noexcept
    {
        active = false;
    }

    ~CommandBuffersRollback() noexcept
    {
        if (!active)
        {
            return;
        }
        layer->driver.vkFreeCommandBuffers(device, commandPool, count, pCommandBuffers);
        for (uint32_t i = 0; i < count; ++i)
        {
            pCommandBuffers[i] = VK_NULL_HANDLE;
        }
    }

  private:
    Device *layer{nullptr};
    VkDevice device{VK_NULL_HANDLE};
    VkCommandPool commandPool{VK_NULL_HANDLE};
    uint32_t count{0};
    VkCommandBuffer *pCommandBuffers{nullptr};
    bool active{false};
};

class CommandPoolRollback
{
  public:
    void arm(Device *owner, VkDevice deviceHandle, const VkAllocationCallbacks *allocator,
             VkCommandPool *commandPool) noexcept
    {
        layer = owner;
        device = deviceHandle;
        pAllocator = allocator;
        pCommandPool = commandPool;
        active = true;
    }

    void release() noexcept
    {
        active = false;
    }

    ~CommandPoolRollback() noexcept
    {
        if (active && pCommandPool != nullptr && *pCommandPool != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyCommandPool(device, *pCommandPool, pAllocator);
            *pCommandPool = VK_NULL_HANDLE;
        }
    }

  private:
    Device *layer{nullptr};
    VkDevice device{VK_NULL_HANDLE};
    const VkAllocationCallbacks *pAllocator{nullptr};
    VkCommandPool *pCommandPool{nullptr};
    bool active{false};
};

bool FindMemoryTypeIndex(Device *layer, uint32_t typeBits, VkMemoryPropertyFlags required,
                         VkMemoryPropertyFlags preferred, uint32_t &selectedIndex,
                         VkMemoryPropertyFlags *selectedProperties = nullptr)
{
    VkPhysicalDeviceMemoryProperties memProps;
    layer->instance->driver.vkGetPhysicalDeviceMemoryProperties(layer->physicalDevice, &memProps);

    const auto find = [&](VkMemoryPropertyFlags wanted) -> bool
    {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            const auto flags = memProps.memoryTypes[i].propertyFlags;
            if ((typeBits & (1u << i)) && (flags & required) == required &&
                (flags & wanted) == wanted)
            {
                selectedIndex = i;
                if (selectedProperties != nullptr)
                {
                    *selectedProperties = flags;
                }
                return true;
            }
        }
        return false;
    };

    return find(preferred) || find(0);
}

bool GetPipelinePropertyData(Device *layer, VkDevice device, VkPipeline pipeline,
                             VkDataGraphPipelinePropertyARM property, std::vector<uint8_t> &data,
                             bool *isText = nullptr)
{
    data.clear();
    if (isText != nullptr)
    {
        *isText = false;
    }

    // Query the property directly rather than treating the advertised availability list as
    // authoritative. Some drivers implement these properties but omit them from
    // vkGetDataGraphPipelineAvailablePropertiesARM. Unsupported direct queries fail cleanly below,
    // while affected drivers can still provide the debug database and statistics metadata.
    const VkDataGraphPipelineInfoARM pipelineInfo{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_INFO_ARM,
        nullptr,
        pipeline,
    };
    VkDataGraphPipelinePropertyQueryResultARM propertyQuery{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT_ARM,
        nullptr,
        property,
        VK_FALSE,
        0,
        nullptr,
    };

    VkResult result = VK_SUCCESS;
    do
    {
        propertyQuery.dataSize = 0;
        propertyQuery.pData = nullptr;

        result = layer->driver.vkGetDataGraphPipelinePropertiesARM(device, &pipelineInfo, 1,
                                                                   &propertyQuery);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        {
            data.clear();
            return false;
        }

        data.resize(propertyQuery.dataSize);
        propertyQuery.pData = data.empty() ? nullptr : data.data();

        result = layer->driver.vkGetDataGraphPipelinePropertiesARM(device, &pipelineInfo, 1,
                                                                   &propertyQuery);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        {
            data.clear();
            return false;
        }

        data.resize(propertyQuery.dataSize);
    } while (result == VK_INCOMPLETE);

    const bool propertyIsText = propertyQuery.isText == VK_TRUE;
    if (isText != nullptr)
    {
        *isText = propertyIsText;
    }

    if (propertyIsText && !data.empty() && data.back() != '\0')
    {
        data.push_back('\0');
    }

    return true;
}

template <typename Handle> uint64_t DiagnosticHandleValue(Handle handle) noexcept
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

void DisableCapture(Instance *instance, const char *message) noexcept
{
    if (instance != nullptr)
    {
        (void)instance->disableCapture(message);
    }
}

template <typename Submitter>
CaptureJobStatus SubmitWriterJob(Instance *instance, Submitter &&submitter,
                                 const char *failureMessage) noexcept
{
    if (instance == nullptr)
    {
        return CaptureJobStatus::Skipped;
    }
    return instance->submitCaptureJob(std::forward<Submitter>(submitter), failureMessage);
}

VkResult CreateOriginalSessionPlaceholder(
    Device *layer, VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDataGraphPipelineSessionARM *pSession,
    const std::shared_ptr<PipelineRecord> &pipelineRecord, const char *unsupportedReason) noexcept
{
    VkResult result = VK_ERROR_UNKNOWN;
    try
    {
        result = layer->driver.vkCreateDataGraphPipelineSessionARM(device, pCreateInfo, pAllocator,
                                                                   pSession);
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
    if (result != VK_SUCCESS)
    {
        return result;
    }

    // Capture-only metadata failure must not make an otherwise legal
    // application session fail. Publication is best effort: on any failure the
    // application keeps the original downstream session, while capture is
    // disabled and no session ID is committed.
    try
    {
        std::lock_guard<std::mutex> lock{g_vulkanLock};
        auto *instance = const_cast<Instance *>(layer->instance);
        if (!instance->isCaptureEnabled())
        {
            return result;
        }
        const capture::SessionId id = instance->sessionIds.peek();
        auto record =
            layer->resourceManager.addSessionRecord(*pSession, id, pipelineRecord, {}, {});
        if (record == nullptr)
        {
            DisableCapture(instance, "capture-only original-session placeholder insertion failed; "
                                     "preserving downstream session");
            return result;
        }
        record->flags = pCreateInfo->flags;
        record->capturable = false;
        record->unsupportedReason = unsupportedReason;
        if (SubmitWriterJob(
                instance,
                [&](capture::CaptureWriter &writer)
                {
                    return writer.trackSession(id, pipelineRecord->id, pCreateInfo->flags,
                                               DiagnosticHandleValue(*pSession));
                },
                "session placeholder writer job was rejected after downstream creation") !=
                CaptureJobStatus::Accepted &&
            !instance->isCaptureEnabled())
        {
            layer->resourceManager.removeSessionRecord(*pSession);
            return result;
        }
        instance->sessionIds.commit();
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock{g_vulkanLock};
        layer->resourceManager.removeSessionRecord(*pSession);
        DisableCapture(
            const_cast<Instance *>(layer->instance),
            "capture-only original-session bookkeeping failed; preserving downstream session");
    }
    return result;
}

capture::PipelineMetadata CopyPipelineMetadata(Device *layer,
                                               const VkDataGraphPipelineCreateInfoARM &createInfo,
                                               std::shared_ptr<ShaderModuleRecord> &shaderRecord)
{
    capture::PipelineMetadata metadata;
    metadata.deviceId = layer->captureDeviceId;
    metadata.flags = createInfo.flags;
    metadata.layoutHandle = DiagnosticHandleValue(createInfo.layout);
    metadata.resourceBindings.reserve(createInfo.resourceInfoCount);
    for (uint32_t i = 0; i < createInfo.resourceInfoCount; ++i)
    {
        const auto &source = createInfo.pResourceInfos[i];
        metadata.resourceBindings.push_back(
            {source.descriptorSet, source.binding, source.arrayElement});
    }

    if (const auto *compiler =
            vku::FindStructInPNextChain<VkDataGraphPipelineCompilerControlCreateInfoARM>(
                createInfo.pNext);
        compiler != nullptr && compiler->pVendorOptions != nullptr)
    {
        metadata.vendorOptions = compiler->pVendorOptions;
    }

    const auto *shader =
        vku::FindStructInPNextChain<VkDataGraphPipelineShaderModuleCreateInfoARM>(createInfo.pNext);
    const auto *identifier =
        vku::FindStructInPNextChain<VkDataGraphPipelineIdentifierCreateInfoARM>(createInfo.pNext);
    metadata.identifierOnly = shader == nullptr && identifier != nullptr;
    if (const auto *engines =
            vku::FindStructInPNextChain<VkDataGraphProcessingEngineCreateInfoARM>(createInfo.pNext))
    {
        for (uint32_t i = 0; i < engines->processingEngineCount; ++i)
        {
            metadata.foreignProcessingEngine |= engines->pProcessingEngines[i].isForeign == VK_TRUE;
        }
    }
    if (shader == nullptr)
    {
        return metadata;
    }

    if (shader->pName != nullptr)
    {
        metadata.shader.entryPoint = shader->pName;
    }
    shaderRecord = layer->resourceManager.getShaderModuleRecord(shader->module);
    if (shaderRecord != nullptr)
    {
        metadata.shader.moduleId = shaderRecord->id;
        metadata.shader.spirvAvailable = !shaderRecord->spirv.empty();
        metadata.shader.friendlyName = shaderRecord->debugName;
    }
    if (shader->pSpecializationInfo != nullptr)
    {
        const auto &specialization = *shader->pSpecializationInfo;
        metadata.shader.specializationDataSize = specialization.dataSize;
        metadata.shader.specializationEntries.reserve(specialization.mapEntryCount);
        for (uint32_t i = 0; i < specialization.mapEntryCount; ++i)
        {
            const auto &entry = specialization.pMapEntries[i];
            metadata.shader.specializationEntries.push_back(
                {entry.constantID, entry.offset, entry.size});
        }
    }
    return metadata;
}

struct PreparedPipelineArtifact
{
    std::string fileName;
    capture::ArtifactType type{capture::ArtifactType::DebugDatabase};
    std::vector<uint8_t> binaryContents;
    std::string textContents;
    bool text{false};
};

struct PreparedPipelinePublication
{
    capture::PipelineId id;
    capture::PipelineMetadata metadata;
    std::optional<uint64_t> diagnosticHandle;
    std::vector<PreparedPipelineArtifact> artifacts;
};

PreparedPipelinePublication PreparePipelinePublication(
    const std::shared_ptr<PipelineRecord> &pipeline, std::vector<uint8_t> debugDatabase,
    std::vector<uint8_t> statisticsInfo, bool statisticsInfoIsText,
    capture::CaptureMetadata &metadata, bool &metadataChanged)
{
    PreparedPipelinePublication publication{
        pipeline->id,
        pipeline->metadata,
        DiagnosticHandleValue(pipeline->handle),
        {},
    };
    publication.artifacts.reserve(3);

    if (!debugDatabase.empty())
    {
        publication.artifacts.push_back({
            "debug_database.bin",
            capture::ArtifactType::DebugDatabase,
            std::move(debugDatabase),
            {},
            false,
        });
    }
    else
    {
        metadata.warnings.emplace_back("debug database was unavailable for pipeline " +
                                       std::to_string(pipeline->id.value()));
        metadataChanged = true;
    }

    if (!statisticsInfo.empty())
    {
        if (statisticsInfoIsText)
        {
            if (statisticsInfo.back() == '\0')
            {
                statisticsInfo.pop_back();
            }
            publication.artifacts.push_back({
                "neural_statistics_info.txt",
                capture::ArtifactType::StatisticsInfo,
                {},
                std::string(reinterpret_cast<const char *>(statisticsInfo.data()),
                            statisticsInfo.size()),
                true,
            });
        }
        else
        {
            publication.artifacts.push_back({
                "neural_statistics_info.bin",
                capture::ArtifactType::StatisticsInfo,
                std::move(statisticsInfo),
                {},
                false,
            });
        }
    }
    else
    {
        metadata.warnings.emplace_back("neural statistics info was unavailable for pipeline " +
                                       std::to_string(pipeline->id.value()));
        metadataChanged = true;
    }

    if (pipeline->shader != nullptr && !pipeline->shader->spirv.empty())
    {
        publication.artifacts.push_back({
            "shader_module_" + std::to_string(pipeline->shader->id.value()) + ".spv",
            capture::ArtifactType::ShaderModule,
            pipeline->shader->spirv,
            {},
            false,
        });
    }
    return publication;
}

CaptureJobStatus QueuePreparedPipelinePublication(Instance *instance,
                                                  PreparedPipelinePublication &publication) noexcept
{
    const auto identity = SubmitWriterJob(
        instance,
        [&](capture::CaptureWriter &writer)
        {
            return writer.trackPipeline(publication.id, std::move(publication.metadata),
                                        publication.diagnosticHandle);
        },
        "pipeline identity writer job was rejected during object creation");
    if (identity != CaptureJobStatus::Accepted)
    {
        return identity;
    }

    for (const auto &artifact : publication.artifacts)
    {
        const auto status = SubmitWriterJob(
            instance,
            [&](capture::CaptureWriter &writer)
            {
                if (artifact.text)
                {
                    return writer.writePipelineTextArtifact(publication.id, artifact.fileName,
                                                            artifact.type, artifact.textContents);
                }
                return writer.writePipelineBinaryArtifact(publication.id, artifact.fileName,
                                                          artifact.type, artifact.binaryContents);
            },
            "static pipeline artifact writer job was rejected during object creation");
        if (status != CaptureJobStatus::Accepted)
        {
            return status;
        }
    }
    return CaptureJobStatus::Accepted;
}

using ReleasedSnapshots = std::vector<CommandBufferRecord::DispatchOccurrence>;

[[nodiscard]] ReleasedSnapshots ReleaseSnapshots(
    const std::shared_ptr<CommandBufferRecord> &record) noexcept
{
    // StatsSnapshot destroys downstream Vulkan allocations when its final
    // shared owner releases it. Detach the owner-bearing occurrences so callers
    // can reset metadata while serialized, then destroy the returned batch only
    // after releasing g_vulkanLock. Swapping the vectors cannot allocate.
    ReleasedSnapshots released;
    released.swap(record->dispatches);
    record->reset();
    return released;
}

void SetSnapshotErrorNoexcept(const std::shared_ptr<CommandBufferRecord> &record,
                              const char *message) noexcept
{
    try
    {
        record->snapshotError = message;
    }
    catch (...)
    {
    }
    LAYER_ERR("%s", message);
}

bool CommandBufferSupportsSnapshots(Device *layer,
                                    const std::shared_ptr<CommandBufferRecord> &commandBuffer,
                                    std::string &reason)
{
    const auto commandPool = layer->resourceManager.getCommandPoolRecord(commandBuffer->parent);
    if (commandPool == nullptr ||
        commandBuffer->queueFamilyIndex >= layer->queueFamilyProperties.size())
    {
        reason = "capture cannot determine the command-buffer queue family";
        return false;
    }
    if (layer->synchronization2Route == capture_policy::Synchronization2Route::Unavailable)
    {
        reason = "selected capture requires an available core or KHR vkCmdPipelineBarrier2 command";
        return false;
    }

    const auto flags = layer->queueFamilyProperties[commandBuffer->queueFamilyIndex].queueFlags;
    switch (capture_policy::ClassifyCommandBuffer(commandBuffer->isProtected,
                                                  commandPool->hasForeignProcessingEngine, flags))
    {
    case capture_policy::CommandBufferDecision::Capture:
        return true;
    case capture_policy::CommandBufferDecision::RejectProtected:
        reason = "selected capture rejects protected command buffers";
        return false;
    case capture_policy::CommandBufferDecision::RejectForeignProcessingEngine:
        reason = "selected capture rejects foreign data-graph processing engines";
        return false;
    case capture_policy::CommandBufferDecision::RejectMissingTransferCapability:
        reason = "selected capture requires a transfer-capable queue family";
        return false;
    }
    reason = "unknown command-buffer capture policy result";
    return false;
}

uint32_t TestPostGraphCopyCount() noexcept
{
    static const uint32_t count = []() noexcept
    {
        constexpr unsigned long maxCopies = 65536;
        const char *text = std::getenv("VK_LAYER_TEST_POST_GRAPH_COPY_COUNT");
        if (text == nullptr || *text == '\0')
        {
            return 1u;
        }
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (end == text || *end != '\0' || parsed < 1 || parsed > maxCopies)
        {
            LAYER_ERR("Ignoring invalid test-only post-graph copy count '%s'", text);
            return 1u;
        }
        return static_cast<uint32_t>(parsed);
    }();
    return count;
}

void RecordPipelineBarrier2(Device *layer, VkCommandBuffer commandBuffer,
                            const VkDependencyInfo *dependencyInfo) noexcept
{
    const auto function = capture_policy::SelectCmdPipelineBarrier2Function(
        layer->synchronization2Route, layer->driver.vkCmdPipelineBarrier2,
        layer->driver.vkCmdPipelineBarrier2KHR);
    if (function != nullptr)
    {
        function(commandBuffer, dependencyInfo);
    }
}

VkResult CreateSnapshot(Device *layer, const std::shared_ptr<SessionRecord> &session,
                        uint32_t queueFamilyIndex, std::shared_ptr<StatsSnapshot> &snapshot)
{
    snapshot.reset();
    auto ownedSnapshot = std::make_shared<StatsSnapshot>(
        layer->device, layer->driver.vkDestroyBuffer, layer->driver.vkFreeMemory);

    const auto &stats = session->statsMemory;
    const VkBufferCreateInfo bufferInfo{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        stats.dataSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };

    VkResult result =
        layer->driver.vkCreateBuffer(layer->device, &bufferInfo, nullptr, &ownedSnapshot->buffer);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    VkMemoryRequirements requirements{};
    layer->driver.vkGetBufferMemoryRequirements(layer->device, ownedSnapshot->buffer,
                                                &requirements);

    VkMemoryPropertyFlags memoryProperties = 0;
    uint32_t memoryTypeIndex = 0;
    if (!FindMemoryTypeIndex(
            layer, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            memoryTypeIndex, &memoryProperties))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    const VkMemoryAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        memoryTypeIndex,
    };

    result = layer->driver.vkAllocateMemory(layer->device, &allocateInfo, nullptr,
                                            &ownedSnapshot->memory);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    result = layer->driver.vkBindBufferMemory(layer->device, ownedSnapshot->buffer,
                                              ownedSnapshot->memory, 0);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    ownedSnapshot->session = session;
    ownedSnapshot->dataSize = stats.dataSize;
    ownedSnapshot->allocationSize = requirements.size;
    ownedSnapshot->queueFamilyIndex = queueFamilyIndex;
    ownedSnapshot->memoryProperties = memoryProperties;
    snapshot = std::move(ownedSnapshot);
    return VK_SUCCESS;
}

VkResult CreateSnapshotWithoutGlobalLock(std::unique_lock<std::mutex> &lock, Device *layer,
                                         const std::shared_ptr<SessionRecord> &session,
                                         uint32_t queueFamilyIndex,
                                         std::shared_ptr<StatsSnapshot> &snapshot)
{
    assert(lock.owns_lock());
    lock.unlock();
    VkResult result = VK_ERROR_UNKNOWN;
    try
    {
        result = CreateSnapshot(layer, session, queueFamilyIndex, snapshot);
    }
    catch (...)
    {
        lock.lock();
        throw;
    }
    lock.lock();
    return result;
}
struct SubmittedCommandBufferRange
{
    uint32_t submitIndex{0};
    uint32_t commandBufferIndex{0};
    size_t firstOccurrence{0};
    size_t occurrenceCount{0};
    VkCommandBuffer archiveCommandBuffer{VK_NULL_HANDLE};
};

void RecordSnapshotArchiveCopies(
    Device *layer, VkCommandBuffer commandBuffer,
    const std::vector<std::pair<std::shared_ptr<StatsSnapshot>, std::shared_ptr<StatsSnapshot>>>
        &copies) noexcept
{
    if (copies.empty())
    {
        return;
    }

    const VkMemoryBarrier2 sourceWriteToArchiveRead{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,   VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,   VK_ACCESS_2_TRANSFER_READ_BIT,
    };
    const VkDependencyInfo sourceDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        nullptr,
        0,
        1,
        &sourceWriteToArchiveRead,
        0,
        nullptr,
        0,
        nullptr,
    };
    RecordPipelineBarrier2(layer, commandBuffer, &sourceDependency);

    for (const auto &[source, destination] : copies)
    {
        const VkBufferCopy region{0, 0, source->dataSize};
        layer->driver.vkCmdCopyBuffer(commandBuffer, source->buffer, destination->buffer, 1,
                                      &region);
    }

    // The archive read must complete before a later execution may overwrite the
    // record-time snapshot. DATA_GRAPH makes the application's session semaphore
    // dependency carry this injected transfer across queues. HOST makes the
    // unique submission destination visible to the asynchronous collector.
    const VkMemoryBarrier2 archiveToSessionAndHost{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM | VK_PIPELINE_STAGE_2_HOST_BIT,
        VK_ACCESS_2_HOST_READ_BIT,
    };
    const VkDependencyInfo archiveDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        nullptr,
        0,
        1,
        &archiveToSessionAndHost,
        0,
        nullptr,
        0,
        nullptr,
    };
    RecordPipelineBarrier2(layer, commandBuffer, &archiveDependency);
}

VkResult MaterializeSubmissionArchive(Device *layer,
                                      std::vector<SubmittedCommandBufferRange> &ranges,
                                      capture::submission::Plan &plan,
                                      std::shared_ptr<SubmissionArchive> &archive,
                                      std::string &reason)
{
    std::vector<std::shared_ptr<StatsSnapshot>> destinations;
    destinations.reserve(plan.selectedCount);
    for (const auto &occurrence : plan.occurrences)
    {
        if (!occurrence.dispatchId.has_value())
        {
            continue;
        }
        std::shared_ptr<StatsSnapshot> destination;
        const VkResult result = CreateSnapshot(
            layer, occurrence.session, occurrence.sourceSnapshot->queueFamilyIndex, destination);
        if (result != VK_SUCCESS)
        {
            reason = "selected capture failed to create a unique submission snapshot destination";
            return result;
        }
        destinations.emplace_back(std::move(destination));
    }
    if (!capture::submission::materialize(plan, std::move(destinations), reason))
    {
        return VK_ERROR_UNKNOWN;
    }
    if (plan.selectedSnapshots.empty())
    {
        return VK_SUCCESS;
    }

    uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uint32_t archiveCount = 0;
    for (const auto &range : ranges)
    {
        bool selected = false;
        for (size_t i = range.firstOccurrence; i < range.firstOccurrence + range.occurrenceCount;
             ++i)
        {
            const auto &occurrence = plan.occurrences[i];
            if (!occurrence.dispatchId.has_value())
            {
                continue;
            }
            selected = true;
            const uint32_t family = occurrence.sourceSnapshot->queueFamilyIndex;
            if (queueFamilyIndex == VK_QUEUE_FAMILY_IGNORED)
            {
                queueFamilyIndex = family;
            }
            else if (queueFamilyIndex != family)
            {
                reason = "selected capture archive spans incompatible queue families";
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
        if (selected)
        {
            ++archiveCount;
        }
    }

    archive =
        std::make_shared<SubmissionArchive>(layer->device, layer->driver.vkDestroyCommandPool);
    const VkCommandPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        queueFamilyIndex,
    };
    VkResult result =
        layer->driver.vkCreateCommandPool(layer->device, &poolInfo, nullptr, &archive->commandPool);
    if (result != VK_SUCCESS)
    {
        reason = "selected capture failed to create a submission archive command pool";
        return result;
    }

    archive->commandBuffers.resize(archiveCount);
    const VkCommandBufferAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        archive->commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        archiveCount,
    };
    result = layer->driver.vkAllocateCommandBuffers(layer->device, &allocateInfo,
                                                    archive->commandBuffers.data());
    if (result != VK_SUCCESS)
    {
        reason = "selected capture failed to allocate submission archive command buffers";
        return result;
    }

    uint32_t archiveIndex = 0;
    for (auto &range : ranges)
    {
        std::vector<std::pair<std::shared_ptr<StatsSnapshot>, std::shared_ptr<StatsSnapshot>>>
            copies;
        for (size_t i = range.firstOccurrence; i < range.firstOccurrence + range.occurrenceCount;
             ++i)
        {
            const auto &occurrence = plan.occurrences[i];
            if (occurrence.dispatchId.has_value())
            {
                copies.emplace_back(occurrence.sourceSnapshot, occurrence.snapshot);
            }
        }
        if (copies.empty())
        {
            continue;
        }

        range.archiveCommandBuffer = archive->commandBuffers[archiveIndex++];
        const VkCommandBufferBeginInfo beginInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        result = layer->driver.vkBeginCommandBuffer(range.archiveCommandBuffer, &beginInfo);
        if (result != VK_SUCCESS)
        {
            reason = "selected capture failed to begin a submission archive command buffer";
            return result;
        }
        RecordSnapshotArchiveCopies(layer, range.archiveCommandBuffer, copies);
        result = layer->driver.vkEndCommandBuffer(range.archiveCommandBuffer);
        if (result != VK_SUCCESS)
        {
            reason = "selected capture failed to end a submission archive command buffer";
            return result;
        }
    }
    return VK_SUCCESS;
}

VkResult MaterializeSubmissionArchiveWithoutGlobalLock(
    std::unique_lock<std::mutex> &lock, Device *layer,
    std::vector<SubmittedCommandBufferRange> &ranges, capture::submission::Plan &plan,
    std::shared_ptr<SubmissionArchive> &archive, std::string &reason)
{
    assert(lock.owns_lock());
    // Submit callers retain Instance::submissionAdmissionMutex while this drops
    // g_vulkanLock. That admission lock serializes materialize() in-flight
    // mutations against other submissions and collector finalization.
    lock.unlock();
    VkResult result = VK_ERROR_UNKNOWN;
    try
    {
        result = MaterializeSubmissionArchive(layer, ranges, plan, archive, reason);
    }
    catch (...)
    {
        lock.lock();
        throw;
    }
    lock.lock();
    return result;
}
bool AcceptAndPublishSubmissionPlan(Instance *instance, capture::submission::Plan &plan) noexcept
{
    const bool committed = instance->commitCaptureIfEnabled(
        [&]() noexcept { capture::submission::accept(instance->dispatchIds, plan); });
    if (!committed)
    {
        return false;
    }
    for (const auto &occurrence : plan.occurrences)
    {
        if (!occurrence.dispatchId.has_value())
        {
            continue;
        }
        (void)SubmitWriterJob(
            instance,
            [&](capture::CaptureWriter &writer)
            {
                return writer.materializeDispatch(occurrence.session->id,
                                                  {occurrence.executedIndex, occurrence.dispatchId},
                                                  std::nullopt);
            },
            "dispatch materialization writer job was rejected after downstream submission");
    }
    return true;
}

std::unique_ptr<capture::GpuCollectorJob> PrepareCollectorJob(
    capture::submission::Plan &plan, std::shared_ptr<SubmissionArchive> archive,
    uint32_t statisticsMode)
{
    capture::fault::Checkpoint(capture::fault::Point::CollectorJobConstruction);
    auto job = std::make_unique<capture::GpuCollectorJob>();
    job->plan = std::move(plan);
    job->archive = std::move(archive);
    job->statisticsMode = statisticsMode;
    return job;
}

VkResult SubmitLegacyMarker(Device *layer, VkQueue queue, VkFence fence) noexcept
{
    const VkSubmitInfo markerSubmit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 0, nullptr, 0, nullptr,
    };
    return layer->driver.vkQueueSubmit(queue, 1, &markerSubmit, fence);
}

VkResult Submit2Marker(Device *layer, capture_policy::Submit2Route route, VkQueue queue,
                       VkFence fence) noexcept
{
    const auto submit = capture_policy::SelectQueueSubmit2Function(
        route, layer->driver.vkQueueSubmit2, layer->driver.vkQueueSubmit2KHR);
    if (submit == nullptr)
    {
        return VK_ERROR_UNKNOWN;
    }
    const VkSubmitInfo2 markerSubmit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 0, nullptr, 0, nullptr,
    };
    return submit(queue, 1, &markerSubmit, fence);
}

void DeferAcceptedSubmissionResources(Device *layer, capture::GpuCollectorJob &job) noexcept
{
    try
    {
        if (job.archive != nullptr)
        {
            layer->deferredSubmissionArchives.emplace_back(job.archive);
        }
        for (const auto &snapshot : job.plan.selectedSnapshots)
        {
            layer->deferredSubmissionSnapshots.emplace_back(snapshot);
        }
        for (const auto &occurrence : job.plan.occurrences)
        {
            if (occurrence.sourceSnapshot != nullptr)
            {
                layer->deferredSubmissionSnapshots.emplace_back(occurrence.sourceSnapshot);
            }
        }
    }
    catch (...)
    {
        // The submission has already been accepted. If retaining cleanup state
        // fails, leak rather than destroy resources still referenced by the GPU.
        if (job.archive != nullptr)
        {
            job.archive->commandPool = VK_NULL_HANDLE;
        }
        for (const auto &snapshot : job.plan.selectedSnapshots)
        {
            snapshot->buffer = VK_NULL_HANDLE;
            snapshot->memory = VK_NULL_HANDLE;
        }
        for (const auto &occurrence : job.plan.occurrences)
        {
            if (occurrence.sourceSnapshot != nullptr)
            {
                occurrence.sourceSnapshot->buffer = VK_NULL_HANDLE;
                occurrence.sourceSnapshot->memory = VK_NULL_HANDLE;
            }
        }
    }
}

void FinishPostForwardCaptureFailure(Device *layer, Instance *instance,
                                     capture::GpuCollectorJob *job, capture::submission::Plan &plan,
                                     const char *reservationFailure,
                                     const char *terminalFailure) noexcept
{
    if (job != nullptr)
    {
        DeferAcceptedSubmissionResources(layer, *job);
    }
    capture::submission::finish(plan, reservationFailure);
    DisableCapture(instance, terminalFailure);
}
} // namespace

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyDevice<user_tag>(VkDevice device,
                                                           const VkAllocationCallbacks *pAllocator)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto layer = Device::destroy(device);

    // vkDestroyDevice is externally synchronized. Once the device is removed
    // from the dispatch map there can be no legal concurrent layer call holding
    // another snapshot owner. Drop command-buffer ownership before destroying
    // the downstream device so final-owner RAII can call vkDestroyBuffer and
    // vkFreeMemory while the device and copied dispatch functions remain valid.
    lock.unlock();
    if (layer->collector != nullptr)
    {
        layer->collector->shutdown();
        layer->collector.reset();
    }
    layer->resourceManager.clearCommandBufferRecords();
    if (!layer->deferredSubmissionArchives.empty() || !layer->deferredSubmissionSnapshots.empty())
    {
        (void)layer->driver.vkDeviceWaitIdle(device);
        layer->deferredSubmissionArchives.clear();
        layer->deferredSubmissionSnapshots.clear();
    }
    layer->driver.vkDestroyDevice(device, pAllocator);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateShaderModule<user_tag>(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule)
{
    ShaderCreationRollback rollback;
    try
    {
        std::unique_lock<std::mutex> lock{g_vulkanLock};
        auto *layer = Device::retrieve(device);
        auto *instance = const_cast<Instance *>(layer->instance);
        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            return layer->driver.vkCreateShaderModule(device, pCreateInfo, pAllocator,
                                                      pShaderModule);
        }
        lock.unlock();

        std::vector<uint8_t> spirv(pCreateInfo->codeSize);
        if (!spirv.empty())
        {
            std::memcpy(spirv.data(), pCreateInfo->pCode, pCreateInfo->codeSize);
        }

        const VkResult result =
            layer->driver.vkCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        rollback.arm(layer, device, pAllocator, pShaderModule);

        lock.lock();
        if (!instance->isCaptureEnabled())
        {
            rollback.release();
            return VK_SUCCESS;
        }
        const capture::ShaderModuleId id = instance->shaderModuleIds.peek();
        capture::fault::Checkpoint(capture::fault::Point::ShaderRecordBeforeInsertion);
        auto record =
            layer->resourceManager.addShaderModuleRecord(*pShaderModule, id, std::move(spirv));
        if (record == nullptr)
        {
            return VK_ERROR_UNKNOWN;
        }
        try
        {
            capture::fault::Checkpoint(
                capture::fault::Point::ShaderRecordAfterInsertionBeforeCommit);
        }
        catch (...)
        {
            layer->resourceManager.removeShaderModuleRecord(*pShaderModule);
            throw;
        }
        instance->shaderModuleIds.commit();
        rollback.release();
        return VK_SUCCESS;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyShaderModule<user_tag>(
    VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks *pAllocator)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);
    layer->resourceManager.removeShaderModuleRecord(shaderModule);
    lock.unlock();
    layer->driver.vkDestroyShaderModule(device, shaderModule, pAllocator);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDataGraphPipelinesARM<user_tag>(
    VkDevice device, VkDeferredOperationKHR deferredOperation, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkDataGraphPipelineCreateInfoARM *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
    PipelineCreationRollback rollback;
    try
    {
        std::unique_lock<std::mutex> lock{g_vulkanLock};
        auto *layer = Device::retrieve(device);
        auto *instance = const_cast<Instance *>(layer->instance);
        if (!layer->neuralStatisticsEnabled)
        {
            // This device cannot use the capture extension. Preserve the
            // application create infos exactly and do not publish tracking
            // records that could instrument sessions or submissions later.
            lock.unlock();
            return layer->driver.vkCreateDataGraphPipelinesARM(
                device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
                pPipelines);
        }
        if (!instance->isCaptureEnabled())
        {
            // Capture has already failed terminally. Pipeline instrumentation
            // and tracking are capture-only, so preserve the application call
            // exactly instead of turning the earlier capture failure into a
            // permanent pipeline-creation failure.
            lock.unlock();
            return layer->driver.vkCreateDataGraphPipelinesARM(
                device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
                pPipelines);
        }

        std::vector<capture::PipelineMetadata> metadata;
        std::vector<std::shared_ptr<ShaderModuleRecord>> shaders;
        std::vector<VkDataGraphPipelineCreateInfoARM> patchedCreateInfos(createInfoCount);
        std::vector<VkDataGraphPipelineNeuralStatisticsCreateInfoARM> statsCreateInfos(
            createInfoCount);
        std::vector<std::string> unsupportedReasons(createInfoCount);
        metadata.reserve(createInfoCount);
        shaders.reserve(createInfoCount);

        const bool deferred = deferredOperation != VK_NULL_HANDLE;
        for (uint32_t i = 0; i < createInfoCount; ++i)
        {
            std::shared_ptr<ShaderModuleRecord> shader;
            metadata.emplace_back(CopyPipelineMetadata(layer, pCreateInfos[i], shader));
            shaders.emplace_back(std::move(shader));
            patchedCreateInfos[i] = pCreateInfos[i];

            const auto *userStatsCreateInfo =
                vku::FindStructInPNextChain<VkDataGraphPipelineNeuralStatisticsCreateInfoARM>(
                    pCreateInfos[i].pNext);
            if (deferred)
            {
                metadata[i].statisticsEnabled = false;
                unsupportedReasons[i] =
                    "selected capture does not support deferred data-graph pipeline creation";
            }
            else if ((pCreateInfos[i].flags & VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT_EXT) !=
                     0)
            {
                // VUID-VkDataGraphPipelineCreateInfoARM-flags-09848 forbids
                // neural-statistics instrumentation on protected-only pipelines.
                metadata[i].statisticsEnabled = false;
                unsupportedReasons[i] = "selected capture rejects protected data-graph pipelines";
            }
            else if (userStatsCreateInfo != nullptr)
            {
                metadata[i].statisticsEnabled = false;
                unsupportedReasons[i] = "selected capture cannot override application-provided "
                                        "pipeline statistics state";
            }
            else
            {
                metadata[i].statisticsEnabled = true;
                statsCreateInfos[i] = {
                    VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_NEURAL_STATISTICS_CREATE_INFO_ARM,
                    patchedCreateInfos[i].pNext,
                    VK_TRUE,
                };
                patchedCreateInfos[i].pNext = &statsCreateInfos[i];
            }
        }

        std::vector<std::vector<uint8_t>> debugDatabases(createInfoCount);
        std::vector<std::vector<uint8_t>> statisticsInfos(createInfoCount);
        std::vector<uint8_t> statisticsInfoIsText(createInfoCount, 0);

        lock.unlock();
        const VkResult downstreamResult = layer->driver.vkCreateDataGraphPipelinesARM(
            device, deferredOperation, pipelineCache, createInfoCount, patchedCreateInfos.data(),
            pAllocator, pPipelines);
        bool hasPublishedPipeline = false;
        for (uint32_t i = 0; i < createInfoCount; ++i)
        {
            hasPublishedPipeline |= pPipelines[i] != VK_NULL_HANDLE;
        }
        // Vulkan pipeline-creation commands may return a non-success result
        // while still publishing earlier successful handles in the batch.
        // Those handles remain application-observable and must participate in
        // the same tracking transaction as VK_SUCCESS/COMPILE_REQUIRED output.
        if (!hasPublishedPipeline)
        {
            return downstreamResult;
        }
        rollback.arm(layer, device, pAllocator, createInfoCount, pPipelines);
        if (!instance->isCaptureEnabled())
        {
            rollback.release();
            return downstreamResult;
        }

        for (uint32_t i = 0; i < createInfoCount; ++i)
        {
            if (pPipelines[i] == VK_NULL_HANDLE || !unsupportedReasons[i].empty())
            {
                continue;
            }
            (void)GetPipelinePropertyData(
                layer, device, pPipelines[i],
                VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_DEBUG_DATABASE_ARM,
                debugDatabases[i]);
            bool infoIsText = false;
            (void)GetPipelinePropertyData(
                layer, device, pPipelines[i],
                VK_DATA_GRAPH_PIPELINE_PROPERTY_NEURAL_ACCELERATOR_STATISTICS_INFO_ARM,
                statisticsInfos[i], &infoIsText);
            statisticsInfoIsText[i] = infoIsText ? 1u : 0u;
        }
        lock.lock();
        if (!instance->isCaptureEnabled())
        {
            rollback.release();
            return downstreamResult;
        }

        size_t publishedCount = 0;
        for (uint32_t i = 0; i < createInfoCount; ++i)
        {
            if (pPipelines[i] != VK_NULL_HANDLE)
            {
                ++publishedCount;
            }
        }
        std::optional<capture::PipelineId> firstId;
        if (publishedCount != 0)
        {
            firstId = instance->pipelineIds.peek();
            if (firstId->value() > std::numeric_limits<uint64_t>::max() - (publishedCount - 1))
            {
                return VK_ERROR_UNKNOWN;
            }
        }

        std::vector<std::shared_ptr<PipelineRecord>> records;
        std::vector<VkPipeline> insertedHandles;
        std::vector<uint32_t> recordSourceIndices;
        records.reserve(publishedCount);
        insertedHandles.reserve(publishedCount);
        recordSourceIndices.reserve(publishedCount);
        try
        {
            size_t ordinal = 0;
            for (uint32_t i = 0; i < createInfoCount; ++i)
            {
                if (pPipelines[i] == VK_NULL_HANDLE)
                {
                    continue;
                }
                capture::fault::Checkpoint(capture::fault::Point::PipelineRecordBeforeInsertion);
                auto record = layer->resourceManager.addPipelineRecord(
                    pPipelines[i], capture::PipelineId(firstId->value() + ordinal),
                    std::move(metadata[i]), std::move(shaders[i]));
                if (record == nullptr)
                {
                    throw std::runtime_error("pipeline record insertion failed");
                }
                insertedHandles.emplace_back(pPipelines[i]);
                record->capturable = unsupportedReasons[i].empty();
                record->unsupportedReason = std::move(unsupportedReasons[i]);
                records.emplace_back(std::move(record));
                recordSourceIndices.emplace_back(i);
                ++ordinal;
                capture::fault::Checkpoint(
                    capture::fault::Point::PipelineRecordAfterInsertionBeforeCommit);
            }
        }
        catch (...)
        {
            for (VkPipeline handle : insertedHandles)
            {
                layer->resourceManager.removePipelineRecord(handle);
            }
            throw;
        }

        // Preconstruct every fallible immutable payload for the full batch
        // before accepting the first writer identity. Once publication starts,
        // only noexcept writer submissions and nonthrowing state transitions
        // remain in the intercepted thread.
        const bool publishCapture = instance->captureWriter != nullptr;
        capture::CaptureMetadata preparedCaptureMetadata;
        bool captureMetadataChanged = false;
        std::vector<PreparedPipelinePublication> publications;
        try
        {
            if (publishCapture)
            {
                preparedCaptureMetadata = instance->captureMetadata;
                publications.reserve(records.size());
                for (size_t recordIndex = 0; recordIndex < records.size(); ++recordIndex)
                {
                    const auto &record = records[recordIndex];
                    if (record->capturable)
                    {
                        const uint32_t sourceIndex = recordSourceIndices[recordIndex];
                        publications.emplace_back(PreparePipelinePublication(
                            record, std::move(debugDatabases[sourceIndex]),
                            std::move(statisticsInfos[sourceIndex]),
                            statisticsInfoIsText[sourceIndex] != 0, preparedCaptureMetadata,
                            captureMetadataChanged));
                    }
                    else
                    {
                        publications.push_back({
                            record->id,
                            record->metadata,
                            DiagnosticHandleValue(record->handle),
                            {},
                        });
                    }
                }
            }
        }
        catch (...)
        {
            for (VkPipeline handle : insertedHandles)
            {
                layer->resourceManager.removePipelineRecord(handle);
            }
            throw;
        }

        // The Vulkan handles, tracking maps, capture IDs, and writer identity
        // records form one object-creation transaction. Immediate enqueue
        // failure must not leave live mappings or committed IDs behind.
        if (publishCapture)
        {
            CaptureJobStatus publicationStatus = CaptureJobStatus::Accepted;
            for (auto &publication : publications)
            {
                publicationStatus = QueuePreparedPipelinePublication(instance, publication);
                if (publicationStatus != CaptureJobStatus::Accepted)
                {
                    break;
                }
            }
            if (publicationStatus == CaptureJobStatus::Accepted && captureMetadataChanged)
            {
                publicationStatus = SubmitWriterJob(
                    instance, [&](capture::CaptureWriter &writer)
                    { return writer.updateMetadata(preparedCaptureMetadata); },
                    "pipeline warning metadata writer job was rejected during object creation");
            }
            if (publicationStatus != CaptureJobStatus::Accepted)
            {
                for (VkPipeline handle : insertedHandles)
                {
                    layer->resourceManager.removePipelineRecord(handle);
                }
                if (publicationStatus == CaptureJobStatus::Skipped)
                {
                    rollback.release();
                    return downstreamResult;
                }
                DisableCapture(instance,
                               "pipeline writer publication was rejected during object creation");
                const auto writerSnapshot = instance->captureWriter->snapshot();
                return writerSnapshot.terminalErrorCode == capture::WriterErrorCode::OutOfHostMemory
                           ? VK_ERROR_OUT_OF_HOST_MEMORY
                           : VK_ERROR_UNKNOWN;
            }
            if (captureMetadataChanged)
            {
                instance->captureMetadata = std::move(preparedCaptureMetadata);
            }
        }

        for (size_t i = 0; i < publishedCount; ++i)
        {
            instance->pipelineIds.commit();
        }
        rollback.release();

        return downstreamResult;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyPipeline<user_tag>(
    VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);

    layer->resourceManager.removePipelineRecord(pipeline);

    lock.unlock();
    layer->driver.vkDestroyPipeline(device, pipeline, pAllocator);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDataGraphPipelineSessionARM<user_tag>(
    VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDataGraphPipelineSessionARM *pSession)
{
    SessionCreationRollback rollback;
    Device *layer = nullptr;
    std::shared_ptr<PipelineRecord> pipelineRecord;
    bool instrumentationForwardStarted = false;
    bool instrumentationForwardCompleted = false;
    try
    {
        std::unique_lock<std::mutex> lock{g_vulkanLock};
        layer = Device::retrieve(device);
        auto *instance = const_cast<Instance *>(layer->instance);

        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            return layer->driver.vkCreateDataGraphPipelineSessionARM(device, pCreateInfo,
                                                                     pAllocator, pSession);
        }
        // If we don't have a record of the pipeline, don't track its sessions.
        pipelineRecord = layer->resourceManager.getPipelineRecord(pCreateInfo->dataGraphPipeline);
        if (pipelineRecord == nullptr)
        {
            lock.unlock();
            return layer->driver.vkCreateDataGraphPipelineSessionARM(device, pCreateInfo,
                                                                     pAllocator, pSession);
        }

        // Session instrumentation has a distinct fallback policy: discard any
        // partially created instrumented session, retry the application's original
        // create info, and track a successful result as noncapturable. A later
        // selected use reports the unsupported reason through terminal capture state.
        auto createOriginalSession = [&](const char *unsupportedReason) -> VkResult
        {
            if (lock.owns_lock())
            {
                lock.unlock();
            }
            rollback.rollbackNow();
            return CreateOriginalSessionPlaceholder(layer, device, pCreateInfo, pAllocator,
                                                    pSession, pipelineRecord, unsupportedReason);
        };

        const bool protectedSession =
            (pCreateInfo->flags & VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_PROTECTED_BIT_ARM) != 0;
        const bool protectedPipeline = (pipelineRecord->metadata.flags &
                                        VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT_EXT) != 0;
        const auto *applicationSessionStatistics =
            vku::FindStructInPNextChain<VkDataGraphPipelineSessionNeuralStatisticsCreateInfoARM>(
                pCreateInfo->pNext);
        std::string captureUnavailableReason;
        if (!pipelineRecord->capturable)
        {
            captureUnavailableReason = pipelineRecord->unsupportedReason;
        }
        else if (protectedSession || protectedPipeline)
        {
            captureUnavailableReason = "selected capture rejects protected data-graph execution";
        }
        else if (pipelineRecord->metadata.foreignProcessingEngine)
        {
            captureUnavailableReason = "selected capture rejects foreign processing engines";
        }
        else if (!pipelineRecord->metadata.statisticsEnabled)
        {
            captureUnavailableReason =
                "selected capture cannot override application-provided pipeline statistics state";
        }
        else if (applicationSessionStatistics != nullptr)
        {
            captureUnavailableReason =
                "selected capture cannot override application-provided session statistics state";
        }
        else if (!layer->synchronization2Enabled ||
                 layer->synchronization2Route == capture_policy::Synchronization2Route::Unavailable)
        {
            captureUnavailableReason = "selected capture requires synchronization2 support";
        }
        else if (layer->driver.vkGetBufferMemoryRequirements2 == nullptr &&
                 layer->driver.vkGetBufferMemoryRequirements2KHR == nullptr)
        {
            captureUnavailableReason =
                "selected capture requires memory-requirements2 dedicated-allocation reporting";
        }
        if (!captureUnavailableReason.empty())
        {
            return createOriginalSession(captureUnavailableReason.c_str());
        }

        VkDataGraphPipelineSessionCreateInfoARM patchedCreateInfo = *pCreateInfo;
        const VkNeuralAcceleratorStatisticsModeARM statsMode =
            layer->instance->layerOptions->getStatsMode();
        VkDataGraphPipelineSessionNeuralStatisticsCreateInfoARM sessionStatsCreateInfo = {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_NEURAL_STATISTICS_CREATE_INFO_ARM,
            patchedCreateInfo.pNext,
            statsMode,
        };
        patchedCreateInfo.pNext = &sessionStatsCreateInfo;

        lock.unlock();
        instrumentationForwardStarted = true;
        VkResult result = layer->driver.vkCreateDataGraphPipelineSessionARM(
            device, &patchedCreateInfo, pAllocator, pSession);
        instrumentationForwardCompleted = true;
        if (result != VK_SUCCESS)
        {
            return createOriginalSession(
                "selected capture session instrumentation creation failed");
        }
        rollback.arm(layer, device, pAllocator, pSession);
        if (!instance->isCaptureEnabled())
        {
            return createOriginalSession("capture ended while creating the instrumented session");
        }

        const VkDataGraphPipelineSessionBindPointRequirementsInfoARM bindPointRequirementsInfo = {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
            nullptr,
            *pSession,
        };
        uint32_t bindPointCount = 0;
        result = layer->driver.vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            device, &bindPointRequirementsInfo, &bindPointCount, nullptr);
        if (result != VK_SUCCESS)
        {
            rollback.rollbackNow();
            return createOriginalSession(
                "selected capture could not query session bind-point requirements");
        }

        std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> bindPointRequirements(
            bindPointCount);
        for (auto &requirement : bindPointRequirements)
        {
            requirement.sType =
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
        }
        result = layer->driver.vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            device, &bindPointRequirementsInfo, &bindPointCount,
            bindPointRequirements.empty() ? nullptr : bindPointRequirements.data());
        if (result != VK_SUCCESS)
        {
            rollback.rollbackNow();
            return createOriginalSession(
                "selected capture could not enumerate session bind-point requirements");
        }
        bindPointRequirements.resize(bindPointCount);

        std::vector<VkDataGraphPipelineSessionBindPointRequirementARM>
            nonStatsBindPointRequirements;
        StatsMemory statsMemory{};
        bool foundStatsBindPoint = false;
        for (const auto &requirement : bindPointRequirements)
        {
            if (requirement.bindPoint !=
                VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_NEURAL_ACCELERATOR_STATISTICS_ARM)
            {
                nonStatsBindPointRequirements.emplace_back(requirement);
                continue;
            }

            if (foundStatsBindPoint ||
                requirement.bindPointType !=
                    VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM ||
                requirement.numObjects != 1)
            {
                LAYER_ERR("Structured capture rejected: invalid neural statistics bind-point "
                          "requirements");
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture found malformed neural-statistics bind-point requirements");
            }
            foundStatsBindPoint = true;

            const VkDataGraphPipelineSessionMemoryRequirementsInfoARM neuralStatsMemoryRequirements{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
                nullptr,
                *pSession,
                requirement.bindPoint,
                0,
            };
            VkMemoryDedicatedRequirements statsDedicatedRequirements{
                VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
                nullptr,
                VK_FALSE,
                VK_FALSE,
            };
            VkMemoryRequirements2 statsRequirements2{
                VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                &statsDedicatedRequirements,
                {},
            };
            layer->driver.vkGetDataGraphPipelineSessionMemoryRequirementsARM(
                device, &neuralStatsMemoryRequirements, &statsRequirements2);
            const VkMemoryRequirements &statsRequirements = statsRequirements2.memoryRequirements;

            const auto &sourceQueueFamilyIndices = layer->captureQueueFamilyIndices;
            const bool concurrentSourceSharing = sourceQueueFamilyIndices.size() > 1;
            const VkBufferCreateInfo sourceBufferInfo{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                nullptr,
                0,
                statsRequirements.size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                concurrentSourceSharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
                concurrentSourceSharing ? static_cast<uint32_t>(sourceQueueFamilyIndices.size())
                                        : 0,
                concurrentSourceSharing ? sourceQueueFamilyIndices.data() : nullptr,
            };
            LAYER_LOG("Structured capture statistics alias sharing=%s familyCount=%u",
                      concurrentSourceSharing ? "concurrent" : "exclusive",
                      sourceBufferInfo.queueFamilyIndexCount);

            VkBuffer sourceBuffer = VK_NULL_HANDLE;
            result =
                layer->driver.vkCreateBuffer(device, &sourceBufferInfo, nullptr, &sourceBuffer);
            if (result != VK_SUCCESS)
            {
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture could not create the statistics alias buffer");
            }
            rollback.adoptBuffer(sourceBuffer);

            VkMemoryDedicatedRequirements sourceDedicatedRequirements{
                VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
                nullptr,
                VK_FALSE,
                VK_FALSE,
            };
            VkMemoryRequirements2 sourceRequirements2{
                VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                &sourceDedicatedRequirements,
                {},
            };
            const VkBufferMemoryRequirementsInfo2 sourceRequirementsInfo{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
                nullptr,
                sourceBuffer,
            };
            if (layer->driver.vkGetBufferMemoryRequirements2 != nullptr)
            {
                layer->driver.vkGetBufferMemoryRequirements2(device, &sourceRequirementsInfo,
                                                             &sourceRequirements2);
            }
            else if (layer->driver.vkGetBufferMemoryRequirements2KHR != nullptr)
            {
                layer->driver.vkGetBufferMemoryRequirements2KHR(device, &sourceRequirementsInfo,
                                                                &sourceRequirements2);
            }
            else
            {
                rollback.rollbackNow();
                return createOriginalSession("selected capture requires memory-requirements2 "
                                             "dedicated-allocation reporting");
            }
            const VkMemoryRequirements &sourceRequirements = sourceRequirements2.memoryRequirements;
            const uint32_t commonMemoryTypeBits =
                statsRequirements.memoryTypeBits & sourceRequirements.memoryTypeBits;
            const VkDeviceSize allocationSize =
                std::max(statsRequirements.size, sourceRequirements.size);

            LAYER_LOG("Structured capture stats requirements: size=%llu align=%llu bits=0x%08X; "
                      "transfer-source buffer: "
                      "size=%llu align=%llu bits=0x%08X; common=0x%08X allocation=%llu",
                      static_cast<unsigned long long>(statsRequirements.size),
                      static_cast<unsigned long long>(statsRequirements.alignment),
                      statsRequirements.memoryTypeBits,
                      static_cast<unsigned long long>(sourceRequirements.size),
                      static_cast<unsigned long long>(sourceRequirements.alignment),
                      sourceRequirements.memoryTypeBits, commonMemoryTypeBits,
                      static_cast<unsigned long long>(allocationSize));
            LAYER_LOG("Structured capture dedicated requirements: stats requires=%u prefers=%u; "
                      "buffer requires=%u prefers=%u",
                      statsDedicatedRequirements.requiresDedicatedAllocation,
                      statsDedicatedRequirements.prefersDedicatedAllocation,
                      sourceDedicatedRequirements.requiresDedicatedAllocation,
                      sourceDedicatedRequirements.prefersDedicatedAllocation);

            if (statsDedicatedRequirements.requiresDedicatedAllocation == VK_TRUE ||
                sourceDedicatedRequirements.requiresDedicatedAllocation == VK_TRUE)
            {
                LAYER_ERR("Structured capture rejected: aliased resources require a dedicated "
                          "allocation");
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture cannot alias a resource requiring dedicated allocation");
            }
            if ((statsRequirements.size & 3u) != 0)
            {
                LAYER_ERR("Structured capture rejected: statistics allocation size %llu is not "
                          "legal for vkCmdFillBuffer",
                          static_cast<unsigned long long>(statsRequirements.size));
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture statistics size is not legal for vkCmdFillBuffer");
            }
            if (commonMemoryTypeBits == 0)
            {
                LAYER_ERR("Structured capture statistics memory and transfer-source buffer have no "
                          "compatible memory type");
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture has no common statistics and buffer memory type");
            }

            VkMemoryPropertyFlags memoryProperties = 0;
            uint32_t memoryTypeIndex = 0;
            if (!FindMemoryTypeIndex(layer, commonMemoryTypeBits, 0,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex,
                                     &memoryProperties))
            {
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture has no acceptable statistics memory type");
            }

            const VkMemoryAllocateInfo allocateInfo{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                nullptr,
                allocationSize,
                memoryTypeIndex,
            };
            VkDeviceMemory memory = VK_NULL_HANDLE;
            result = layer->driver.vkAllocateMemory(device, &allocateInfo, nullptr, &memory);
            if (result != VK_SUCCESS)
            {
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture could not allocate statistics memory");
            }
            rollback.adoptMemory(memory);

            result = layer->driver.vkBindBufferMemory(device, sourceBuffer, memory, 0);
            if (result != VK_SUCCESS)
            {
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture could not bind the statistics alias buffer");
            }

            const VkBindDataGraphPipelineSessionMemoryInfoARM bindMemoryInfo{
                VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
                nullptr,
                *pSession,
                requirement.bindPoint,
                0,
                memory,
                0,
            };
            result =
                layer->driver.vkBindDataGraphPipelineSessionMemoryARM(device, 1, &bindMemoryInfo);
            if (result != VK_SUCCESS)
            {
                rollback.rollbackNow();
                return createOriginalSession(
                    "selected capture could not bind neural-statistics session memory");
            }

            statsMemory = {
                memory, sourceBuffer, statsRequirements.size, allocationSize, memoryProperties,
            };
            LAYER_LOG("Structured capture aliased statistics allocation successfully: "
                      "memoryType=%u properties=0x%08X",
                      memoryTypeIndex, static_cast<unsigned>(memoryProperties));
        }

        if (!foundStatsBindPoint || statsMemory.handle == VK_NULL_HANDLE ||
            statsMemory.sourceBuffer == VK_NULL_HANDLE || statsMemory.dataSize == 0)
        {
            LAYER_ERR("Structured capture rejected: the created session has no usable neural "
                      "statistics memory");
            rollback.rollbackNow();
            return createOriginalSession(
                "selected capture found no usable neural-statistics session memory");
        }

        lock.lock();
        if (!instance->isCaptureEnabled())
        {
            return createOriginalSession("capture ended while preparing the instrumented session");
        }
        const capture::SessionId sessionId = instance->sessionIds.peek();
        auto sessionRecord = layer->resourceManager.addSessionRecord(
            *pSession, sessionId, pipelineRecord, std::move(nonStatsBindPointRequirements),
            statsMemory);
        if (sessionRecord == nullptr)
        {
            return VK_ERROR_UNKNOWN;
        }
        sessionRecord->flags = pCreateInfo->flags;
        sessionRecord->capturable = true;
        sessionRecord->unsupportedReason.clear();
        if (SubmitWriterJob(
                instance,
                [&](capture::CaptureWriter &writer)
                {
                    return writer.trackSession(sessionId, pipelineRecord->id, pCreateInfo->flags,
                                               DiagnosticHandleValue(*pSession));
                },
                "session tracking writer job was rejected after downstream creation") !=
                CaptureJobStatus::Accepted &&
            !instance->isCaptureEnabled())
        {
            layer->resourceManager.removeSessionRecord(*pSession);
            lock.unlock();
            rollback.rollbackNow();
            return CreateOriginalSessionPlaceholder(
                layer, device, pCreateInfo, pAllocator, pSession, pipelineRecord,
                "capture writer rejected instrumented session publication");
        }
        instance->sessionIds.commit();

        rollback.release();
        return VK_SUCCESS;
    }
    catch (const std::bad_alloc &)
    {
        rollback.rollbackNow();
        if (layer != nullptr && pipelineRecord != nullptr &&
            (!instrumentationForwardStarted || instrumentationForwardCompleted))
        {
            return CreateOriginalSessionPlaceholder(
                layer, device, pCreateInfo, pAllocator, pSession, pipelineRecord,
                "capture-only session allocation failed; original session preserved");
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        rollback.rollbackNow();
        if (layer != nullptr && pipelineRecord != nullptr &&
            (!instrumentationForwardStarted || instrumentationForwardCompleted))
        {
            return CreateOriginalSessionPlaceholder(
                layer, device, pCreateInfo, pAllocator, pSession, pipelineRecord,
                "capture-only session bookkeeping failed; original session preserved");
        }
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyDataGraphPipelineSessionARM<user_tag>(
    VkDevice device, VkDataGraphPipelineSessionARM session, const VkAllocationCallbacks *pAllocator)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);

    StatsMemory statsMemory{};
    if (const auto record = layer->resourceManager.getSessionRecord(session))
    {
        statsMemory = record->statsMemory;
        layer->resourceManager.removeSessionRecord(session);
    }

    lock.unlock();
    if (statsMemory.sourceBuffer != VK_NULL_HANDLE)
    {
        layer->driver.vkDestroyBuffer(device, statsMemory.sourceBuffer, nullptr);
    }
    layer->driver.vkDestroyDataGraphPipelineSessionARM(device, session, pAllocator);
    if (statsMemory.handle != VK_NULL_HANDLE)
    {
        layer->driver.vkFreeMemory(device, statsMemory.handle, nullptr);
    }
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL
layer_vkGetDataGraphPipelineSessionBindPointRequirementsARM<user_tag>(
    VkDevice device, const VkDataGraphPipelineSessionBindPointRequirementsInfoARM *pInfo,
    uint32_t *pBindPointRequirementCount,
    VkDataGraphPipelineSessionBindPointRequirementARM *pBindPointRequirements)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);

    const auto record = layer->resourceManager.getSessionRecord(pInfo->session);
    if (record == nullptr || !record->statisticsCaptureActive)
    {
        lock.unlock();
        return layer->driver.vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            device, pInfo, pBindPointRequirementCount, pBindPointRequirements);
    }

    const uint32_t total = static_cast<uint32_t>(record->filteredBindPointRequirements.size());
    if (pBindPointRequirements == nullptr)
    {
        *pBindPointRequirementCount = total;
        return VK_SUCCESS;
    }

    const uint32_t capacity = *pBindPointRequirementCount;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t i = 0; i < written; ++i)
    {
        pBindPointRequirements[i] = record->filteredBindPointRequirements[i];
    }

    *pBindPointRequirementCount = written;
    return (capacity < total) ? VK_INCOMPLETE : VK_SUCCESS;
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkAllocateCommandBuffers<user_tag>(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers)
{
    CommandBuffersRollback rollback;
    try
    {
        std::unique_lock<std::mutex> lock{g_vulkanLock};
        auto *layer = Device::retrieve(device);
        auto *instance = const_cast<Instance *>(layer->instance);

        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            return layer->driver.vkAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
        }

        lock.unlock();
        const VkResult result =
            layer->driver.vkAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        rollback.arm(layer, device, pAllocateInfo->commandPool, pAllocateInfo->commandBufferCount,
                     pCommandBuffers);

        lock.lock();
        if (!instance->isCaptureEnabled())
        {
            rollback.release();
            return VK_SUCCESS;
        }
        uint32_t publishedCount = 0;
        try
        {
            for (; publishedCount < pAllocateInfo->commandBufferCount; ++publishedCount)
            {
                if (layer->resourceManager.addCommandBufferRecord(
                        pCommandBuffers[publishedCount], pAllocateInfo->commandPool) == nullptr)
                {
                    for (uint32_t i = 0; i < publishedCount; ++i)
                    {
                        layer->resourceManager.removeCommandBufferRecord(pCommandBuffers[i]);
                    }
                    return VK_ERROR_UNKNOWN;
                }
            }
        }
        catch (const std::bad_alloc &)
        {
            for (uint32_t i = 0; i < publishedCount; ++i)
            {
                layer->resourceManager.removeCommandBufferRecord(pCommandBuffers[i]);
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (...)
        {
            for (uint32_t i = 0; i < publishedCount; ++i)
            {
                layer->resourceManager.removeCommandBufferRecord(pCommandBuffers[i]);
            }
            return VK_ERROR_UNKNOWN;
        }

        rollback.release();
        return VK_SUCCESS;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkBeginCommandBuffer<user_tag>(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo)
{
    ReleasedSnapshots releasedSnapshots;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(commandBuffer);
    auto *instance = const_cast<Instance *>(layer->instance);
    if (!instance->isCaptureEnabled())
    {
        lock.unlock();
        return layer->driver.vkBeginCommandBuffer(commandBuffer, pBeginInfo);
    }

    auto record = layer->resourceManager.getCommandBufferRecord(commandBuffer);
    assert(record);

    try
    {
        lock.unlock();
        const VkResult result = layer->driver.vkBeginCommandBuffer(commandBuffer, pBeginInfo);
        lock.lock();
        if (result == VK_SUCCESS)
        {
            releasedSnapshots = ReleaseSnapshots(record);
            record->usageFlags = pBeginInfo->flags;
        }
        lock.unlock();
        releasedSnapshots.clear();
        return result;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkResetCommandBuffer<user_tag>(VkCommandBuffer commandBuffer,
                                                                    VkCommandBufferResetFlags flags)
{
    ReleasedSnapshots releasedSnapshots;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(commandBuffer);
    auto *instance = const_cast<Instance *>(layer->instance);
    if (!instance->isCaptureEnabled())
    {
        lock.unlock();
        return layer->driver.vkResetCommandBuffer(commandBuffer, flags);
    }

    auto record = layer->resourceManager.getCommandBufferRecord(commandBuffer);
    assert(record);

    lock.unlock();
    const VkResult result = layer->driver.vkResetCommandBuffer(commandBuffer, flags);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    lock.lock();
    releasedSnapshots = ReleaseSnapshots(record);
    lock.unlock();
    releasedSnapshots.clear();
    return result;
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkFreeCommandBuffers<user_tag>(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers)
{
    ReleasedSnapshots releasedSnapshots;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);

    for (uint32_t i = 0; i < commandBufferCount; ++i)
    {
        auto record = layer->resourceManager.getCommandBufferRecord(pCommandBuffers[i]);
        if (record != nullptr)
        {
            releasedSnapshots = ReleaseSnapshots(record);
        }
        layer->resourceManager.removeCommandBufferRecord(pCommandBuffers[i]);

        if (!releasedSnapshots.empty())
        {
            lock.unlock();
            releasedSnapshots.clear();
            lock.lock();
        }
    }

    lock.unlock();
    layer->driver.vkFreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateCommandPool<user_tag>(
    VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool)
{
    CommandPoolRollback rollback;
    try
    {
        std::unique_lock<std::mutex> lock{g_vulkanLock};
        auto *layer = Device::retrieve(device);

        auto *instance = const_cast<Instance *>(layer->instance);

        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            return layer->driver.vkCreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
        }
        lock.unlock();
        const VkResult result =
            layer->driver.vkCreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        rollback.arm(layer, device, pAllocator, pCommandPool);
        lock.lock();

        if (!instance->isCaptureEnabled())
        {
            rollback.release();
            return VK_SUCCESS;
        }
        auto record = layer->resourceManager.addCommandPoolRecord(
            *pCommandPool, pCreateInfo->queueFamilyIndex, pCreateInfo->flags);
        if (record == nullptr)
        {
            return VK_ERROR_UNKNOWN;
        }

        try
        {
            const auto *engines =
                vku::FindStructInPNextChain<VkDataGraphProcessingEngineCreateInfoARM>(
                    pCreateInfo->pNext);
            if (engines != nullptr)
            {
                for (uint32_t i = 0; i < engines->processingEngineCount; ++i)
                {
                    if (engines->pProcessingEngines[i].isForeign == VK_TRUE)
                    {
                        record->hasForeignProcessingEngine = true;
                        LAYER_LOG("Structured capture command pool classified as using a foreign "
                                  "processing engine");
                        break;
                    }
                }
            }
        }
        catch (const std::bad_alloc &)
        {
            layer->resourceManager.removeCommandPoolRecord(*pCommandPool);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (...)
        {
            layer->resourceManager.removeCommandPoolRecord(*pCommandPool);
            return VK_ERROR_UNKNOWN;
        }

        rollback.release();
        return VK_SUCCESS;
    }
    catch (const std::bad_alloc &)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        return VK_ERROR_UNKNOWN;
    }
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkResetCommandPool<user_tag>(VkDevice device,
                                                                  VkCommandPool commandPool,
                                                                  VkCommandPoolResetFlags flags)
{
    ReleasedSnapshots releasedSnapshots;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);
    auto *instance = const_cast<Instance *>(layer->instance);
    if (!instance->isCaptureEnabled())
    {
        lock.unlock();
        return layer->driver.vkResetCommandPool(device, commandPool, flags);
    }

    lock.unlock();
    VkResult result = layer->driver.vkResetCommandPool(device, commandPool, flags);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    lock.lock();
    auto record = layer->resourceManager.getCommandPoolRecord(commandPool);
    assert(record);
    // Vulkan externally synchronizes command-pool reset, so a conforming
    // application cannot mutate this child map while the lock is temporarily
    // dropped to retire a detached snapshot batch.
    for (const auto &[_, weakCommandBuffer] : record->cbufferMap)
    {
        if (auto commandBufferRecord = weakCommandBuffer.lock())
        {
            releasedSnapshots = ReleaseSnapshots(commandBufferRecord);
            if (!releasedSnapshots.empty())
            {
                lock.unlock();
                releasedSnapshots.clear();
                lock.lock();
            }
        }
    }

    lock.unlock();
    return result;
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkDestroyCommandPool<user_tag>(
    VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator)
{
    ReleasedSnapshots releasedSnapshots;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);

    auto record = layer->resourceManager.getCommandPoolRecord(commandPool);
    if (record != nullptr)
    {
        for (const auto &[_, weakCommandBuffer] : record->cbufferMap)
        {
            if (auto commandBufferRecord = weakCommandBuffer.lock())
            {
                releasedSnapshots = ReleaseSnapshots(commandBufferRecord);
                if (!releasedSnapshots.empty())
                {
                    lock.unlock();
                    releasedSnapshots.clear();
                    lock.lock();
                }
            }
        }
    }
    layer->resourceManager.removeCommandPoolRecord(commandPool);

    lock.unlock();
    layer->driver.vkDestroyCommandPool(device, commandPool, pAllocator);
}

template <>
VKAPI_ATTR void VKAPI_CALL layer_vkCmdDispatchDataGraphARM<user_tag>(
    VkCommandBuffer commandBuffer, VkDataGraphPipelineSessionARM session,
    const VkDataGraphPipelineDispatchInfoARM *pInfo)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(commandBuffer);
    auto *instance = const_cast<Instance *>(layer->instance);
    if (!instance->isCaptureEnabled())
    {
        lock.unlock();
        layer->driver.vkCmdDispatchDataGraphARM(commandBuffer, session, pInfo);
        return;
    }
    auto commandBufferRecord = layer->resourceManager.getCommandBufferRecord(commandBuffer);
    assert(commandBufferRecord);

    auto sessionRecord = layer->resourceManager.getSessionRecord(session);
    std::shared_ptr<StatsSnapshot> snapshot;
    bool metadataRecorded = false;
    bool publishOccurrence = true;
    if (sessionRecord != nullptr)
    {
        try
        {
            capture::fault::Checkpoint(capture::fault::Point::DispatchMetadataRecording);
            CommandBufferRecord::DispatchOccurrence occurrence;
            occurrence.session = sessionRecord;
            if (!sessionRecord->statisticsCaptureActive)
            {
                occurrence.unsupportedReason =
                    !sessionRecord->unsupportedReason.empty()
                        ? sessionRecord->unsupportedReason
                        : "selected capture mechanism is unavailable for this session";
            }
            else
            {
                std::string reason;
                if (!CommandBufferSupportsSnapshots(layer, commandBufferRecord, reason))
                {
                    occurrence.unsupportedReason = std::move(reason);
                }
                else
                {
                    const VkResult result = CreateSnapshotWithoutGlobalLock(
                        lock, layer, sessionRecord, commandBufferRecord->queueFamilyIndex,
                        snapshot);
                    if (result == VK_SUCCESS)
                    {
                        if (!instance->isCaptureEnabled())
                        {
                            publishOccurrence = false;
                        }
                        else
                        {
                            occurrence.snapshot = snapshot;
                        }
                    }
                    else
                    {
                        occurrence.unsupportedReason =
                            "selected capture failed to create a host-visible snapshot destination";
                        DisableCapture(instance, occurrence.unsupportedReason.c_str());
                    }
                }
            }
            if (publishOccurrence)
            {
                capture::fault::Checkpoint(capture::fault::Point::DispatchMetadataPublication);
                commandBufferRecord->dispatches.emplace_back(std::move(occurrence));
                metadataRecorded = true;
            }
        }
        catch (const std::bad_alloc &)
        {
            DisableCapture(instance, "capture bookkeeping failed during command recording");
        }
        catch (...)
        {
            DisableCapture(instance,
                           "unexpected capture bookkeeping failure during command recording");
        }
    }

    lock.unlock();
    if (!metadataRecorded || snapshot == nullptr)
    {
        layer->driver.vkCmdDispatchDataGraphARM(commandBuffer, session, pInfo);
        return;
    }

    // Session statistics storage is shared by every command buffer that uses the
    // session. The application must synchronize cross-queue session reuse with
    // a semaphore, just as it must synchronize its own accesses to that shared
    // session state. Including DATA_GRAPH in this barrier's source scope extends
    // an application wait at the data-graph stage forward to the layer's clear;
    // the matching post-copy bridge below extends the prior copy backward to the
    // application's data-graph-stage signal.
    const VkMemoryBarrier2 priorUseToClear{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        nullptr,
        VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM | VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
    };
    const VkDependencyInfo priorUseDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &priorUseToClear, 0, nullptr, 0, nullptr};
    RecordPipelineBarrier2(layer, commandBuffer, &priorUseDependency);
    layer->driver.vkCmdFillBuffer(commandBuffer, sessionRecord->statsMemory.sourceBuffer, 0,
                                  snapshot->dataSize, 0);

    const VkMemoryBarrier2 clearToStats{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
        VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
    };
    const VkDependencyInfo clearDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &clearToStats, 0, nullptr, 0, nullptr};
    RecordPipelineBarrier2(layer, commandBuffer, &clearDependency);

    layer->driver.vkCmdDispatchDataGraphARM(commandBuffer, session, pInfo);

    const VkMemoryBarrier2 statsToTransfer{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,     nullptr,
        VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM, VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,       VK_ACCESS_2_TRANSFER_READ_BIT,
    };
    const VkDependencyInfo statsDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &statsToTransfer, 0, nullptr, 0, nullptr};
    RecordPipelineBarrier2(layer, commandBuffer, &statsDependency);

    const VkBufferCopy copyRegion{0, 0, snapshot->dataSize};
    const uint32_t copyCount = TestPostGraphCopyCount();
    const VkMemoryBarrier2 copyToCopy{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
    };
    const VkDependencyInfo copyDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &copyToCopy, 0, nullptr, 0, nullptr};
    for (uint32_t copyIndex = 0; copyIndex < copyCount; ++copyIndex)
    {
        layer->driver.vkCmdCopyBuffer(commandBuffer, sessionRecord->statsMemory.sourceBuffer,
                                      snapshot->buffer, 1, &copyRegion);
        if (copyIndex + 1 < copyCount)
        {
            RecordPipelineBarrier2(layer, commandBuffer, &copyDependency);
        }
    }

    // A pipeline barrier cannot synchronize different queues by itself. This
    // bridge instead makes the snapshot read precede the DATA_GRAPH stage, so an
    // application semaphore signal/wait scoped to that stage carries the layer's
    // copy-before-next-clear dependency across queues. Unsynchronized cross-queue
    // reuse remains an application error rather than something the layer repairs.
    const VkMemoryBarrier2 copyToSubsequentDataGraph{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,     nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,       VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM, 0,
    };
    const VkDependencyInfo subsequentDataGraphDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                         nullptr,
                                                         0,
                                                         1,
                                                         &copyToSubsequentDataGraph,
                                                         0,
                                                         nullptr,
                                                         0,
                                                         nullptr};
    RecordPipelineBarrier2(layer, commandBuffer, &subsequentDataGraphDependency);

    const VkMemoryBarrier2 copyToHost{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,   VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_HOST_BIT,       VK_ACCESS_2_HOST_READ_BIT,
    };
    const VkDependencyInfo hostDependency{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &copyToHost, 0, nullptr, 0, nullptr};
    RecordPipelineBarrier2(layer, commandBuffer, &hostDependency);
}

template <>
VKAPI_ATTR void VKAPI_CALL
layer_vkCmdExecuteCommands<user_tag>(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                     const VkCommandBuffer *pCommandBuffers)
{
    struct SecondaryExecution
    {
        VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
        std::vector<std::pair<std::shared_ptr<StatsSnapshot>, std::shared_ptr<StatsSnapshot>>>
            archiveCopies;
        std::vector<CommandBufferRecord::DispatchOccurrence> propagatedOccurrences;
    };

    static_assert(std::is_nothrow_move_constructible_v<CommandBufferRecord::DispatchOccurrence>);

    std::vector<SecondaryExecution> executions;
    std::string propagatedSnapshotError;
    bool stagingFailed = false;
    bool metadataComplete = false;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(commandBuffer);
    auto *instance = const_cast<Instance *>(layer->instance);
    if (!instance->isCaptureEnabled())
    {
        lock.unlock();
        layer->driver.vkCmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);
        return;
    }
    auto primaryRecord = layer->resourceManager.getCommandBufferRecord(commandBuffer);
    assert(primaryRecord);

    try
    {
        executions.reserve(commandBufferCount);
        for (uint32_t i = 0; i < commandBufferCount; ++i)
        {
            capture::fault::Checkpoint(
                capture::fault::Point::SecondaryExecutionMetadataPropagation);
            // Publish the staging owner before any destination snapshot is
            // created. If later staging throws, the outer vector outlives the
            // global lock and therefore retires every completed destination
            // only after the catch path unlocks it.
            executions.emplace_back();
            auto &execution = executions.back();
            execution.commandBuffer = pCommandBuffers[i];
            const auto secondary =
                layer->resourceManager.getCommandBufferRecord(pCommandBuffers[i]);
            if (secondary == nullptr)
            {
                propagatedSnapshotError =
                    "vkCmdExecuteCommands references an untracked secondary command buffer";
                continue;
            }
            if (!secondary->snapshotError.empty())
            {
                propagatedSnapshotError = secondary->snapshotError;
            }

            execution.propagatedOccurrences = secondary->dispatches;
            execution.archiveCopies.reserve(secondary->dispatches.size());
            for (auto &primaryOccurrence : execution.propagatedOccurrences)
            {
                if (primaryOccurrence.snapshot != nullptr)
                {
                    auto sourceSnapshot = primaryOccurrence.snapshot;
                    std::shared_ptr<StatsSnapshot> occurrenceSnapshot;
                    const VkResult result = CreateSnapshotWithoutGlobalLock(
                        lock, layer, primaryOccurrence.session, primaryRecord->queueFamilyIndex,
                        occurrenceSnapshot);
                    if (result == VK_SUCCESS)
                    {
                        execution.archiveCopies.emplace_back(sourceSnapshot, occurrenceSnapshot);
                        if (!instance->isCaptureEnabled())
                        {
                            stagingFailed = true;
                        }
                        else
                        {
                            primaryOccurrence.snapshot = std::move(occurrenceSnapshot);
                        }
                    }
                    else
                    {
                        primaryOccurrence.snapshot.reset();
                        primaryOccurrence.unsupportedReason =
                            "selected capture failed to create a unique secondary occurrence "
                            "snapshot";
                    }
                }
                if (stagingFailed)
                {
                    break;
                }
            }
            if (stagingFailed)
            {
                break;
            }
        }
        if (!stagingFailed)
        {
            size_t propagatedSize = primaryRecord->dispatches.size();
            for (const auto &execution : executions)
            {
                if (execution.propagatedOccurrences.size() >
                    primaryRecord->dispatches.max_size() - propagatedSize)
                {
                    throw std::length_error(
                        "secondary occurrence count exceeds command metadata capacity");
                }
                propagatedSize += execution.propagatedOccurrences.size();
            }
            primaryRecord->dispatches.reserve(propagatedSize);
            if (!propagatedSnapshotError.empty())
            {
                primaryRecord->snapshotError.swap(propagatedSnapshotError);
            }
            for (auto &execution : executions)
            {
                for (auto &occurrence : execution.propagatedOccurrences)
                {
                    primaryRecord->dispatches.emplace_back(std::move(occurrence));
                }
            }
            metadataComplete = true;
        }
    }
    catch (const std::bad_alloc &)
    {
        constexpr const char *failure = "secondary occurrence snapshot allocation failed";
        SetSnapshotErrorNoexcept(primaryRecord, failure);
        DisableCapture(instance, failure);
    }
    catch (...)
    {
        constexpr const char *failure = "secondary command metadata propagation failed";
        SetSnapshotErrorNoexcept(primaryRecord, failure);
        DisableCapture(instance, failure);
    }

    lock.unlock();
    if (!metadataComplete)
    {
        // Metadata propagation may already have populated a prefix of
        // executions. Never forward that prefix: the application supplied one
        // complete vkCmdExecuteCommands call, so an internal capture failure
        // must forward that original call exactly once and abandon archiving.
        layer->driver.vkCmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);
        return;
    }
    // Split the application array so the layer can archive each secondary
    // occurrence before the next occurrence is allowed to overwrite its
    // record-time snapshot destination.
    for (const auto &execution : executions)
    {
        layer->driver.vkCmdExecuteCommands(commandBuffer, 1, &execution.commandBuffer);
        RecordSnapshotArchiveCopies(layer, commandBuffer, execution.archiveCopies);
    }
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit<user_tag>(VkQueue queue, uint32_t submitCount,
                                                             const VkSubmitInfo *pSubmits,
                                                             VkFence fence)
{
    capture::submission::Plan plan;
    std::unique_ptr<capture::GpuCollectorJob> collectorJob;
    std::optional<capture::GpuCollector::Reservation> collectorReservation;
    std::shared_ptr<SubmissionArchive> archive;
    std::vector<SubmittedCommandBufferRange> ranges;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(queue);
    auto *instance = const_cast<Instance *>(layer->instance);
    lock.unlock();
    std::unique_lock<std::mutex> admission{instance->submissionAdmissionMutex, std::defer_lock};

    for (;;)
    {
        admission.lock();
        lock.lock();
        plan = {};
        collectorJob.reset();
        collectorReservation.reset();
        archive.reset();
        ranges.clear();
        capture::GpuCollector::CapacityState capacityState;
        bool waitForCapacity = false;

        if (!instance->isCaptureEnabled())
        {
            // Capture is already terminal: forward the original submission
            // without capture planning, rewriting, or bookkeeping.
            // Bypass tracking entirely so stale or incomplete capture metadata
            // cannot reject a valid application submission.
            lock.unlock();
            const VkResult result =
                layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }

        try
        {
            std::vector<capture::submission::SubmittedOccurrence> submitted;
            std::string reason;
            for (uint32_t i = 0; i < submitCount; ++i)
            {
                const bool protectedSubmit = capture_policy::IsLegacySubmitProtected(pSubmits[i]);
                for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; ++j)
                {
                    SubmittedCommandBufferRange range;
                    range.submitIndex = i;
                    range.commandBufferIndex = j;
                    range.firstOccurrence = submitted.size();
                    if (!capture::submission::appendCommandBufferOccurrences(
                            layer->resourceManager.getCommandBufferRecord(
                                pSubmits[i].pCommandBuffers[j]),
                            protectedSubmit, submitted, reason))
                    {
                        DisableCapture(instance, reason.c_str());
                        capture::submission::cancel(plan);
                        lock.unlock();
                        const VkResult result =
                            layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    range.occurrenceCount = submitted.size() - range.firstOccurrence;
                    ranges.emplace_back(range);
                }
            }
            const auto buildResult =
                capture::submission::build(instance->dispatchFilter, instance->isCaptureEnabled(),
                                           instance->dispatchIds, submitted, plan, reason);
            if (buildResult == capture::submission::BuildResult::Rejected)
            {
                DisableCapture(instance, reason.c_str());
                capture::submission::cancel(plan);
                lock.unlock();
                const VkResult result =
                    layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            if (plan.selectedCount != 0)
            {
                for (uint32_t i = 0; i < submitCount; ++i)
                {
                    const auto *deviceGroup =
                        vku::FindStructInPNextChain<VkDeviceGroupSubmitInfo>(pSubmits[i].pNext);
                    if (deviceGroup != nullptr &&
                        (deviceGroup->commandBufferCount != pSubmits[i].commandBufferCount ||
                         (deviceGroup->commandBufferCount != 0 &&
                          deviceGroup->pCommandBufferDeviceMasks == nullptr)))
                    {
                        DisableCapture(instance, "selected capture found malformed legacy "
                                                 "device-group submit command masks");
                        return VK_ERROR_VALIDATION_FAILED_EXT;
                    }
                }
                collectorReservation = layer->collector->tryReserve(capacityState);
                if (!collectorReservation.has_value())
                {
                    if (!capacityState.accepting)
                    {
                        DisableCapture(instance, "GPU collector admission is closed");
                        capture::submission::cancel(plan);
                        lock.unlock();
                        const VkResult result =
                            layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    waitForCapacity = true;
                }
                else
                {
                    const VkResult materialized = MaterializeSubmissionArchiveWithoutGlobalLock(
                        lock, layer, ranges, plan, archive, reason);
                    // Materialization drops g_vulkanLock, so capture may have
                    // terminalized while capture-only Vulkan resources were
                    // being staged. Retire those resources outside the lock
                    // and forward the untouched application submission before
                    // performing any further capture-only allocation.
                    if (!instance->isCaptureEnabled())
                    {
                        capture::submission::cancel(plan);
                        collectorReservation.reset();
                        lock.unlock();
                        const VkResult result =
                            layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    if (materialized != VK_SUCCESS)
                    {
                        capture::submission::cancel(plan);
                        collectorReservation.reset();
                        if (materialized != VK_ERROR_OUT_OF_HOST_MEMORY &&
                            materialized != VK_ERROR_OUT_OF_DEVICE_MEMORY)
                        {
                            DisableCapture(instance, reason.c_str());
                        }
                        return materialized;
                    }
                    const uint32_t statisticsMode = instance->layerOptions->getStatsModeIndex();
                    collectorJob = PrepareCollectorJob(plan, archive, statisticsMode);
                }
            }
        }
        catch (const std::bad_alloc &)
        {
            auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
            capture::submission::cancel(abandonedPlan);
            if (!instance->isCaptureEnabled())
            {
                collectorReservation.reset();
                lock.unlock();
                const VkResult result =
                    layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (...)
        {
            auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
            capture::submission::cancel(abandonedPlan);
            if (!instance->isCaptureEnabled())
            {
                collectorReservation.reset();
                lock.unlock();
                const VkResult result =
                    layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            return VK_ERROR_UNKNOWN;
        }

        if (!waitForCapacity)
        {
            break;
        }
        lock.unlock();
        admission.unlock();
        (void)layer->collector->waitForCapacityChange(capacityState);
    }

    std::vector<vku::safe_VkSubmitInfo> safeSubmits;
    std::vector<VkSubmitInfo> rewrittenSubmits;
    const VkSubmitInfo *forwardedSubmits = pSubmits;
    try
    {
        if (collectorJob != nullptr)
        {
            capture::fault::Checkpoint(capture::fault::Point::LegacySubmitRewrite);
            safeSubmits.reserve(submitCount);
            rewrittenSubmits.reserve(submitCount);
            size_t rangeIndex = 0;
            for (uint32_t i = 0; i < submitCount; ++i)
            {
                safeSubmits.emplace_back(&pSubmits[i]);
                auto &safeSubmit = safeSubmits.back();
                const auto *applicationDeviceGroup =
                    vku::FindStructInPNextChain<VkDeviceGroupSubmitInfo>(pSubmits[i].pNext);
                auto *copiedDeviceGroup = const_cast<VkDeviceGroupSubmitInfo *>(
                    vku::FindStructInPNextChain<VkDeviceGroupSubmitInfo>(safeSubmit.pNext));

                std::vector<VkCommandBuffer> commands;
                std::vector<uint32_t> commandBufferDeviceMasks;
                commands.reserve(pSubmits[i].commandBufferCount * 2);
                if (applicationDeviceGroup != nullptr)
                {
                    commandBufferDeviceMasks.reserve(pSubmits[i].commandBufferCount * 2);
                }
                for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; ++j)
                {
                    commands.emplace_back(pSubmits[i].pCommandBuffers[j]);
                    if (applicationDeviceGroup != nullptr)
                    {
                        commandBufferDeviceMasks.emplace_back(
                            applicationDeviceGroup->pCommandBufferDeviceMasks[j]);
                    }
                    const auto &range = ranges[rangeIndex++];
                    if (range.archiveCommandBuffer != VK_NULL_HANDLE)
                    {
                        commands.emplace_back(range.archiveCommandBuffer);
                        if (applicationDeviceGroup != nullptr)
                        {
                            commandBufferDeviceMasks.emplace_back(
                                applicationDeviceGroup->pCommandBufferDeviceMasks[j]);
                        }
                    }
                }

                auto *copiedCommands = new VkCommandBuffer[commands.size()];
                std::copy(commands.begin(), commands.end(), copiedCommands);
                delete[] safeSubmit.pCommandBuffers;
                safeSubmit.commandBufferCount = static_cast<uint32_t>(commands.size());
                safeSubmit.pCommandBuffers = copiedCommands;

                if (applicationDeviceGroup != nullptr)
                {
                    auto *safeDeviceGroup =
                        reinterpret_cast<vku::safe_VkDeviceGroupSubmitInfo *>(copiedDeviceGroup);
                    auto *copiedMasks = new uint32_t[commandBufferDeviceMasks.size()];
                    std::copy(commandBufferDeviceMasks.begin(), commandBufferDeviceMasks.end(),
                              copiedMasks);
                    delete[] safeDeviceGroup->pCommandBufferDeviceMasks;
                    safeDeviceGroup->commandBufferCount =
                        static_cast<uint32_t>(commandBufferDeviceMasks.size());
                    safeDeviceGroup->pCommandBufferDeviceMasks = copiedMasks;
                }

                rewrittenSubmits.emplace_back(*safeSubmit.ptr());
            }
            forwardedSubmits = rewrittenSubmits.data();
        }
    }
    catch (const std::bad_alloc &)
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            const VkResult result =
                layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            const VkResult result =
                layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }
        return VK_ERROR_UNKNOWN;
    }

    if (!instance->isCaptureEnabled())
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        lock.unlock();
        const VkResult result = layer->driver.vkQueueSubmit(queue, submitCount, pSubmits, fence);
        admission.unlock();
        return result;
    }

    lock.unlock();
    const VkResult result =
        layer->driver.vkQueueSubmit(queue, submitCount, forwardedSubmits, fence);
    lock.lock();
    auto &acceptedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
    if (result != VK_SUCCESS)
    {
        capture::submission::cancel(acceptedPlan);
        collectorReservation.reset();
        return result;
    }
    if (!AcceptAndPublishSubmissionPlan(instance, acceptedPlan))
    {
        // The rewritten submission may contain archive command buffers and
        // snapshot destinations that the GPU still references. Retain them
        // until device teardown, but cancel the unaccepted plan so a terminal
        // transition during the downstream call commits no capture identity,
        // session index, or writer publication.
        if (collectorJob != nullptr)
        {
            DeferAcceptedSubmissionResources(layer, *collectorJob);
        }
        capture::submission::cancel(acceptedPlan);
        collectorReservation.reset();
        return result;
    }
    if (collectorJob == nullptr)
    {
        capture::submission::finish(acceptedPlan, nullptr);
        return result;
    }
    if (!instance->isCaptureEnabled())
    {
        DeferAcceptedSubmissionResources(layer, *collectorJob);
        capture::submission::finish(acceptedPlan,
                                    "capture disabled during accepted dispatch publication");
        collectorReservation.reset();
        return result;
    }

    lock.unlock();
    const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    VkFence markerFence = VK_NULL_HANDLE;
    VkResult markerResult =
        layer->driver.vkCreateFence(layer->device, &fenceInfo, nullptr, &markerFence);
    if (markerResult == VK_SUCCESS)
    {
        markerResult = SubmitLegacyMarker(layer, queue, markerFence);
    }
    if (markerResult != VK_SUCCESS)
    {
        if (markerFence != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyFence(layer->device, markerFence, nullptr);
        }
        lock.lock();
        FinishPostForwardCaptureFailure(
            layer, instance, collectorJob.get(), acceptedPlan,
            "capture disabled after post-forward marker submission failure",
            "post-forward marker submission failed; preserving the accepted application result");
        collectorReservation.reset();
        return result;
    }

    collectorJob->fence = markerFence;
    const bool admitted = collectorReservation->commit(std::move(collectorJob));
    collectorReservation.reset();
    admission.unlock();
    if (!admitted)
    {
        DisableCapture(instance, "GPU collector admission failed after marker submission; "
                                 "preserving the accepted application result");
    }
    return result;
}

static VkResult QueueSubmit2Common(capture_policy::Submit2Route route, VkQueue queue,
                                   uint32_t submitCount, const VkSubmitInfo2 *pSubmits,
                                   VkFence fence)
{
    capture::submission::Plan plan;
    std::unique_ptr<capture::GpuCollectorJob> collectorJob;
    std::optional<capture::GpuCollector::Reservation> collectorReservation;
    std::shared_ptr<SubmissionArchive> archive;
    std::vector<SubmittedCommandBufferRange> ranges;
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(queue);
    auto *instance = const_cast<Instance *>(layer->instance);
    const auto submitFunction = capture_policy::SelectQueueSubmit2Function(
        route, layer->driver.vkQueueSubmit2, layer->driver.vkQueueSubmit2KHR);
    if (submitFunction == nullptr)
    {
        return VK_ERROR_UNKNOWN;
    }
    lock.unlock();
    std::unique_lock<std::mutex> admission{instance->submissionAdmissionMutex, std::defer_lock};

    for (;;)
    {
        admission.lock();
        lock.lock();
        plan = {};
        collectorJob.reset();
        collectorReservation.reset();
        archive.reset();
        ranges.clear();
        capture::GpuCollector::CapacityState capacityState;
        bool waitForCapacity = false;

        if (!instance->isCaptureEnabled())
        {
            // As with legacy submission, an earlier capture failure must not
            // reject or rewrite later application work.
            lock.unlock();
            const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }

        try
        {
            std::vector<capture::submission::SubmittedOccurrence> submitted;
            std::string reason;
            for (uint32_t i = 0; i < submitCount; ++i)
            {
                const bool protectedSubmit = capture_policy::IsSubmit2Protected(pSubmits[i]);
                for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; ++j)
                {
                    SubmittedCommandBufferRange range;
                    range.submitIndex = i;
                    range.commandBufferIndex = j;
                    range.firstOccurrence = submitted.size();
                    if (!capture::submission::appendCommandBufferOccurrences(
                            layer->resourceManager.getCommandBufferRecord(
                                pSubmits[i].pCommandBufferInfos[j].commandBuffer),
                            protectedSubmit, submitted, reason))
                    {
                        DisableCapture(instance, reason.c_str());
                        capture::submission::cancel(plan);
                        lock.unlock();
                        const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    range.occurrenceCount = submitted.size() - range.firstOccurrence;
                    ranges.emplace_back(range);
                }
            }
            const auto buildResult =
                capture::submission::build(instance->dispatchFilter, instance->isCaptureEnabled(),
                                           instance->dispatchIds, submitted, plan, reason);
            if (buildResult == capture::submission::BuildResult::Rejected)
            {
                DisableCapture(instance, reason.c_str());
                capture::submission::cancel(plan);
                lock.unlock();
                const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            if (plan.selectedCount != 0)
            {
                collectorReservation = layer->collector->tryReserve(capacityState);
                if (!collectorReservation.has_value())
                {
                    if (!capacityState.accepting)
                    {
                        DisableCapture(instance, "GPU collector admission is closed");
                        capture::submission::cancel(plan);
                        lock.unlock();
                        const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    waitForCapacity = true;
                }
                else
                {
                    const VkResult materialized = MaterializeSubmissionArchiveWithoutGlobalLock(
                        lock, layer, ranges, plan, archive, reason);
                    // See the legacy path: if capture became terminal while the
                    // global lock was released, forward the original submission
                    // even if capture-only materialization reported failure.
                    // Recheck before interpreting that result or allocating the
                    // collector job and rewritten submit arrays.
                    if (!instance->isCaptureEnabled())
                    {
                        capture::submission::cancel(plan);
                        collectorReservation.reset();
                        lock.unlock();
                        const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                        admission.unlock();
                        return result;
                    }
                    if (materialized != VK_SUCCESS)
                    {
                        capture::submission::cancel(plan);
                        collectorReservation.reset();
                        if (materialized != VK_ERROR_OUT_OF_HOST_MEMORY &&
                            materialized != VK_ERROR_OUT_OF_DEVICE_MEMORY)
                        {
                            DisableCapture(instance, reason.c_str());
                        }
                        return materialized;
                    }
                    const uint32_t statisticsMode = instance->layerOptions->getStatsModeIndex();
                    collectorJob = PrepareCollectorJob(plan, archive, statisticsMode);
                }
            }
        }
        catch (const std::bad_alloc &)
        {
            auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
            capture::submission::cancel(abandonedPlan);
            if (!instance->isCaptureEnabled())
            {
                collectorReservation.reset();
                lock.unlock();
                const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (...)
        {
            auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
            capture::submission::cancel(abandonedPlan);
            if (!instance->isCaptureEnabled())
            {
                collectorReservation.reset();
                lock.unlock();
                const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
                admission.unlock();
                return result;
            }
            return VK_ERROR_UNKNOWN;
        }

        if (!waitForCapacity)
        {
            break;
        }
        lock.unlock();
        admission.unlock();
        (void)layer->collector->waitForCapacityChange(capacityState);
    }

    std::vector<VkSubmitInfo2> rewrittenSubmits;
    std::vector<std::vector<VkCommandBufferSubmitInfo>> rewrittenCommandBuffers;
    const VkSubmitInfo2 *forwardedSubmits = pSubmits;
    try
    {
        if (collectorJob != nullptr)
        {
            capture::fault::Checkpoint(capture::fault::Point::Submit2Rewrite);
            rewrittenSubmits.assign(pSubmits, pSubmits + submitCount);
            rewrittenCommandBuffers.resize(submitCount);
            size_t rangeIndex = 0;
            for (uint32_t i = 0; i < submitCount; ++i)
            {
                auto &commands = rewrittenCommandBuffers[i];
                commands.reserve(pSubmits[i].commandBufferInfoCount * 2);
                for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; ++j)
                {
                    commands.emplace_back(pSubmits[i].pCommandBufferInfos[j]);
                    const auto &range = ranges[rangeIndex++];
                    if (range.archiveCommandBuffer != VK_NULL_HANDLE)
                    {
                        commands.push_back({
                            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                            nullptr,
                            range.archiveCommandBuffer,
                            pSubmits[i].pCommandBufferInfos[j].deviceMask,
                        });
                    }
                }
                rewrittenSubmits[i].commandBufferInfoCount = static_cast<uint32_t>(commands.size());
                rewrittenSubmits[i].pCommandBufferInfos = commands.data();
            }
            forwardedSubmits = rewrittenSubmits.data();
        }
    }
    catch (const std::bad_alloc &)
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    catch (...)
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        if (!instance->isCaptureEnabled())
        {
            lock.unlock();
            const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
            admission.unlock();
            return result;
        }
        return VK_ERROR_UNKNOWN;
    }

    if (!instance->isCaptureEnabled())
    {
        auto &abandonedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
        capture::submission::cancel(abandonedPlan);
        collectorReservation.reset();
        lock.unlock();
        const VkResult result = submitFunction(queue, submitCount, pSubmits, fence);
        admission.unlock();
        return result;
    }

    lock.unlock();
    const VkResult result = submitFunction(queue, submitCount, forwardedSubmits, fence);
    lock.lock();
    auto &acceptedPlan = collectorJob != nullptr ? collectorJob->plan : plan;
    if (result != VK_SUCCESS)
    {
        capture::submission::cancel(acceptedPlan);
        collectorReservation.reset();
        return result;
    }
    if (!AcceptAndPublishSubmissionPlan(instance, acceptedPlan))
    {
        // See the legacy path: downstream success makes the rewritten archive
        // resources GPU-visible, even though the capture transaction must be
        // abandoned without committing any capture bookkeeping.
        if (collectorJob != nullptr)
        {
            DeferAcceptedSubmissionResources(layer, *collectorJob);
        }
        capture::submission::cancel(acceptedPlan);
        collectorReservation.reset();
        return result;
    }
    if (collectorJob == nullptr)
    {
        capture::submission::finish(acceptedPlan, nullptr);
        return result;
    }
    if (!instance->isCaptureEnabled())
    {
        DeferAcceptedSubmissionResources(layer, *collectorJob);
        capture::submission::finish(acceptedPlan,
                                    "capture disabled during accepted dispatch publication");
        collectorReservation.reset();
        return result;
    }

    lock.unlock();
    const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    VkFence markerFence = VK_NULL_HANDLE;
    VkResult markerResult =
        layer->driver.vkCreateFence(layer->device, &fenceInfo, nullptr, &markerFence);
    if (markerResult == VK_SUCCESS)
    {
        markerResult = Submit2Marker(layer, route, queue, markerFence);
    }
    if (markerResult != VK_SUCCESS)
    {
        if (markerFence != VK_NULL_HANDLE)
        {
            layer->driver.vkDestroyFence(layer->device, markerFence, nullptr);
        }
        lock.lock();
        FinishPostForwardCaptureFailure(
            layer, instance, collectorJob.get(), acceptedPlan,
            "capture disabled after post-forward submit2 marker submission failure",
            "post-forward submit2 marker submission failed; preserving the accepted application "
            "result");
        collectorReservation.reset();
        return result;
    }

    collectorJob->fence = markerFence;
    const bool admitted = collectorReservation->commit(std::move(collectorJob));
    collectorReservation.reset();
    admission.unlock();
    if (!admitted)
    {
        DisableCapture(instance, "GPU collector admission failed after submit2 marker submission; "
                                 "preserving the accepted application result");
    }
    return result;
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit2<user_tag>(VkQueue queue, uint32_t submitCount,
                                                              const VkSubmitInfo2 *pSubmits,
                                                              VkFence fence)
{
    return QueueSubmit2Common(capture_policy::Submit2Route::Core, queue, submitCount, pSubmits,
                              fence);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueueSubmit2KHR<user_tag>(VkQueue queue,
                                                                 uint32_t submitCount,
                                                                 const VkSubmitInfo2 *pSubmits,
                                                                 VkFence fence)
{
    return QueueSubmit2Common(capture_policy::Submit2Route::Khr, queue, submitCount, pSubmits,
                              fence);
}

template <>
VKAPI_ATTR VkResult VKAPI_CALL layer_vkSetDebugUtilsObjectNameEXT<user_tag>(
    VkDevice device, const VkDebugUtilsObjectNameInfoEXT *pNameInfo)
{
    std::unique_lock<std::mutex> lock{g_vulkanLock};
    auto *layer = Device::retrieve(device);
    auto *instance = const_cast<Instance *>(layer->instance);
    lock.unlock();
    const VkResult result = layer->driver.vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    lock.lock();
    try
    {
        const std::string name = pNameInfo->pObjectName == nullptr ? "" : pNameInfo->pObjectName;
        switch (pNameInfo->objectType)
        {
        case VK_OBJECT_TYPE_PIPELINE:
        {
            const auto handle = reinterpret_cast<VkPipeline>(pNameInfo->objectHandle);
            const auto record = layer->resourceManager.getPipelineRecord(handle);
            if (record != nullptr)
            {
                record->setDebugName(name);
                (void)SubmitWriterJob(
                    instance, [&](capture::CaptureWriter &writer)
                    { return writer.updatePipelineFriendlyName(record->id, name); },
                    "pipeline debug-name writer job was rejected");
            }
            break;
        }
        case VK_OBJECT_TYPE_DATA_GRAPH_PIPELINE_SESSION_ARM:
        {
            const auto handle =
                reinterpret_cast<VkDataGraphPipelineSessionARM>(pNameInfo->objectHandle);
            const auto record = layer->resourceManager.getSessionRecord(handle);
            if (record != nullptr)
            {
                record->setDebugName(name);
                (void)SubmitWriterJob(
                    instance, [&](capture::CaptureWriter &writer)
                    { return writer.updateSessionFriendlyName(record->id, name); },
                    "session debug-name writer job was rejected");
            }
            break;
        }
        case VK_OBJECT_TYPE_SHADER_MODULE:
        {
            const auto handle = reinterpret_cast<VkShaderModule>(pNameInfo->objectHandle);
            const auto record = layer->resourceManager.getShaderModuleRecord(handle);
            if (record != nullptr)
            {
                record->setDebugName(name);
                for (const auto &pipeline : layer->resourceManager.pipelinesUsingShader(record))
                {
                    pipeline->metadata.shader.friendlyName = name;
                    (void)SubmitWriterJob(
                        instance, [&](capture::CaptureWriter &writer)
                        { return writer.updatePipelineMetadata(pipeline->id, pipeline->metadata); },
                        "shader debug-name metadata writer job was rejected");
                }
            }
            break;
        }
        default:
            break;
        }
    }
    catch (...)
    {
        LAYER_ERR("debug-name capture bookkeeping failed after the downstream name update");
    }
    return result;
}
