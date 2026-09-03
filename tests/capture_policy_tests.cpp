/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_policy.hpp"
#include "resource_tracking.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
int failures = 0;
int coreSubmitCalls = 0;
int khrSubmitCalls = 0;
int coreBarrierCalls = 0;
int khrBarrierCalls = 0;

void Check(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Handle> Handle FakeHandle(uintptr_t value)
{
    return reinterpret_cast<Handle>(value);
}

VKAPI_ATTR VkResult VKAPI_CALL CoreSubmit(VkQueue, uint32_t, const VkSubmitInfo2 *, VkFence)
{
    ++coreSubmitCalls;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL KhrSubmit(VkQueue, uint32_t, const VkSubmitInfo2 *, VkFence)
{
    ++khrSubmitCalls;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL CoreBarrier(VkCommandBuffer, const VkDependencyInfo *)
{
    ++coreBarrierCalls;
}

VKAPI_ATTR void VKAPI_CALL KhrBarrier(VkCommandBuffer, const VkDependencyInfo *)
{
    ++khrBarrierCalls;
}
} // namespace

int main()
{
    using namespace capture_policy;

    Check(ClassifyDeferredPipelineOperation(VK_NULL_HANDLE) == DeferredPipelineDecision::Capture,
          "null deferred operation is capturable");
    Check(ClassifyDeferredPipelineOperation(reinterpret_cast<VkDeferredOperationKHR>(
              uintptr_t{1})) == DeferredPipelineDecision::Reject,
          "non-null deferred operation is rejected");

    Check(
        ClassifyCommandBuffer(false, false, VK_QUEUE_DATA_GRAPH_BIT_ARM | VK_QUEUE_TRANSFER_BIT) ==
            CommandBufferDecision::Capture,
        "ordinary transfer-capable data-graph command buffer is accepted");
    Check(ClassifyCommandBuffer(true, false, VK_QUEUE_TRANSFER_BIT) ==
              CommandBufferDecision::RejectProtected,
          "protected command buffer is rejected");
    Check(ClassifyCommandBuffer(false, true, VK_QUEUE_TRANSFER_BIT) ==
              CommandBufferDecision::RejectForeignProcessingEngine,
          "foreign processing engine is rejected");
    Check(ClassifyCommandBuffer(false, false, VK_QUEUE_DATA_GRAPH_BIT_ARM) ==
              CommandBufferDecision::RejectMissingTransferCapability,
          "queue family without copy capability is rejected");

    VkProtectedSubmitInfo protectedInfo{
        VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
        nullptr,
        VK_TRUE,
    };
    VkSubmitInfo legacySubmit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, &protectedInfo, 0, nullptr, nullptr, 0, nullptr, 0, nullptr,
    };
    Check(IsLegacySubmitProtected(legacySubmit),
          "legacy protected submit is detected through pNext");
    protectedInfo.protectedSubmit = VK_FALSE;
    Check(!IsLegacySubmitProtected(legacySubmit), "legacy non-protected submit is accepted");

    VkSubmitInfo2 submit2{
        VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        nullptr,
        VK_SUBMIT_PROTECTED_BIT,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
    };
    Check(IsSubmit2Protected(submit2), "submit2 protected flag is detected");
    submit2.flags = 0;
    Check(!IsSubmit2Protected(submit2), "submit2 without protected flag is accepted");
    Check(RejectProtectedSelectedCapture(true, true),
          "selected capture in a protected submission is rejected");
    Check(!RejectProtectedSelectedCapture(false, true),
          "protected submission without selected capture remains transparent");
    Check(!RejectProtectedSelectedCapture(true, false),
          "unprotected selected capture remains accepted");

    DeviceStatsManager manager;
    const VkCommandPool protectedPoolHandle = FakeHandle<VkCommandPool>(uintptr_t{0x1000});
    const VkCommandBuffer protectedCommandBufferHandle =
        FakeHandle<VkCommandBuffer>(uintptr_t{0x2000});
    const auto protectedPool =
        manager.addCommandPoolRecord(protectedPoolHandle, 7, VK_COMMAND_POOL_CREATE_PROTECTED_BIT);
    const auto protectedCommandBuffer =
        manager.addCommandBufferRecord(protectedCommandBufferHandle, protectedPoolHandle);
    Check(protectedPool != nullptr && protectedPool->isProtected,
          "protected command-pool flag is tracked");
    Check(protectedCommandBuffer != nullptr && protectedCommandBuffer->isProtected,
          "protected command-pool state propagates to allocated command buffers");
    Check(protectedCommandBuffer != nullptr && protectedCommandBuffer->queueFamilyIndex == 7,
          "command-pool queue family propagates to allocated command buffers");

    const VkShaderModule shaderHandle = FakeHandle<VkShaderModule>(uintptr_t{0x3000});
    std::vector<uint8_t> originalSpirv{3, 2, 35, 7};
    const auto shader0 =
        manager.addShaderModuleRecord(shaderHandle, capture::ShaderModuleId(0), originalSpirv);
    originalSpirv[0] = 99;
    Check(shader0 != nullptr && shader0->id == capture::ShaderModuleId(0) &&
              shader0->spirv == std::vector<uint8_t>({3, 2, 35, 7}),
          "shader tracking owns a deep SPIR-V copy with a stable typed ID");

    capture::PipelineMetadata pipelineMetadata;
    pipelineMetadata.shader.moduleId = shader0->id;
    pipelineMetadata.shader.spirvAvailable = true;
    const VkPipeline pipelineHandle = FakeHandle<VkPipeline>(uintptr_t{0x4000});
    const auto pipeline0 = manager.addPipelineRecord(pipelineHandle, capture::PipelineId(0),
                                                     pipelineMetadata, shader0);
    manager.removeShaderModuleRecord(shaderHandle);
    Check(pipeline0 != nullptr && pipeline0->shader == shader0 &&
              pipeline0->shader->spirv.size() == 4,
          "pipeline ownership keeps shader bytes alive after shader-handle destruction");

    const auto shader1 =
        manager.addShaderModuleRecord(shaderHandle, capture::ShaderModuleId(1), {});
    Check(shader1 != nullptr && shader1->id == capture::ShaderModuleId(1) && shader1 != shader0 &&
              shader1->spirv.empty(),
          "destroy/recreate handle reuse receives a new shader ID and supports missing SPIR-V");

    const VkDataGraphPipelineSessionARM sessionHandle =
        FakeHandle<VkDataGraphPipelineSessionARM>(uintptr_t{0x5000});
    const auto session0 =
        manager.addSessionRecord(sessionHandle, capture::SessionId(0), pipeline0, {}, {});
    manager.removeSessionRecord(sessionHandle);
    const auto session1 =
        manager.addSessionRecord(sessionHandle, capture::SessionId(1), pipeline0, {}, {});
    Check(session0 != nullptr && session1 != nullptr && session0->id == capture::SessionId(0) &&
              session1->id == capture::SessionId(1) && session0 != session1,
          "session handle reuse preserves independent stable IDs");

    auto coreSubmit = SelectQueueSubmit2Function(Submit2Route::Core, CoreSubmit, KhrSubmit);
    auto khrSubmit = SelectQueueSubmit2Function(Submit2Route::Khr, CoreSubmit, KhrSubmit);
    Check(SelectQueueSubmit2Function(Submit2Route::Core, nullptr, KhrSubmit) == nullptr,
          "missing core submit2 route is reported rather than silently using KHR");
    Check(SelectQueueSubmit2Function(Submit2Route::Khr, CoreSubmit, nullptr) == nullptr,
          "missing KHR submit2 route is reported rather than silently using core");
    coreSubmit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE);
    khrSubmit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE);
    Check(coreSubmitCalls == 1 && khrSubmitCalls == 1,
          "core and KHR submit2 routes call their matching downstream entrypoints");

    Check(SelectSynchronization2Route(CoreBarrier, KhrBarrier) == Synchronization2Route::Core,
          "core synchronization2 route is preferred when available");
    Check(SelectSynchronization2Route(nullptr, KhrBarrier) == Synchronization2Route::Khr,
          "KHR synchronization2 route is selected when core is unavailable");
    Check(SelectSynchronization2Route(nullptr, nullptr) == Synchronization2Route::Unavailable,
          "missing synchronization2 commands are detected");

    auto coreBarrier =
        SelectCmdPipelineBarrier2Function(Synchronization2Route::Core, CoreBarrier, KhrBarrier);
    auto khrBarrier =
        SelectCmdPipelineBarrier2Function(Synchronization2Route::Khr, CoreBarrier, KhrBarrier);
    coreBarrier(VK_NULL_HANDLE, nullptr);
    khrBarrier(VK_NULL_HANDLE, nullptr);
    Check(coreBarrierCalls == 1 && khrBarrierCalls == 1,
          "core and KHR barrier routes call their matching downstream entrypoints");

    if (failures != 0)
    {
        std::cerr << failures << " capture policy test(s) failed\n";
        return 1;
    }
    std::cout << "All capture policy tests passed\n";
    return 0;
}
