/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_model.hpp"
#include "capture_model_harness.hpp"
#include "fault_injection.hpp"
#include "instance_creation_transaction.hpp"
#include "vulkan_exception_policy.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#define CaptureModel CaptureModelHarness

namespace
{
int failures = 0;
int publishCalls = 0;
int trackedDestroyCalls = 0;
int untrackedDestroyCalls = 0;

void Check(bool condition, std::string_view message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function> bool Throws(Function &&function)
{
    try
    {
        function();
    }
    catch (const Exception &)
    {
        return true;
    }
    catch (...)
    {
    }
    return false;
}

capture::CaptureMetadata Metadata()
{
    capture::CaptureMetadata metadata;
    metadata.devices.push_back(capture::DeviceMetadata{capture::LogicalDeviceId(0),
                                                       "exception-test-device", std::nullopt,
                                                       std::nullopt, std::nullopt, std::nullopt});
    return metadata;
}

VkInstance FakeInstance()
{
    return reinterpret_cast<VkInstance>(uintptr_t{0x12340000});
}

VkInstance FakeProvisionalInstance()
{
    return reinterpret_cast<VkInstance>(uintptr_t{0x56780000});
}

void FakePublish(VkInstance, LayerOptionsPtr &&)
{
    ++publishCalls;
}

void FakeDestroyTracked(VkInstance, const VkAllocationCallbacks *,
                        PFN_vkGetInstanceProcAddr) noexcept
{
    ++trackedDestroyCalls;
}

VKAPI_ATTR void VKAPI_CALL FakeDownstreamDestroy(VkInstance, const VkAllocationCallbacks *)
{
    ++untrackedDestroyCalls;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FakeNextGetInstanceProcAddr(VkInstance, const char *)
{
    return reinterpret_cast<PFN_vkVoidFunction>(FakeDownstreamDestroy);
}

constexpr InstancePublicationOperations kFakeOperations{
    FakePublish,
    FakeDestroyTracked,
};

void ResetInstanceCounters()
{
    publishCalls = 0;
    trackedDestroyCalls = 0;
    untrackedDestroyCalls = 0;
}

void TestTrackedInstancePublicationRollback()
{
    ResetInstanceCounters();
    VkInstance instance = VK_NULL_HANDLE;
    auto options = std::make_unique<LayerOptions>();
    capture::fault::ScopedInjection injection(
        capture::fault::Point::InstanceAfterDownstreamCreateBeforePublication);
    const VkResult result =
        PublishCreatedInstanceTransaction(FakeInstance(), nullptr, &instance, VK_NULL_HANDLE,
                                          nullptr, std::move(options), kFakeOperations);

    Check(result == VK_ERROR_OUT_OF_HOST_MEMORY,
          "post-create allocation failure translates to VK_ERROR_OUT_OF_HOST_MEMORY");
    Check(publishCalls == 0,
          "publication fault occurs after the tracked create and before state publication");
    Check(trackedDestroyCalls == 1, "post-create publication failure removes tracked state and "
                                    "destroys the downstream instance");
    Check(instance == VK_NULL_HANDLE, "rolled-back instance is not published to the application");
}

void TestUntrackedDownstreamRollback()
{
    ResetInstanceCounters();
    DestroyUntrackedCreatedInstance(FakeInstance(), nullptr, FakeNextGetInstanceProcAddr);
    Check(untrackedDestroyCalls == 1, "a downstream handle created before framework tracking is "
                                      "destroyed through the next layer");
}

void TestUnexpectedPublicationRollback()
{
    ResetInstanceCounters();
    VkInstance instance = FakeProvisionalInstance();
    auto options = std::make_unique<LayerOptions>();
    capture::fault::ScopedInjection injection(
        capture::fault::Point::InstanceAfterDownstreamCreateBeforePublication,
        capture::fault::Failure::Unexpected);
    const VkResult result = PublishCreatedInstanceTransaction(FakeInstance(), nullptr, &instance,
                                                              FakeProvisionalInstance(), nullptr,
                                                              std::move(options), kFakeOperations);

    Check(result == VK_ERROR_INITIALIZATION_FAILED,
          "unexpected publication failure translates to VK_ERROR_INITIALIZATION_FAILED");
    Check(trackedDestroyCalls == 1 && instance == FakeProvisionalInstance(),
          "unexpected post-create failure rolls back and restores loader provisional output state");
}

void TestSuccessfulInstancePublication()
{
    ResetInstanceCounters();
    VkInstance instance = VK_NULL_HANDLE;
    auto options = std::make_unique<LayerOptions>();
    const VkResult result =
        PublishCreatedInstanceTransaction(FakeInstance(), nullptr, &instance, VK_NULL_HANDLE,
                                          nullptr, std::move(options), kFakeOperations);

    Check(result == VK_SUCCESS && instance == FakeInstance(),
          "successful publication returns the created instance");
    Check(publishCalls == 1 && trackedDestroyCalls == 0,
          "successful publication performs no rollback");
}

void CheckTerminalAllocationFailure(const capture::CaptureModel &model, std::string_view context)
{
    Check(model.status() == capture::CaptureStatus::Error,
          std::string(context) + ": status is error");
    Check(model.terminalFailure() == capture::CaptureTerminalFailure::AllocationFailure,
          std::string(context) + ": allocation failure is recorded");
    Check(!model.errorMessage().empty(), std::string(context) + ": terminal error is serializable");
    Check(model.validateInvariants(), std::string(context) + ": invariants remain valid");
}

void TestPipelineAndSessionTrackingRollback()
{
    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        capture::fault::ScopedInjection injection(
            capture::fault::Point::TrackPipelineAfterInsertion);
        Check(Throws<std::bad_alloc>([&] { (void)model.trackPipeline(); }),
              "pipeline publication fault is observable as allocation failure");
        CheckTerminalAllocationFailure(model, "pipeline tracking rollback");
        Check(Throws<std::out_of_range>([&] { (void)model.pipelinePath(capture::PipelineId(0)); }),
              "failed pipeline ID is not published");
        Check(Throws<capture::CaptureTerminalError>([&] { (void)model.trackPipeline(); }),
              "terminal model rejects future pipeline tracking");
    }

    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        capture::fault::ScopedInjection injection(
            capture::fault::Point::TrackSessionAfterInsertion);
        Check(Throws<std::bad_alloc>([&] { (void)model.trackSession(pipeline); }),
              "session publication fault is observable as allocation failure");
        CheckTerminalAllocationFailure(model, "session tracking rollback");
        Check(Throws<std::out_of_range>([&] { (void)model.sessionPath(capture::SessionId(0)); }),
              "failed session ID is not published");
    }
}

void TestHierarchyMaterializationRollback()
{
    for (const auto point : {
             capture::fault::Point::MaterializeAfterPipelineInsertion,
             capture::fault::Point::MaterializeAfterSessionInsertion,
             capture::fault::Point::ReserveDispatchAfterInsertionBeforeLinkage,
         })
    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        const auto session = model.trackSession(pipeline);
        capture::fault::ScopedInjection injection(point);
        Check(Throws<std::bad_alloc>([&] { (void)model.reserveExecutedDispatch(session); }),
              "materialization transaction exposes injected allocation failure");
        Check(model.pipelines().empty() && model.sessions().empty() && model.dispatches().empty(),
              "failed hierarchy materialization leaves no orphan nodes or links");
        CheckTerminalAllocationFailure(model, "hierarchy materialization rollback");
    }
}

void TestMetadataAndArtifactRollback()
{
    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        model.updatePipelineFriendlyName(pipeline, "original");
        capture::fault::ScopedInjection injection(
            capture::fault::Point::UpdatePipelineNameBeforeCommit);
        Check(Throws<std::bad_alloc>(
                  [&] { model.updatePipelineFriendlyName(pipeline, "replacement"); }),
              "pipeline-name update fault is injected at the commit boundary");
        CheckTerminalAllocationFailure(model, "pipeline-name rollback");
    }

    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        const auto session = model.trackSession(pipeline);
        model.updateSessionFriendlyName(session, "original");
        capture::fault::ScopedInjection injection(
            capture::fault::Point::UpdateSessionNameBeforeCommit);
        Check(Throws<std::bad_alloc>([&]
                                     { model.updateSessionFriendlyName(session, "replacement"); }),
              "session-name update fault is injected at the commit boundary");
        CheckTerminalAllocationFailure(model, "session-name rollback");
    }

    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        const auto session = model.trackSession(pipeline);
        const auto dispatch = model.reserveExecutedDispatch(session);
        const auto pipelineArtifacts = model.findPipeline(pipeline)->artifacts.size();
        capture::fault::ScopedInjection injection(
            capture::fault::Point::AddPipelineArtifactBeforeCommit);
        Check(Throws<std::bad_alloc>(
                  [&]
                  {
                      model.addPipelineArtifact(
                          pipeline, {"artifact.bin", capture::ArtifactType::DebugDatabase, 1});
                  }),
              "pipeline-artifact update fault is injected before mutation");
        Check(model.findPipeline(pipeline)->artifacts.size() == pipelineArtifacts,
              "failed pipeline-artifact mutation leaves the materialized node unchanged");
        CheckTerminalAllocationFailure(model, "pipeline-artifact rollback");
        (void)dispatch;
    }

    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        const auto session = model.trackSession(pipeline);
        const auto dispatch = model.reserveExecutedDispatch(session);
        capture::fault::ScopedInjection injection(
            capture::fault::Point::AddDispatchArtifactBeforeCommit);
        Check(Throws<std::bad_alloc>(
                  [&]
                  {
                      model.addDispatchArtifact(
                          *dispatch.selectedDispatch,
                          {"dispatch.bin", capture::ArtifactType::DispatchStatistics, 2});
                  }),
              "dispatch-artifact update fault is injected before mutation");
        Check(model.findDispatch(*dispatch.selectedDispatch)->artifacts.empty(),
              "failed dispatch-artifact mutation leaves the dispatch unchanged");
        CheckTerminalAllocationFailure(model, "dispatch-artifact rollback");
    }
}

void TestTerminalStatusAndCounterPolicy()
{
    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        capture::fault::ScopedInjection injection(capture::fault::Point::MarkErrorBeforeCommit);
        Check(Throws<std::bad_alloc>([&] { model.markError("custom error"); }),
              "terminal status update participates in fault injection");
        CheckTerminalAllocationFailure(model, "terminal status rollback");
        Check(model.errorMessage() != "custom error",
              "failed custom-error publication cannot diverge from terminal status");
    }

    {
        capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
        const auto pipeline = model.trackPipeline();
        const auto session = model.trackSession(pipeline);
        capture::fault::ScopedInjection injection(
            capture::fault::Point::ReserveDispatchAfterInsertionBeforeLinkage,
            capture::fault::Failure::CounterExhaustion);
        Check(Throws<std::overflow_error>([&] { (void)model.reserveExecutedDispatch(session); }),
              "counter exhaustion is propagated to the owner boundary");
        Check(model.terminalFailure() == capture::CaptureTerminalFailure::CounterExhaustion &&
                  model.status() == capture::CaptureStatus::Error,
              "counter exhaustion permanently transitions capture to terminal error");
        Check(model.pipelines().empty() && model.sessions().empty() && model.dispatches().empty() &&
                  model.validateInvariants(),
              "counter exhaustion rolls back hierarchy publication before disabling capture");
    }
}

void TestVulkanExceptionPolicy()
{
    std::exception_ptr allocation;
    try
    {
        throw std::bad_alloc{};
    }
    catch (...)
    {
        allocation = std::current_exception();
    }
    const auto configuration =
        capture::abi::TranslateException(allocation, capture::abi::BoundaryPhase::Configuration);
    Check(configuration.action == capture::abi::Action::ReturnError &&
              configuration.result == VK_ERROR_OUT_OF_HOST_MEMORY,
          "configuration bad_alloc translation is reusable and explicit");

    const auto postForward = capture::abi::TranslateException(
        allocation, capture::abi::BoundaryPhase::PostForwardCapture);
    Check(postForward.action == capture::abi::Action::DisableCaptureAndPreserveDownstreamResult,
          "post-forward Stage 2 policy disables capture and preserves the downstream result");
}
} // namespace

int main()
{
    TestTrackedInstancePublicationRollback();
    TestUntrackedDownstreamRollback();
    TestUnexpectedPublicationRollback();
    TestSuccessfulInstancePublication();
    TestPipelineAndSessionTrackingRollback();
    TestHierarchyMaterializationRollback();
    TestMetadataAndArtifactRollback();
    TestTerminalStatusAndCounterPolicy();
    TestVulkanExceptionPolicy();

    if (failures != 0)
    {
        std::cerr << failures << " exception-safety test(s) failed\n";
        return 1;
    }
    std::cout << "All exception-safety tests passed\n";
    return 0;
}
