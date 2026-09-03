/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_model.hpp"

#include "fault_injection.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <new>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace capture
{
namespace
{
const std::string kAllocationFailureMessage = "capture disabled after host allocation failure";
const std::string kCounterExhaustionMessage =
    "capture disabled after capture-local sequence exhaustion";
const std::string kUnexpectedFailureMessage = "capture disabled after unexpected model failure";

bool ContainsOnce(const std::vector<SessionId> &values, SessionId value) noexcept
{
    return std::count(values.begin(), values.end(), value) == 1;
}

bool ContainsOnce(const std::vector<DispatchId> &values, DispatchId value) noexcept
{
    return std::count(values.begin(), values.end(), value) == 1;
}

bool ContainsDevice(const CaptureMetadata &metadata, LogicalDeviceId id) noexcept
{
    return std::ranges::any_of(metadata.devices,
                               [id](const DeviceMetadata &device) { return device.id == id; });
}

bool HasValidDevices(const CaptureMetadata &metadata) noexcept
{
    for (std::size_t index = 0; index < metadata.devices.size(); ++index)
    {
        const auto &device = metadata.devices[index];
        if (device.name.empty())
        {
            return false;
        }
        for (std::size_t prior = 0; prior < index; ++prior)
        {
            if (metadata.devices[prior].id == device.id)
            {
                return false;
            }
        }
    }
    return true;
}

void ValidateMetadata(const CaptureMetadata &metadata)
{
    if (metadata.statisticsMode > 1)
    {
        throw std::invalid_argument("statistics mode must be 0 or 1");
    }
    if (!HasValidDevices(metadata))
    {
        throw std::invalid_argument("capture metadata contains an invalid logical device table");
    }
}
} // namespace

static_assert(std::is_nothrow_move_assignable_v<std::string>);
static_assert(std::is_nothrow_move_constructible_v<ArtifactDescriptor>);
static_assert(std::is_nothrow_destructible_v<PipelineNode>);
static_assert(std::is_nothrow_destructible_v<SessionNode>);
static_assert(std::is_nothrow_destructible_v<DispatchNode>);

std::string_view ToString(CaptureStatus status) noexcept
{
    switch (status)
    {
    case CaptureStatus::Incomplete:
        return "incomplete";
    case CaptureStatus::Complete:
        return "complete";
    case CaptureStatus::Error:
        return "error";
    }
    return "error";
}

std::string_view ToString(ArtifactType type) noexcept
{
    switch (type)
    {
    case ArtifactType::DebugDatabase:
        return "debug_database";
    case ArtifactType::StatisticsInfo:
        return "statistics_info";
    case ArtifactType::ShaderModule:
        return "shader_module";
    case ArtifactType::DispatchStatistics:
        return "dispatch_statistics";
    }
    return "dispatch_statistics";
}

std::string MakeIndexedFolderName(std::string_view prefix, uint64_t id)
{
    return names::MakeIndexedDirectoryName(prefix, id);
}

std::string PipelineFolderName(PipelineId id)
{
    return names::MakeIndexedDirectoryName(names::kPipelineDirectoryPrefix, id.value());
}

std::string SessionFolderName(SessionId id)
{
    return names::MakeIndexedDirectoryName(names::kSessionDirectoryPrefix, id.value());
}

std::string DispatchFolderName(DispatchId id)
{
    return names::MakeIndexedDirectoryName(names::kDispatchDirectoryPrefix, id.value());
}

std::filesystem::path PipelineRelativePath(PipelineId id)
{
    return std::filesystem::path(PipelineFolderName(id));
}

std::filesystem::path SessionRelativePath(PipelineId pipelineId, SessionId sessionId)
{
    return PipelineRelativePath(pipelineId) / SessionFolderName(sessionId);
}

std::filesystem::path DispatchRelativePath(PipelineId pipelineId, SessionId sessionId,
                                           DispatchId dispatchId)
{
    return SessionRelativePath(pipelineId, sessionId) / DispatchFolderName(dispatchId);
}

CaptureRootDecision EvaluateCaptureRoot(const std::filesystem::path &captureRoot,
                                        bool alreadyExists,
                                        CaptureRootCollisionPolicy policy) noexcept
{
    if (captureRoot.empty())
    {
        return CaptureRootDecision::RejectEmptyPath;
    }
    if (alreadyExists && policy == CaptureRootCollisionPolicy::RejectExisting)
    {
        return CaptureRootDecision::RejectExisting;
    }
    return CaptureRootDecision::CreateNew;
}

CaptureModel::CaptureModel(std::filesystem::path captureRoot, CaptureMetadata metadata,
                           CaptureRootCollisionPolicy collisionPolicy)
    : captureRoot_(std::move(captureRoot)), metadata_(std::move(metadata)),
      collisionPolicy_(collisionPolicy)
{
    if (captureRoot_.empty())
    {
        throw std::invalid_argument("capture root must not be empty");
    }
    ValidateMetadata(metadata_);
}

void CaptureModel::trackPipeline(PipelineId pipelineId, PipelineMetadata pipelineMetadata,
                                 std::optional<uint64_t> diagnosticHandle)
{
    ensureMutable();
    if (!ContainsDevice(metadata_, pipelineMetadata.deviceId))
    {
        throw std::invalid_argument("pipeline references an unknown logical device");
    }
    bool inserted = false;
    try
    {
        inserted =
            pendingPipelines_
                .emplace(pipelineId,
                         PipelineNode{
                             pipelineId, {}, diagnosticHandle, std::move(pipelineMetadata), {}, {}})
                .second;
        if (!inserted)
        {
            throw std::logic_error("duplicate pipeline ID");
        }
        fault::Checkpoint(fault::Point::TrackPipelineAfterInsertion);
    }
    catch (const std::bad_alloc &)
    {
        if (inserted)
        {
            pendingPipelines_.erase(pipelineId);
        }
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        if (inserted)
        {
            pendingPipelines_.erase(pipelineId);
        }
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::trackSession(SessionId sessionId, PipelineId pipelineId, uint64_t flags,
                                std::optional<uint64_t> diagnosticHandle)
{
    (void)requirePipeline(pipelineId);
    ensureMutable();
    bool inserted = false;
    try
    {
        inserted =
            pendingSessions_
                .emplace(sessionId, SessionNode{sessionId,
                                                pipelineId,
                                                requirePipeline(pipelineId).metadata.deviceId,
                                                {},
                                                diagnosticHandle,
                                                flags,
                                                {}})
                .second;
        if (!inserted)
        {
            throw std::logic_error("duplicate session ID");
        }
        fault::Checkpoint(fault::Point::TrackSessionAfterInsertion);
    }
    catch (const std::bad_alloc &)
    {
        if (inserted)
        {
            pendingSessions_.erase(sessionId);
        }
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        if (inserted)
        {
            pendingSessions_.erase(sessionId);
        }
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::updateMetadata(CaptureMetadata captureMetadata)
{
    ensureMutable();
    try
    {
        ValidateMetadata(captureMetadata);
        const auto pipelinesReferenceKnownDevices = [&](const auto &pipelines)
        {
            return std::ranges::all_of(
                pipelines, [&](const auto &entry)
                { return ContainsDevice(captureMetadata, entry.second.metadata.deviceId); });
        };
        if (!pipelinesReferenceKnownDevices(pendingPipelines_) ||
            !pipelinesReferenceKnownDevices(pipelines_))
        {
            throw std::invalid_argument(
                "capture metadata removes a logical device referenced by a pipeline");
        }

        metadata_ = std::move(captureMetadata);
    }
    catch (const std::invalid_argument &)
    {
        throw;
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::updatePipelineFriendlyName(PipelineId pipelineId, std::string friendlyName)
{
    auto &pipeline = requirePipeline(pipelineId);
    ensureMutable();
    try
    {
        fault::Checkpoint(fault::Point::UpdatePipelineNameBeforeCommit);
        pipeline.friendlyName = std::move(friendlyName);
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::updatePipelineMetadata(PipelineId pipelineId, PipelineMetadata pipelineMetadata)
{
    auto &pipeline = requirePipeline(pipelineId);
    ensureMutable();
    if (pipelineMetadata.deviceId != pipeline.metadata.deviceId)
    {
        throw std::invalid_argument("pipeline logical device identity cannot change");
    }
    try
    {
        pipeline.metadata = std::move(pipelineMetadata);
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::updateSessionFriendlyName(SessionId sessionId, std::string friendlyName)
{
    auto &session = requireSession(sessionId);
    ensureMutable();
    try
    {
        fault::Checkpoint(fault::Point::UpdateSessionNameBeforeCommit);
        session.friendlyName = std::move(friendlyName);
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::materializeExecutedDispatch(
    SessionId sessionId, DispatchReservation reservation,
    std::optional<uint64_t> diagnosticCommandBufferHandle)
{
    (void)requireSession(sessionId);
    ensureMutable();
    if (!reservation.selected())
    {
        return;
    }

    try
    {
        const DispatchId dispatchId = *reservation.selectedDispatch;
        const auto pendingPipeline = pendingPipelines_.find(requireSession(sessionId).pipelineId);
        const auto pendingSession = pendingSessions_.find(sessionId);
        const bool materializePipeline = pendingPipeline != pendingPipelines_.end();
        const bool materializeSession = pendingSession != pendingSessions_.end();

        PipelineNode pipelineCandidate;
        SessionNode sessionCandidate;
        if (materializePipeline)
        {
            pipelineCandidate = pendingPipeline->second;
            pipelineCandidate.sessions.reserve(pipelineCandidate.sessions.size() +
                                               (materializeSession ? 1 : 0));
        }
        else if (materializeSession)
        {
            pipelines_.at(pendingSession->second.pipelineId)
                .sessions.reserve(pipelines_.at(pendingSession->second.pipelineId).sessions.size() +
                                  1);
        }

        if (materializeSession)
        {
            sessionCandidate = pendingSession->second;
            sessionCandidate.dispatches.reserve(1);
        }
        else
        {
            sessions_.at(sessionId).dispatches.reserve(sessions_.at(sessionId).dispatches.size() +
                                                       1);
        }

        DispatchNode dispatchCandidate{
            dispatchId, sessionId, reservation.executedIndex, diagnosticCommandBufferHandle, {}};

        bool pipelineInserted = false;
        bool sessionInserted = false;
        bool dispatchInserted = false;
        try
        {
            if (materializePipeline)
            {
                pipelineInserted =
                    pipelines_.emplace(pipelineCandidate.id, std::move(pipelineCandidate)).second;
                if (!pipelineInserted)
                {
                    throw std::logic_error("pipeline materialization collision");
                }
                fault::Checkpoint(fault::Point::MaterializeAfterPipelineInsertion);
            }

            if (materializeSession)
            {
                sessionInserted =
                    sessions_.emplace(sessionCandidate.id, std::move(sessionCandidate)).second;
                if (!sessionInserted)
                {
                    throw std::logic_error("session materialization collision");
                }
                fault::Checkpoint(fault::Point::MaterializeAfterSessionInsertion);
            }

            dispatchInserted = dispatches_.emplace(dispatchId, std::move(dispatchCandidate)).second;
            if (!dispatchInserted)
            {
                throw std::logic_error("dispatch ID collision");
            }
            fault::Checkpoint(fault::Point::ReserveDispatchAfterInsertionBeforeLinkage);
        }
        catch (...)
        {
            if (dispatchInserted)
            {
                dispatches_.erase(dispatchId);
            }
            if (sessionInserted)
            {
                sessions_.erase(sessionId);
            }
            if (pipelineInserted)
            {
                pipelines_.erase(requireSession(sessionId).pipelineId);
            }
            throw;
        }

        auto &materializedSession = sessions_.at(sessionId);
        materializedSession.dispatches.push_back(dispatchId);
        if (materializeSession)
        {
            pipelines_.at(materializedSession.pipelineId).sessions.push_back(sessionId);
            pendingSessions_.erase(sessionId);
        }
        if (materializePipeline)
        {
            pendingPipelines_.erase(materializedSession.pipelineId);
        }
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (const std::overflow_error &)
    {
        failTerminal(CaptureTerminalFailure::CounterExhaustion);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::preparePipelineArtifactCommits(PipelineId pipelineId,
                                                  std::size_t additionalCount)
{
    auto &pipeline = requirePipeline(pipelineId);
    ensureMutable();
    try
    {
        if (additionalCount > pipeline.artifacts.max_size() - pipeline.artifacts.size())
        {
            throw std::length_error("pipeline artifact descriptor capacity exhausted");
        }
        fault::Checkpoint(fault::Point::AddPipelineArtifactBeforeCommit);
        pipeline.artifacts.reserve(pipeline.artifacts.size() + additionalCount);
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (const std::overflow_error &)
    {
        failTerminal(CaptureTerminalFailure::CounterExhaustion);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::commitPipelineArtifact(PipelineId pipelineId,
                                          ArtifactDescriptor artifact) noexcept
{
    PipelineNode *pipeline = nullptr;
    if (auto pending = pendingPipelines_.find(pipelineId); pending != pendingPipelines_.end())
    {
        pipeline = &pending->second;
    }
    else if (auto materialized = pipelines_.find(pipelineId); materialized != pipelines_.end())
    {
        pipeline = &materialized->second;
    }
    assert(pipeline != nullptr);
    assert(pipeline->artifacts.size() < pipeline->artifacts.capacity());
    pipeline->artifacts.push_back(std::move(artifact));
}

void CaptureModel::prepareDispatchArtifactCommit(DispatchId dispatchId)
{
    auto node = dispatches_.find(dispatchId);
    if (node == dispatches_.end())
    {
        throw std::out_of_range("unknown dispatch ID");
    }
    ensureMutable();
    try
    {
        fault::Checkpoint(fault::Point::AddDispatchArtifactBeforeCommit);
        node->second.artifacts.reserve(node->second.artifacts.size() + 1);
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (const std::overflow_error &)
    {
        failTerminal(CaptureTerminalFailure::CounterExhaustion);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

void CaptureModel::commitDispatchArtifact(DispatchId dispatchId,
                                          ArtifactDescriptor artifact) noexcept
{
    auto node = dispatches_.find(dispatchId);
    assert(node != dispatches_.end());
    assert(node->second.artifacts.size() < node->second.artifacts.capacity());
    node->second.artifacts.push_back(std::move(artifact));
}

void CaptureModel::addPipelineArtifact(PipelineId pipelineId, ArtifactDescriptor artifact)
{
    preparePipelineArtifactCommits(pipelineId, 1);
    commitPipelineArtifact(pipelineId, std::move(artifact));
}

void CaptureModel::addDispatchArtifact(DispatchId dispatchId, ArtifactDescriptor artifact)
{
    prepareDispatchArtifactCommit(dispatchId);
    commitDispatchArtifact(dispatchId, std::move(artifact));
}

void CaptureModel::markComplete()
{
    ensureMutable();
    errorMessage_.clear();
    status_ = CaptureStatus::Complete;
}

void CaptureModel::markError(std::string message)
{
    ensureMutable();
    try
    {
        fault::Checkpoint(fault::Point::MarkErrorBeforeCommit);
        errorMessage_ = std::move(message);
        status_ = CaptureStatus::Error;
    }
    catch (const std::bad_alloc &)
    {
        failTerminal(CaptureTerminalFailure::AllocationFailure);
        throw;
    }
    catch (const std::overflow_error &)
    {
        failTerminal(CaptureTerminalFailure::CounterExhaustion);
        throw;
    }
    catch (...)
    {
        failTerminal(CaptureTerminalFailure::UnexpectedFailure);
        throw;
    }
}

const std::filesystem::path &CaptureModel::captureRoot() const noexcept
{
    return captureRoot_;
}

const CaptureMetadata &CaptureModel::metadata() const noexcept
{
    return metadata_;
}

CaptureStatus CaptureModel::status() const noexcept
{
    return status_;
}

CaptureTerminalFailure CaptureModel::terminalFailure() const noexcept
{
    return terminalFailure_;
}

const std::string &CaptureModel::errorMessage() const noexcept
{
    switch (terminalFailure_)
    {
    case CaptureTerminalFailure::AllocationFailure:
        return kAllocationFailureMessage;
    case CaptureTerminalFailure::CounterExhaustion:
        return kCounterExhaustionMessage;
    case CaptureTerminalFailure::UnexpectedFailure:
        return kUnexpectedFailureMessage;
    case CaptureTerminalFailure::None:
        return errorMessage_;
    }
    return kUnexpectedFailureMessage;
}

CaptureRootCollisionPolicy CaptureModel::collisionPolicy() const noexcept
{
    return collisionPolicy_;
}

const std::map<PipelineId, PipelineNode> &CaptureModel::pipelines() const noexcept
{
    return pipelines_;
}

const std::map<SessionId, SessionNode> &CaptureModel::sessions() const noexcept
{
    return sessions_;
}

const std::map<DispatchId, DispatchNode> &CaptureModel::dispatches() const noexcept
{
    return dispatches_;
}

bool CaptureModel::containsPipeline(PipelineId id) const noexcept
{
    return pendingPipelines_.contains(id) || pipelines_.contains(id);
}

bool CaptureModel::containsSession(SessionId id) const noexcept
{
    return pendingSessions_.contains(id) || sessions_.contains(id);
}

bool CaptureModel::isPipelineMaterialized(PipelineId id) const noexcept
{
    return pipelines_.contains(id);
}

bool CaptureModel::isSessionMaterialized(SessionId id) const noexcept
{
    return sessions_.contains(id);
}

const PipelineNode *CaptureModel::findPipeline(PipelineId id) const noexcept
{
    const auto node = pipelines_.find(id);
    return node == pipelines_.end() ? nullptr : &node->second;
}

const SessionNode *CaptureModel::findSession(SessionId id) const noexcept
{
    const auto node = sessions_.find(id);
    return node == sessions_.end() ? nullptr : &node->second;
}

const DispatchNode *CaptureModel::findDispatch(DispatchId id) const noexcept
{
    const auto node = dispatches_.find(id);
    return node == dispatches_.end() ? nullptr : &node->second;
}

std::filesystem::path CaptureModel::pipelinePath(PipelineId id) const
{
    (void)requirePipeline(id);
    return captureRoot_ / PipelineRelativePath(id);
}

std::filesystem::path CaptureModel::sessionPath(SessionId id) const
{
    const auto &session = requireSession(id);
    return captureRoot_ / SessionRelativePath(session.pipelineId, id);
}

std::filesystem::path CaptureModel::dispatchPath(DispatchId id) const
{
    const auto node = dispatches_.find(id);
    if (node == dispatches_.end())
    {
        throw std::out_of_range("unknown dispatch ID");
    }
    const auto &session = requireSession(node->second.sessionId);
    return captureRoot_ / DispatchRelativePath(session.pipelineId, session.id, id);
}

bool CaptureModel::validateInvariants() const noexcept
{
    if (!HasValidDevices(metadata_))
    {
        return false;
    }

    for (const auto &[id, pipeline] : pendingPipelines_)
    {
        if (id != pipeline.id || pipelines_.contains(id) || !pipeline.sessions.empty() ||
            !ContainsDevice(metadata_, pipeline.metadata.deviceId))
        {
            return false;
        }
    }

    for (const auto &[id, session] : pendingSessions_)
    {
        const auto pendingPipeline = pendingPipelines_.find(session.pipelineId);
        const auto materializedPipeline = pipelines_.find(session.pipelineId);
        const PipelineNode *pipeline =
            pendingPipeline != pendingPipelines_.end() ? &pendingPipeline->second
            : materializedPipeline != pipelines_.end() ? &materializedPipeline->second
                                                       : nullptr;
        if (id != session.id || sessions_.contains(id) || !session.dispatches.empty() ||
            pipeline == nullptr || session.deviceId != pipeline->metadata.deviceId)
        {
            return false;
        }
    }

    for (const auto &[id, pipeline] : pipelines_)
    {
        if (id != pipeline.id || pendingPipelines_.contains(id) ||
            !ContainsDevice(metadata_, pipeline.metadata.deviceId))
        {
            return false;
        }
        for (const SessionId sessionId : pipeline.sessions)
        {
            const auto session = sessions_.find(sessionId);
            if (session == sessions_.end() || session->second.pipelineId != id ||
                !ContainsOnce(pipeline.sessions, sessionId))
            {
                return false;
            }
        }
    }

    for (const auto &[id, session] : sessions_)
    {
        const auto pipeline = pipelines_.find(session.pipelineId);
        if (id != session.id || pendingSessions_.contains(id) || pipeline == pipelines_.end() ||
            session.deviceId != pipeline->second.metadata.deviceId ||
            !ContainsOnce(pipeline->second.sessions, id))
        {
            return false;
        }
        for (const DispatchId dispatchId : session.dispatches)
        {
            const auto dispatch = dispatches_.find(dispatchId);
            if (dispatch == dispatches_.end() || dispatch->second.sessionId != id ||
                !ContainsOnce(session.dispatches, dispatchId))
            {
                return false;
            }
        }
    }

    for (const auto &[id, dispatch] : dispatches_)
    {
        const auto session = sessions_.find(dispatch.sessionId);
        if (id != dispatch.id || session == sessions_.end() ||
            !ContainsOnce(session->second.dispatches, id))
        {
            return false;
        }
    }

    return true;
}

PipelineNode &CaptureModel::requirePipeline(PipelineId id)
{
    if (auto pending = pendingPipelines_.find(id); pending != pendingPipelines_.end())
    {
        return pending->second;
    }
    if (auto materialized = pipelines_.find(id); materialized != pipelines_.end())
    {
        return materialized->second;
    }
    throw std::out_of_range("unknown pipeline ID");
}

const PipelineNode &CaptureModel::requirePipeline(PipelineId id) const
{
    if (auto pending = pendingPipelines_.find(id); pending != pendingPipelines_.end())
    {
        return pending->second;
    }
    if (auto materialized = pipelines_.find(id); materialized != pipelines_.end())
    {
        return materialized->second;
    }
    throw std::out_of_range("unknown pipeline ID");
}

SessionNode &CaptureModel::requireSession(SessionId id)
{
    if (auto pending = pendingSessions_.find(id); pending != pendingSessions_.end())
    {
        return pending->second;
    }
    if (auto materialized = sessions_.find(id); materialized != sessions_.end())
    {
        return materialized->second;
    }
    throw std::out_of_range("unknown session ID");
}

const SessionNode &CaptureModel::requireSession(SessionId id) const
{
    if (auto pending = pendingSessions_.find(id); pending != pendingSessions_.end())
    {
        return pending->second;
    }
    if (auto materialized = sessions_.find(id); materialized != sessions_.end())
    {
        return materialized->second;
    }
    throw std::out_of_range("unknown session ID");
}

void CaptureModel::ensureMutable() const
{
    if (terminalFailure_ != CaptureTerminalFailure::None || status_ != CaptureStatus::Incomplete)
    {
        throw CaptureTerminalError{};
    }
}

void CaptureModel::failTerminal(CaptureTerminalFailure failure) noexcept
{
    terminalFailure_ = failure;
    status_ = CaptureStatus::Error;
    errorMessage_.clear();
}
} // namespace capture
