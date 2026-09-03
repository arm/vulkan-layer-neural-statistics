/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_store.hpp"

#include "capture_namespace.hpp"
#include "capture_serialization.hpp"
#include "fault_injection.hpp"

#include <stdexcept>
#include <utility>

namespace capture
{
CaptureStore::CaptureStore(std::filesystem::path root, CaptureMetadata metadata)
    : root_(std::move(root))
{
    // Validate and allocate the complete in-memory configuration before
    // claiming any filesystem name.
    model_ = std::make_unique<CaptureModel>(root_, std::move(metadata));
    fileSystem_ = std::make_unique<CaptureFileSystem>(CaptureFileSystem::ClaimRoot(root_));

    // Until this succeeds the root has been claimed but does not contain a
    // valid capture document, so ordinary best-effort cleanup may remove it.
    publishCapture();
    fileSystem_->preserveClaimedRoot();
}

void CaptureStore::trackPipeline(PipelineId pipelineId, PipelineMetadata metadata,
                                 std::optional<uint64_t> diagnosticHandle)
{
    model_->trackPipeline(pipelineId, std::move(metadata), diagnosticHandle);
}

void CaptureStore::trackSession(SessionId sessionId, PipelineId pipelineId, uint64_t flags,
                                std::optional<uint64_t> diagnosticHandle)
{
    model_->trackSession(sessionId, pipelineId, flags, diagnosticHandle);
}

void CaptureStore::materializeDispatch(SessionId sessionId, DispatchReservation reservation,
                                       std::optional<uint64_t> diagnosticCommandBufferHandle)
{
    model_->materializeExecutedDispatch(sessionId, reservation, diagnosticCommandBufferHandle);
    if (reservation.selected())
    {
        publishMaterializedDispatch(*reservation.selectedDispatch);
    }
}

void CaptureStore::updateMetadata(CaptureMetadata metadata)
{
    model_->updateMetadata(std::move(metadata));
    publishCapture();
}

void CaptureStore::updatePipelineFriendlyName(PipelineId pipelineId,
                                              const std::string &friendlyName)
{
    model_->updatePipelineFriendlyName(pipelineId, friendlyName);
    if (model_->findPipeline(pipelineId) != nullptr)
    {
        publishPipeline(pipelineId);
    }
}

void CaptureStore::updatePipelineMetadata(PipelineId pipelineId, PipelineMetadata metadata)
{
    model_->updatePipelineMetadata(pipelineId, std::move(metadata));
    if (model_->findPipeline(pipelineId) != nullptr)
    {
        publishPipeline(pipelineId);
    }
}

void CaptureStore::updateSessionFriendlyName(SessionId sessionId, const std::string &friendlyName)
{
    model_->updateSessionFriendlyName(sessionId, friendlyName);
    if (model_->findSession(sessionId) != nullptr)
    {
        publishSession(sessionId);
    }
}

std::string CaptureStore::reserveArtifactPath(const std::filesystem::path &parent,
                                              const std::filesystem::path &fileName)
{
    const auto validation = names::ValidateArtifactFileName(fileName);
    if (!validation.valid)
    {
        throw std::invalid_argument(validation.error);
    }
    std::string key = parent.generic_string();
    if (!key.empty())
    {
        key.push_back('/');
    }
    key += validation.collisionKey;
    if (!artifactPathKeys_.insert(key).second)
    {
        throw std::invalid_argument(
            "artifact path collides under capture case-sensitive semantics");
    }
    return validation.collisionKey;
}

void CaptureStore::writePipelineArtifact(PipelineId pipelineId,
                                         const std::filesystem::path &fileName, ArtifactType type,
                                         const std::vector<uint8_t> &contents)
{
    if (!model_->containsPipeline(pipelineId))
    {
        throw std::out_of_range("unknown pipeline ID");
    }

    const auto parent = PipelineRelativePath(pipelineId);
    const std::string collisionKey = reserveArtifactPath(parent, fileName);
    const auto relative = parent / fileName;
    ArtifactDescriptor descriptor{relative, type, contents.size()};
    if (!model_->isPipelineMaterialized(pipelineId))
    {
        auto &pipelineArtifacts = pendingPipelineArtifacts_[pipelineId];
        const bool inserted =
            pipelineArtifacts
                .emplace(collisionKey, PendingArtifact{fileName, std::move(descriptor), contents})
                .second;
        if (!inserted)
        {
            throw std::invalid_argument("artifact path was already queued");
        }
        return;
    }

    model_->preparePipelineArtifactCommits(pipelineId, 1);
    fileSystem_->publish(relative, contents, FilePublicationMode::CreateNew);
    model_->commitPipelineArtifact(pipelineId, std::move(descriptor));
    publishPipeline(pipelineId);
}

void CaptureStore::writeDispatchArtifact(DispatchId dispatchId,
                                         const std::filesystem::path &fileName, ArtifactType type,
                                         const std::vector<uint8_t> &contents)
{
    const DispatchNode *dispatch = model_->findDispatch(dispatchId);
    if (dispatch == nullptr)
    {
        throw std::out_of_range("unknown dispatch ID");
    }
    const SessionNode *session = model_->findSession(dispatch->sessionId);
    const auto parent = DispatchRelativePath(session->pipelineId, session->id, dispatchId);
    (void)reserveArtifactPath(parent, fileName);
    const auto relative = parent / fileName;
    ArtifactDescriptor descriptor{relative, type, contents.size()};

    model_->prepareDispatchArtifactCommit(dispatchId);
    fileSystem_->publish(relative, contents, FilePublicationMode::CreateNew);
    model_->commitDispatchArtifact(dispatchId, std::move(descriptor));
    publishDispatch(dispatchId);
}

void CaptureStore::publishDocuments()
{
    for (const auto &[dispatchId, dispatch] : model_->dispatches())
    {
        const SessionNode *session = model_->findSession(dispatch.sessionId);
        fileSystem_->createDirectories(
            DispatchRelativePath(session->pipelineId, session->id, dispatchId));
        publishDispatch(dispatchId);
    }
    for (const auto &[sessionId, _] : model_->sessions())
    {
        publishSession(sessionId);
    }
    for (const auto &[pipelineId, _] : model_->pipelines())
    {
        publishPipeline(pipelineId);
    }
    publishCapture();
}

void CaptureStore::finishComplete()
{
    publishDocuments();
    const std::string completeDocument = SerializeCaptureJson(*model_, CaptureStatus::Complete, {});
    fileSystem_->publishText(names::kCaptureDocument, completeDocument,
                             FilePublicationMode::Replace);
    model_->markComplete();
}

void CaptureStore::finishError(const std::string &message)
{
    model_->markError(message);
    publishDocuments();
}

void CaptureStore::bestEffortRecordError(const std::string &message) noexcept
{
    if (!model_ || !fileSystem_)
    {
        return;
    }
    try
    {
        if (model_->status() == CaptureStatus::Incomplete)
        {
            model_->markError(message);
        }
    }
    catch (...)
    {
    }
    try
    {
        publishDocuments();
    }
    catch (...)
    {
        // Keep the last atomically published capture document. Publishing only
        // capture.json after a hierarchy failure could create dangling document
        // references, so recovery never advances the root document alone.
    }
}

CaptureStatus CaptureStore::status() const noexcept
{
    return model_->status();
}

const std::string &CaptureStore::errorMessage() const noexcept
{
    return model_->errorMessage();
}

void CaptureStore::publishMaterializedDispatch(DispatchId dispatchId)
{
    const DispatchNode *dispatch = model_->findDispatch(dispatchId);
    const SessionNode *session = model_->findSession(dispatch->sessionId);
    const PipelineId pipelineId = session->pipelineId;
    fileSystem_->createDirectories(DispatchRelativePath(pipelineId, session->id, dispatchId));

    if (auto pending = pendingPipelineArtifacts_.find(pipelineId);
        pending != pendingPipelineArtifacts_.end())
    {
        model_->preparePipelineArtifactCommits(pipelineId, pending->second.size());
        for (auto &[_, artifact] : pending->second)
        {
            fileSystem_->publish(PipelineRelativePath(pipelineId) / artifact.fileName,
                                 artifact.contents, FilePublicationMode::CreateNew);
            model_->commitPipelineArtifact(pipelineId, std::move(artifact.descriptor));
        }
        pendingPipelineArtifacts_.erase(pending);
    }

    publishDispatch(dispatchId);
    publishSession(session->id);
    publishPipeline(pipelineId);
    publishCapture();
}

void CaptureStore::publishCapture()
{
    fileSystem_->publishText(names::kCaptureDocument, SerializeCaptureJson(*model_),
                             FilePublicationMode::Replace);
}

void CaptureStore::publishPipeline(PipelineId pipelineId)
{
    fileSystem_->publishText(PipelineRelativePath(pipelineId) / names::kPipelineDocument,
                             SerializePipelineJson(*model_, pipelineId),
                             FilePublicationMode::Replace);
}

void CaptureStore::publishSession(SessionId sessionId)
{
    const SessionNode *session = model_->findSession(sessionId);
    fileSystem_->publishText(
        SessionRelativePath(session->pipelineId, sessionId) / names::kSessionDocument,
        SerializeSessionJson(*model_, sessionId), FilePublicationMode::Replace);
}

void CaptureStore::publishDispatch(DispatchId dispatchId)
{
    const DispatchNode *dispatch = model_->findDispatch(dispatchId);
    const SessionNode *session = model_->findSession(dispatch->sessionId);
    fileSystem_->publishText(DispatchRelativePath(session->pipelineId, session->id, dispatchId) /
                                 names::kDispatchDocument,
                             SerializeDispatchJson(*model_, dispatchId),
                             FilePublicationMode::Replace);
}
} // namespace capture
