/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_filesystem.hpp"
#include "capture_model.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace capture
{
// Single-threaded capture-domain transaction owner. CaptureWriter constructs
// and calls this object only on its writer thread.
class CaptureStore
{
  public:
    CaptureStore(std::filesystem::path root, CaptureMetadata metadata);

    void trackPipeline(PipelineId pipelineId, PipelineMetadata metadata,
                       std::optional<uint64_t> diagnosticHandle);
    void trackSession(SessionId sessionId, PipelineId pipelineId, uint64_t flags,
                      std::optional<uint64_t> diagnosticHandle);
    void materializeDispatch(SessionId sessionId, DispatchReservation reservation,
                             std::optional<uint64_t> diagnosticCommandBufferHandle);
    void updateMetadata(CaptureMetadata metadata);

    void updatePipelineFriendlyName(PipelineId pipelineId, const std::string &friendlyName);
    void updatePipelineMetadata(PipelineId pipelineId, PipelineMetadata metadata);
    void updateSessionFriendlyName(SessionId sessionId, const std::string &friendlyName);

    void writePipelineArtifact(PipelineId pipelineId, const std::filesystem::path &fileName,
                               ArtifactType type, const std::vector<uint8_t> &contents);
    void writeDispatchArtifact(DispatchId dispatchId, const std::filesystem::path &fileName,
                               ArtifactType type, const std::vector<uint8_t> &contents);

    void publishDocuments();
    void finishComplete();
    void finishError(const std::string &message);
    void bestEffortRecordError(const std::string &message) noexcept;

    CaptureStatus status() const noexcept;
    const std::string &errorMessage() const noexcept;

  private:
    struct PendingArtifact
    {
        std::filesystem::path fileName;
        ArtifactDescriptor descriptor;
        std::vector<uint8_t> contents;
    };

    std::string reserveArtifactPath(const std::filesystem::path &parent,
                                    const std::filesystem::path &fileName);
    void publishMaterializedDispatch(DispatchId dispatchId);
    void publishCapture();
    void publishPipeline(PipelineId pipelineId);
    void publishSession(SessionId sessionId);
    void publishDispatch(DispatchId dispatchId);

    std::filesystem::path root_;
    std::unique_ptr<CaptureModel> model_;
    std::unique_ptr<CaptureFileSystem> fileSystem_;
    std::map<PipelineId, std::map<std::string, PendingArtifact>> pendingPipelineArtifacts_;
    std::set<std::string> artifactPathKeys_;
};
} // namespace capture
