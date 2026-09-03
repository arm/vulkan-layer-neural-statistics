/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_model.hpp"

#include <map>
#include <mutex>

namespace capture
{
class CaptureModelHarness : public CaptureModel
{
  public:
    CaptureModelHarness(
        std::filesystem::path root, CaptureMetadata metadata, DispatchFilter filter,
        CaptureRootCollisionPolicy policy = CaptureRootCollisionPolicy::RejectExisting)
        : CaptureModel(std::move(root), std::move(metadata), policy), filter_(filter)
    {
    }

    PipelineId trackPipeline(std::optional<uint64_t> diagnosticHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        const PipelineId id = pipelineIds_.peek();
        CaptureModel::trackPipeline(id, {}, diagnosticHandle);
        pipelineIds_.commit();
        return id;
    }

    SessionId trackSession(PipelineId pipelineId,
                           std::optional<uint64_t> diagnosticHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        const SessionId id = sessionIds_.peek();
        CaptureModel::trackSession(id, pipelineId, 0, diagnosticHandle);
        sessionIds_.commit();
        executedIndices_.emplace(id, 0);
        return id;
    }

    DispatchReservation reserveExecutedDispatch(
        SessionId sessionId, std::optional<uint64_t> diagnosticCommandBufferHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        auto it = executedIndices_.find(sessionId);
        if (it == executedIndices_.end())
        {
            throw std::out_of_range("unknown session ID");
        }
        DispatchReservation reservation{it->second, std::nullopt};
        if (filter_.matches(it->second))
        {
            reservation.selectedDispatch = dispatchIds_.peek();
        }
        CaptureModel::materializeExecutedDispatch(sessionId, reservation,
                                                  diagnosticCommandBufferHandle);
        ++it->second;
        if (reservation.selected())
        {
            dispatchIds_.commit();
        }
        return reservation;
    }

  private:
    DispatchFilter filter_;
    IdAllocator<PipelineId> pipelineIds_;
    IdAllocator<SessionId> sessionIds_;
    IdAllocator<DispatchId> dispatchIds_;
    std::map<SessionId, uint64_t> executedIndices_;
    std::mutex identityMutex_;
};
} // namespace capture
