/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_model.hpp"

#include <string>
#include <string_view>

namespace capture
{
// Deterministic JSON: ordered keys, stable ID ordering, four-space indentation.
std::string SerializeCaptureJson(const CaptureModel &model);
std::string SerializeCaptureJson(const CaptureModel &model, CaptureStatus status,
                                 std::string_view errorMessage);
std::string SerializePipelineJson(const CaptureModel &model, PipelineId pipelineId);
std::string SerializeSessionJson(const CaptureModel &model, SessionId sessionId);
std::string SerializeDispatchJson(const CaptureModel &model, DispatchId dispatchId);
} // namespace capture
