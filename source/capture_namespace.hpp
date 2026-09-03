/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace capture::names
{
inline constexpr std::string_view kCaptureDocument = "capture.json";
inline constexpr std::string_view kPipelineDocument = "pipeline.json";
inline constexpr std::string_view kSessionDocument = "session.json";
inline constexpr std::string_view kDispatchDocument = "dispatch.json";

inline constexpr std::string_view kPipelineDirectoryPrefix = "pipeline";
inline constexpr std::string_view kSessionDirectoryPrefix = "session";
inline constexpr std::string_view kDispatchDirectoryPrefix = "dispatch";
inline constexpr uint32_t kDirectoryIndexWidth = 6;

// Internal publication names are never part of the public capture format.
// Artifact validation reserves this prefix and the .tmp suffix using exact,
// case-sensitive comparisons, matching the capture collision-key policy.
inline constexpr std::string_view kInternalTemporaryPrefix = ".capture-internal-";
inline constexpr std::string_view kReservedTemporarySuffix = ".tmp";
inline constexpr std::size_t kMaximumArtifactNameBytes = 120;

struct ArtifactNameValidation
{
    bool valid{false};
    std::string collisionKey;
    std::string error;
};

std::string PortableNameKey(std::string_view asciiName);
std::string MakeIndexedDirectoryName(std::string_view prefix, uint64_t id);
bool IsReservedCaptureName(std::string_view portableKey) noexcept;
ArtifactNameValidation ValidateArtifactFileName(const std::filesystem::path &fileName);
} // namespace capture::names
