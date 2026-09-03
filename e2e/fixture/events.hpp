/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <cstdint>
#include <limits>
#include <string>
namespace fixture
{
std::int64_t nowNs();
void emitEvent(const std::string &name, std::uint32_t submission = 0, std::uint32_t attempt = 0,
               int result = 0, std::uint64_t size = std::numeric_limits<std::uint64_t>::max());
void emitSecondaryWrapperRecorded(std::uint32_t executions);
void emitApplicationFenceComplete(std::uint32_t submission);
void writeEventsFile(const std::string &path, std::int64_t startNs, std::int64_t endNs,
                     int exitCode);
} // namespace fixture
