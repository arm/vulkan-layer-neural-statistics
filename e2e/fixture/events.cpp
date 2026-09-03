/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "events.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
namespace fixture
{
std::int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
void emitEvent(const std::string &name, std::uint32_t submission, std::uint32_t attempt, int result,
               std::uint64_t size)
{
    static std::mutex outputMutex;
    const std::lock_guard<std::mutex> lock(outputMutex);
    std::cout << "E2E_EVENT {\"event\":\"" << name << "\",\"timestamp_ns\":" << nowNs()
              << ",\"submission\":" << submission << ",\"attempt\":" << attempt
              << ",\"result\":" << result;
    if (size != std::numeric_limits<std::uint64_t>::max())
        std::cout << ",\"size\":" << size;
    std::cout << "}" << std::endl;
}
void emitSecondaryWrapperRecorded(std::uint32_t executions)
{
    emitEvent("secondary_wrapper_recorded", 0, 0, static_cast<int>(executions));
}
void emitApplicationFenceComplete(std::uint32_t submission)
{
    emitEvent("application_fence_complete", submission, 0, 0);
}
void writeEventsFile(const std::string &path, std::int64_t startNs, std::int64_t endNs,
                     int exitCode)
{
    if (path.empty())
        return;
    const std::filesystem::path output(path);
    if (output.has_parent_path())
        std::filesystem::create_directories(output.parent_path());
    std::ofstream stream(output, std::ios::out | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot open events output " + path);
    stream << "{\n  \"schema_version\": 1,\n  \"process_start_ns\": " << startNs
           << ",\n  \"process_end_ns\": " << endNs << ",\n  \"exit_code\": " << exitCode << "\n}\n";
}
} // namespace fixture
