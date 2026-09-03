/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "dispatch_filter.hpp"
#include "resource_tracking.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace capture::submission
{
struct SubmittedOccurrence
{
    CommandBufferRecord::DispatchOccurrence occurrence;
    bool protectedSubmit{false};
};

enum class BuildResult
{
    Ready,
    Rejected,
};

struct PlannedOccurrence
{
    std::shared_ptr<SessionRecord> session;
    std::shared_ptr<StatsSnapshot> sourceSnapshot;
    std::shared_ptr<StatsSnapshot> snapshot;
    uint64_t executedIndex{0};
    std::optional<DispatchId> dispatchId;
};

struct Plan
{
    std::vector<PlannedOccurrence> occurrences;
    std::vector<std::shared_ptr<StatsSnapshot>> selectedSnapshots;
    uint64_t selectedCount{0};
    bool reserved{false};
    bool accepted{false};
};

bool appendCommandBufferOccurrences(const std::shared_ptr<CommandBufferRecord> &record,
                                    bool protectedSubmit,
                                    std::vector<SubmittedOccurrence> &occurrences,
                                    std::string &reason);

BuildResult build(const DispatchFilter &filter, bool captureEnabled,
                  const IdAllocator<DispatchId> &dispatchIds,
                  const std::vector<SubmittedOccurrence> &submitted, Plan &plan,
                  std::string &reason);

bool materialize(Plan &plan, std::vector<std::shared_ptr<StatsSnapshot>> snapshots,
                 std::string &reason);

void cancel(Plan &plan) noexcept;
void accept(IdAllocator<DispatchId> &dispatchIds, Plan &plan) noexcept;
void finish(Plan &plan, const char *captureFailure) noexcept;
} // namespace capture::submission
