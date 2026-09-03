/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "submission_planner.hpp"

#include <limits>
#include <unordered_map>

namespace capture::submission
{
namespace
{
bool isFirstSelectedSession(const std::vector<std::shared_ptr<StatsSnapshot>> &snapshots,
                            size_t index) noexcept
{
    const SessionRecord *session = snapshots[index]->session.get();
    for (size_t i = 0; i < index; ++i)
    {
        if (snapshots[i]->session.get() == session)
        {
            return false;
        }
    }
    return true;
}

void releaseReservations(Plan &plan, const char *captureFailure) noexcept
{
    if (!plan.reserved)
    {
        return;
    }
    for (auto &occurrence : plan.occurrences)
    {
        if (!occurrence.dispatchId.has_value() || occurrence.sourceSnapshot == nullptr)
        {
            continue;
        }
        if (occurrence.sourceSnapshot->inFlightUses != 0)
        {
            --occurrence.sourceSnapshot->inFlightUses;
        }
        occurrence.sourceSnapshot->inFlight = occurrence.sourceSnapshot->inFlightUses != 0;
    }
    for (size_t i = 0; i < plan.selectedSnapshots.size(); ++i)
    {
        plan.selectedSnapshots[i]->inFlightUses = 0;
        plan.selectedSnapshots[i]->inFlight = false;
        if (!isFirstSelectedSession(plan.selectedSnapshots, i))
        {
            continue;
        }
        const auto &session = plan.selectedSnapshots[i]->session;
        if (captureFailure != nullptr && session->captureError.empty())
        {
            try
            {
                session->captureError = captureFailure;
            }
            catch (...)
            {
            }
        }
    }
    plan.reserved = false;
}
} // namespace

bool appendCommandBufferOccurrences(const std::shared_ptr<CommandBufferRecord> &record,
                                    bool protectedSubmit,
                                    std::vector<SubmittedOccurrence> &occurrences,
                                    std::string &reason)
{
    if (record == nullptr)
    {
        reason = "submission references an untracked command buffer";
        return false;
    }
    if (!record->snapshotError.empty())
    {
        reason = record->snapshotError;
        return false;
    }
    occurrences.reserve(occurrences.size() + record->dispatches.size());
    for (const auto &occurrence : record->dispatches)
    {
        occurrences.push_back({occurrence, protectedSubmit});
    }
    return true;
}

BuildResult build(const DispatchFilter &filter, bool captureEnabled,
                  const IdAllocator<DispatchId> &dispatchIds,
                  const std::vector<SubmittedOccurrence> &submitted, Plan &plan,
                  std::string &reason)
{
    plan = {};

    // Once capture is terminal, planning is bypassed. Leave the plan empty
    // so callers forward the original submission without archive command
    // buffers, reservations, or capture-identity bookkeeping.
    if (!captureEnabled)
    {
        return BuildResult::Ready;
    }

    plan.occurrences.reserve(submitted.size());

    std::unordered_map<SessionRecord *, uint64_t> perSessionOffsets;
    std::optional<DispatchId> firstDispatch;

    for (size_t submittedIndex = 0; submittedIndex < submitted.size(); ++submittedIndex)
    {
        const auto &submittedOccurrence = submitted[submittedIndex];
        const auto &occurrence = submittedOccurrence.occurrence;
        if (occurrence.session == nullptr)
        {
            reason = "submission contains an occurrence without a tracked session";
            return BuildResult::Rejected;
        }

        auto &offset = perSessionOffsets[occurrence.session.get()];
        if (occurrence.session->executionIndexExhausted ||
            occurrence.session->nextExecutionIndex > std::numeric_limits<uint64_t>::max() - offset)
        {
            reason = "per-session executed-dispatch sequence exhausted";
            return BuildResult::Rejected;
        }
        const uint64_t executedIndex = occurrence.session->nextExecutionIndex + offset;
        ++offset;

        PlannedOccurrence planned{occurrence.session, occurrence.snapshot, nullptr, executedIndex,
                                  std::nullopt};
        if (!filter.matches(executedIndex))
        {
            plan.occurrences.emplace_back(std::move(planned));
            continue;
        }

        if (submittedOccurrence.protectedSubmit)
        {
            reason = "selected capture rejects protected submissions";
            return BuildResult::Rejected;
        }
        if (!occurrence.unsupportedReason.empty())
        {
            reason = occurrence.unsupportedReason;
            return BuildResult::Rejected;
        }
        if (occurrence.snapshot == nullptr)
        {
            reason = "selected capture has no recorded snapshot destination";
            return BuildResult::Rejected;
        }
        if (!occurrence.session->captureError.empty())
        {
            reason = occurrence.session->captureError;
            return BuildResult::Rejected;
        }
        if (!firstDispatch.has_value())
        {
            if (dispatchIds.exhausted())
            {
                reason = "selected-dispatch ID sequence exhausted";
                return BuildResult::Rejected;
            }
            firstDispatch = dispatchIds.peek();
        }
        if (firstDispatch->value() > std::numeric_limits<uint64_t>::max() - plan.selectedCount)
        {
            reason = "selected-dispatch ID sequence exhausted";
            return BuildResult::Rejected;
        }

        planned.dispatchId = DispatchId(firstDispatch->value() + plan.selectedCount);
        ++plan.selectedCount;
        plan.occurrences.emplace_back(std::move(planned));
    }
    return BuildResult::Ready;
}

bool materialize(Plan &plan, std::vector<std::shared_ptr<StatsSnapshot>> snapshots,
                 std::string &reason)
{
    if (snapshots.size() != plan.selectedCount)
    {
        reason = "submission snapshot materialization count does not match the selected occurrence "
                 "count";
        return false;
    }

    plan.selectedSnapshots = std::move(snapshots);
    size_t selectedIndex = 0;
    for (auto &occurrence : plan.occurrences)
    {
        if (!occurrence.dispatchId.has_value())
        {
            continue;
        }
        const auto &snapshot = plan.selectedSnapshots[selectedIndex++];
        if (snapshot == nullptr)
        {
            reason = "submission snapshot materialization produced a null destination";
            plan.selectedSnapshots.clear();
            return false;
        }
        snapshot->inFlightUses = 1;
        snapshot->inFlight = true;
        ++occurrence.sourceSnapshot->inFlightUses;
        occurrence.sourceSnapshot->inFlight = true;
        occurrence.snapshot = snapshot;
    }
    plan.reserved = true;
    return true;
}

void cancel(Plan &plan) noexcept
{
    releaseReservations(plan, nullptr);
}

void accept(IdAllocator<DispatchId> &dispatchIds, Plan &plan) noexcept
{
    if (plan.accepted)
    {
        return;
    }
    for (auto &occurrence : plan.occurrences)
    {
        if (occurrence.session->nextExecutionIndex == std::numeric_limits<uint64_t>::max())
        {
            occurrence.session->executionIndexExhausted = true;
        }
        else
        {
            ++occurrence.session->nextExecutionIndex;
        }
        if (!occurrence.dispatchId.has_value())
        {
            continue;
        }
        occurrence.snapshot->executionIndex = occurrence.executedIndex;
        occurrence.snapshot->dispatchId = occurrence.dispatchId;
        occurrence.sourceSnapshot->executionIndex = occurrence.executedIndex;
        occurrence.sourceSnapshot->dispatchId = occurrence.dispatchId;
        dispatchIds.commit();
    }
    plan.accepted = true;
}

void finish(Plan &plan, const char *captureFailure) noexcept
{
    releaseReservations(plan, captureFailure);
}
} // namespace capture::submission
