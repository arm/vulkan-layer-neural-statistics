/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "submission_planner.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
int failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Handle> Handle Fake(uintptr_t value)
{
    return reinterpret_cast<Handle>(value);
}

std::shared_ptr<PipelineRecord> Pipeline(uint64_t id = 0)
{
    return std::make_shared<PipelineRecord>(Fake<VkPipeline>(0x1000 + id), capture::PipelineId(id),
                                            capture::PipelineMetadata{}, nullptr);
}

std::shared_ptr<SessionRecord> Session(uint64_t id)
{
    auto session = std::make_shared<SessionRecord>(Fake<VkDataGraphPipelineSessionARM>(0x2000 + id),
                                                   Pipeline(id));
    session->id = capture::SessionId(id);
    session->statisticsCaptureActive = true;
    return session;
}

std::shared_ptr<StatsSnapshot> Snapshot(const std::shared_ptr<SessionRecord> &session,
                                        uint32_t family = 0)
{
    auto snapshot = std::make_shared<StatsSnapshot>(VK_NULL_HANDLE, nullptr, nullptr);
    snapshot->session = session;
    snapshot->queueFamilyIndex = family;
    return snapshot;
}

capture::submission::SubmittedOccurrence Occurrence(const std::shared_ptr<SessionRecord> &session,
                                                    const std::shared_ptr<StatsSnapshot> &snapshot,
                                                    std::string reason = {},
                                                    bool protectedSubmit = false)
{
    return {{session, snapshot, std::move(reason)}, protectedSubmit};
}

bool Build(const DispatchFilter &filter, capture::IdAllocator<capture::DispatchId> &ids,
           const std::vector<capture::submission::SubmittedOccurrence> &submitted,
           capture::submission::Plan &plan, std::string &reason, bool enabled = true)
{
    if (capture::submission::build(filter, enabled, ids, submitted, plan, reason) !=
        capture::submission::BuildResult::Ready)
    {
        return false;
    }
    std::vector<std::shared_ptr<StatsSnapshot>> destinations;
    for (const auto &occurrence : plan.occurrences)
    {
        if (occurrence.dispatchId.has_value())
        {
            destinations.emplace_back(
                Snapshot(occurrence.session, occurrence.sourceSnapshot->queueFamilyIndex));
        }
    }
    return capture::submission::materialize(plan, std::move(destinations), reason);
}

void TestRollbackAndCommit()
{
    auto session = Session(0);
    auto snapshot = Snapshot(session);
    capture::IdAllocator<capture::DispatchId> ids;
    capture::submission::Plan plan;
    std::string reason;

    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, snapshot)}, plan, reason),
          "selected occurrence plans successfully");
    Check(plan.selectedSnapshots[0]->inFlight && snapshot->inFlightUses == 1,
          "planning reserves a unique destination and tracks its recorded source");
    capture::submission::cancel(plan);
    Check(session->nextExecutionIndex == 0 && ids.nextValue() == 0,
          "downstream failure advances neither executed index nor dispatch ID");
    Check(!snapshot->inFlight, "cancellation releases the destination reservation");

    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, snapshot)}, plan, reason),
          "cancelled exact submission can be replanned");
    capture::submission::accept(ids, plan);
    capture::submission::accept(ids, plan);
    Check(session->nextExecutionIndex == 1 && ids.nextValue() == 1,
          "accepted submission commits every sequence exactly once");
    Check(plan.selectedSnapshots[0]->executionIndex == 0 &&
              plan.selectedSnapshots[0]->dispatchId == capture::DispatchId(0),
          "accepted submission destination receives the reserved index and ID");
    capture::submission::finish(plan, nullptr);

    Check(Build(DispatchFilter::Single(99), ids, {Occurrence(session, snapshot)}, plan, reason),
          "unselected occurrence remains transparent");
    capture::submission::accept(ids, plan);
    capture::submission::finish(plan, nullptr);
    Check(session->nextExecutionIndex == 2 && ids.nextValue() == 1,
          "unselected occurrence commits its executed index without consuming a dispatch ID");
}

void TestUniqueSubmissionSnapshots()
{
    auto session = Session(1);
    auto recordedSnapshot = Snapshot(session);
    capture::IdAllocator<capture::DispatchId> ids;
    capture::submission::Plan firstPlan;
    std::string reason;

    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, recordedSnapshot)}, firstPlan,
                reason),
          "first selected occurrence receives a submission-owned destination");
    auto firstDestination = firstPlan.selectedSnapshots.front();
    capture::submission::accept(ids, firstPlan);

    capture::submission::Plan overlappingPlan;
    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, recordedSnapshot)},
                overlappingPlan, reason),
          "overlapping reuse receives another submission-owned destination");
    Check(overlappingPlan.selectedSnapshots.front() != firstDestination &&
              recordedSnapshot != firstDestination && recordedSnapshot->inFlightUses == 2,
          "record-time source tracks two independent submission destinations");
    capture::submission::cancel(overlappingPlan);
    capture::submission::finish(firstPlan, nullptr);
}

void TestRepeatedDestinations()
{
    auto session = Session(2);
    auto recordedDestination = Snapshot(session);
    capture::IdAllocator<capture::DispatchId> ids;
    capture::submission::Plan plan;
    std::string reason;
    Check(
        Build(DispatchFilter::All(), ids,
              {Occurrence(session, recordedDestination), Occurrence(session, recordedDestination)},
              plan, reason),
        "capture-all accepts repeated record-time destinations");
    Check(plan.selectedSnapshots.size() == 2 &&
              plan.selectedSnapshots[0] != plan.selectedSnapshots[1] &&
              plan.occurrences[0].sourceSnapshot == recordedDestination &&
              plan.occurrences[1].sourceSnapshot == recordedDestination,
          "every repeated occurrence receives a unique submission destination");
    capture::submission::accept(ids, plan);
    Check(session->nextExecutionIndex == 2 && plan.selectedSnapshots[0]->executionIndex == 0 &&
              plan.selectedSnapshots[1]->executionIndex == 1,
          "repeated occurrences retain deterministic execution identities");
    capture::submission::finish(plan, nullptr);
}

void TestOrderingAndRoutes()
{
    for (int route = 0; route < 3; ++route)
    {
        auto a = Session(10 + route * 2);
        auto b = Session(11 + route * 2);
        auto a0 = Snapshot(a);
        auto b0 = Snapshot(b);
        auto a1 = Snapshot(a);
        capture::IdAllocator<capture::DispatchId> ids;
        capture::submission::Plan plan;
        std::string reason;
        const std::vector<capture::submission::SubmittedOccurrence> submitted{
            Occurrence(a, a0), Occurrence(b, b0), Occurrence(a, a1)};
        Check(Build(DispatchFilter::All(), ids, submitted, plan, reason),
              "legacy/core/KHR routes share successful production planning");
        capture::submission::accept(ids, plan);
        Check(plan.occurrences[0].executedIndex == 0 && plan.occurrences[1].executedIndex == 0 &&
                  plan.occurrences[2].executedIndex == 1,
              "primary/secondary/composite order is preserved per session");
        Check(plan.occurrences[0].dispatchId == capture::DispatchId(0) &&
                  plan.occurrences[1].dispatchId == capture::DispatchId(1) &&
                  plan.occurrences[2].dispatchId == capture::DispatchId(2),
              "capture-wide dispatch IDs follow deterministic submitted occurrence order");
        capture::submission::finish(plan, nullptr);
    }
}

void TestPolicyReasonsAndCrossFamilyReuse()
{
    const struct Case
    {
        const char *reason;
        bool protectedSubmit;
    } cases[]{
        {"selected capture rejects protected data-graph execution", false},
        {"selected capture rejects foreign processing engines", false},
        {"selected capture requires a transfer-capable queue family", false},
        {"selected capture mechanism is unavailable for this session", false},
        {"", true},
    };
    for (const auto &item : cases)
    {
        auto session = Session(30);
        auto snapshot = Snapshot(session, 2);
        capture::IdAllocator<capture::DispatchId> ids;
        capture::submission::Plan plan;
        std::string reason;
        Check(!Build(DispatchFilter::All(), ids,
                     {Occurrence(session, snapshot, item.reason, item.protectedSubmit)}, plan,
                     reason),
              "selected protected/foreign/transfer/unsupported occurrence rejects pre-forward");
        Check(Build(DispatchFilter::Single(9), ids,
                    {Occurrence(session, snapshot, item.reason, item.protectedSubmit)}, plan,
                    reason),
              "the same unsupported occurrence is transparent when unselected");
        capture::submission::accept(ids, plan);
        capture::submission::finish(plan, nullptr);
    }

    auto session = Session(31);
    auto firstFamily = Snapshot(session, 4);
    auto secondFamily = Snapshot(session, 5);
    capture::IdAllocator<capture::DispatchId> ids;
    capture::submission::Plan plan;
    std::string reason;
    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, firstFamily)}, plan, reason),
          "selected capture accepts the session on its first queue family");
    capture::submission::accept(ids, plan);
    capture::submission::finish(plan, nullptr);
    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, secondFamily)}, plan, reason),
          "selected capture accepts later reuse of the session on another queue family");
    capture::submission::accept(ids, plan);
    Check(plan.occurrences[0].executedIndex == 1 &&
              plan.occurrences[0].dispatchId == capture::DispatchId(1),
          "cross-family reuse preserves per-session execution and capture-wide dispatch ordering");
    capture::submission::finish(plan, nullptr);
}

void TestCommandBufferExpansion()
{
    auto session = Session(40);
    auto first = Snapshot(session);
    auto second = Snapshot(session);
    auto command = std::make_shared<CommandBufferRecord>(Fake<VkCommandBuffer>(0x9000),
                                                         Fake<VkCommandPool>(0x9100), 0, false);
    command->dispatches.push_back({session, first, {}});
    command->dispatches.push_back({session, second, {}});
    std::vector<capture::submission::SubmittedOccurrence> submitted;
    std::string reason;
    Check(
        capture::submission::appendCommandBufferOccurrences(command, false, submitted, reason) &&
            capture::submission::appendCommandBufferOccurrences(command, false, submitted, reason),
        "multiple submit records and repeated primary expand through production code");
    Check(submitted.size() == 4 && submitted[0].occurrence.snapshot == first &&
              submitted[1].occurrence.snapshot == second &&
              submitted[2].occurrence.snapshot == first,
          "submit-record and repeated-primary expansion preserves exact order");
}

void TestConcurrentAdmission()
{
    capture::IdAllocator<capture::DispatchId> ids;
    std::mutex admission;
    auto a = Session(50);
    auto b = Session(51);
    auto aSnapshot = Snapshot(a);
    auto bSnapshot = Snapshot(b);
    std::atomic<bool> ok{true};
    std::vector<uint64_t> assigned;
    std::mutex assignedMutex;

    auto runSelected = [&](const std::shared_ptr<SessionRecord> &session,
                           const std::shared_ptr<StatsSnapshot> &snapshot)
    {
        std::lock_guard<std::mutex> guard{admission};
        capture::submission::Plan plan;
        std::string reason;
        if (!Build(DispatchFilter::All(), ids, {Occurrence(session, snapshot)}, plan, reason))
        {
            ok = false;
            return;
        }
        capture::submission::accept(ids, plan);
        {
            std::lock_guard<std::mutex> output{assignedMutex};
            assigned.push_back(plan.occurrences[0].dispatchId->value());
        }
        capture::submission::finish(plan, nullptr);
    };

    std::thread t0(runSelected, a, aSnapshot);
    std::thread t1(runSelected, b, bSnapshot);
    t0.join();
    t1.join();
    Check(
        ok && assigned.size() == 2 && assigned[0] != assigned[1] && ids.nextValue() == 2,
        "concurrent queues/sessions receive unique capture-wide IDs under admission serialization");

    auto same = Session(52);
    std::atomic<uint64_t> first{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> second{std::numeric_limits<uint64_t>::max()};
    auto runUnselected = [&](std::atomic<uint64_t> &output)
    {
        std::lock_guard<std::mutex> guard{admission};
        capture::submission::Plan plan;
        std::string reason;
        if (!Build(DispatchFilter::Single(99), ids, {Occurrence(same, Snapshot(same))}, plan,
                   reason))
        {
            ok = false;
            return;
        }
        output = plan.occurrences[0].executedIndex;
        capture::submission::accept(ids, plan);
        capture::submission::finish(plan, nullptr);
    };
    std::thread t2(runUnselected, std::ref(first));
    std::thread t3(runUnselected, std::ref(second));
    t2.join();
    t3.join();
    Check(ok && first != second && same->nextExecutionIndex == 2 && ids.nextValue() == 2,
          "concurrent all-unselected submissions reserve unique per-session executed indices only");
}

void TestDisabledCaptureBypass()
{
    auto session = Session(60);
    auto snapshot = Snapshot(session);
    capture::IdAllocator<capture::DispatchId> ids;
    capture::submission::Plan plan;
    std::string reason;

    const auto result = capture::submission::build(DispatchFilter::All(),
                                                   false,
                                                   ids,
                                                   {Occurrence(session, snapshot, "stale capture failure", true)},
                                                   plan,
                                                   reason);
    Check(result == capture::submission::BuildResult::Ready && plan.occurrences.empty() && plan.selectedCount == 0
              && plan.selectedSnapshots.empty() && !plan.reserved,
          "disabled capture produces an empty uninstrumented submission plan");
    capture::submission::accept(ids, plan);
    Check(session->nextExecutionIndex == 0 && ids.nextValue() == 0 && snapshot->dispatchId == std::nullopt,
          "accepting a disabled-capture bypass consumes no capture bookkeeping");
}

void TestExhaustion()
{
    auto session = Session(60);
    session->nextExecutionIndex = std::numeric_limits<uint64_t>::max();
    auto snapshot = Snapshot(session);
    capture::IdAllocator<capture::DispatchId> ids(std::numeric_limits<uint64_t>::max());
    capture::submission::Plan plan;
    std::string reason;
    Check(Build(DispatchFilter::All(), ids, {Occurrence(session, snapshot)}, plan, reason),
          "last legal executed index and dispatch ID can be reserved atomically");
    capture::submission::accept(ids, plan);
    capture::submission::finish(plan, nullptr);
    Check(session->executionIndexExhausted && ids.exhausted(),
          "committing last values marks both sequences exhausted without wrapping");
    Check(!Build(DispatchFilter::All(), ids, {Occurrence(session, snapshot)}, plan, reason),
          "exhausted sequences reject before partial publication");
}
} // namespace

int main()
{
    TestRollbackAndCommit();
    TestUniqueSubmissionSnapshots();
    TestRepeatedDestinations();
    TestOrderingAndRoutes();
    TestPolicyReasonsAndCrossFamilyReuse();
    TestCommandBufferExpansion();
    TestConcurrentAdmission();
    TestDisabledCaptureBypass();
    TestExhaustion();

    if (failures != 0)
    {
        std::cerr << failures << " submission planner test(s) failed\n";
        return 1;
    }
    std::cout << "All submission planner tests passed\n";
    return 0;
}
