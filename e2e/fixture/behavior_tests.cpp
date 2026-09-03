/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "events.hpp"
#include "topology_plan.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

void expect(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::size_t occurrences(const std::string &text, const std::string &needle)
{
    std::size_t count = 0;
    for (std::size_t offset = 0; (offset = text.find(needle, offset)) != std::string::npos;
         offset += needle.size())
        ++count;
    return count;
}

} // namespace

int main()
{
    fixture::Options options;
    auto plan = fixture::ordinaryRecordingPlan(options);
    expect(fixture::commandBufferUsage(plan.primary) == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
           "an ordinary one-shot primary must remain one-time-submit");

    options.resubmissions = 2;
    plan = fixture::ordinaryRecordingPlan(options);
    expect(fixture::commandBufferUsage(plan.primary) == 0,
           "a completed sequentially reused primary must not be one-time-submit");

    options = {};
    options.primaryCommandBuffersPerSubmit = 2;
    plan = fixture::ordinaryRecordingPlan(options);
    expect(fixture::commandBufferUsage(plan.primary) ==
               VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
           "a duplicated primary in one submit must permit simultaneous overlap");

    options = {};
    options.secondaryExecutions = 2;
    plan = fixture::ordinaryRecordingPlan(options);
    expect(fixture::commandBufferUsage(plan.secondary) ==
               VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
           "a duplicated secondary execution must permit simultaneous overlap");

    std::ostringstream output;
    auto *previous = std::cout.rdbuf(output.rdbuf());
    fixture::emitSecondaryWrapperRecorded(2);
    fixture::emitApplicationFenceComplete(7);
    std::cout.rdbuf(previous);
    const auto events = output.str();
    expect(occurrences(events, "\"event\":\"secondary_wrapper_recorded\"") == 1,
           "one wrapper recording must emit exactly one wrapper event");
    expect(occurrences(events, "\"event\":\"application_fence_complete\"") == 1,
           "one completed fence must emit exactly one completion event");
    expect(events.find("\"result\":2") != std::string::npos,
           "wrapper event must report the actual secondary execution count");
    expect(events.find("\"submission\":7") != std::string::npos,
           "fence event must report the actual submission");
    return 0;
}
