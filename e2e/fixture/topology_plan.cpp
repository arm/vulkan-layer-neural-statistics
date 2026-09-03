/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "topology_plan.hpp"

#include <algorithm>

namespace fixture
{

VkCommandBufferUsageFlags commandBufferUsage(RecordingUse use)
{
    switch (use)
    {
    case RecordingUse::oneShot:
        return VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    case RecordingUse::completedSequentialReuse:
        return 0;
    case RecordingUse::simultaneousOverlap:
        return VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    }
    return 0;
}

OrdinaryRecordingPlan ordinaryRecordingPlan(const Options &options)
{
    const auto submissionCount = std::max(options.resubmissions, options.dispatchRepeats);
    const auto primary = options.primaryCommandBuffersPerSubmit > 1
                             ? RecordingUse::simultaneousOverlap
                         : submissionCount > 1 ? RecordingUse::completedSequentialReuse
                                               : RecordingUse::oneShot;
    const auto secondary = options.secondaryExecutions > 1 ? RecordingUse::simultaneousOverlap
                           : submissionCount > 1           ? RecordingUse::completedSequentialReuse
                                                           : RecordingUse::oneShot;
    return {primary, secondary};
}

} // namespace fixture
