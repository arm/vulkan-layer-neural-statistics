/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "options.hpp"

#include <vulkan/vulkan.h>

namespace fixture
{

enum class RecordingUse
{
    oneShot,
    completedSequentialReuse,
    simultaneousOverlap,
};

struct OrdinaryRecordingPlan
{
    RecordingUse primary;
    RecordingUse secondary;
};

VkCommandBufferUsageFlags commandBufferUsage(RecordingUse use);
OrdinaryRecordingPlan ordinaryRecordingPlan(const Options &options);

} // namespace fixture
