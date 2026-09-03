/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.h>
namespace fixture
{
struct GraphConstantSpec
{
    VkFormat format;
    std::array<std::int64_t, 4> dimensions;
    std::uint32_t dimensionCount;
};
inline constexpr std::array<GraphConstantSpec, 45> graphConstantSpecs{{
    {VK_FORMAT_R8_SINT, {32, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {64, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {64, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {64, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {64, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 0, 0, 0}, 1},  {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1}, {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1}, {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1}, {VK_FORMAT_R8_SINT, {256, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 3, 3, 12}, 4}, {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 3, 3, 32}, 4}, {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 3, 3, 32}, 4}, {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 3, 3, 32}, 4}, {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {64, 3, 3, 32}, 4}, {VK_FORMAT_R32_SINT, {64, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {64, 3, 3, 64}, 4}, {VK_FORMAT_R32_SINT, {64, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {32, 3, 3, 64}, 4}, {VK_FORMAT_R32_SINT, {32, 0, 0, 0}, 1},
    {VK_FORMAT_R8_SINT, {16, 3, 3, 32}, 4}, {VK_FORMAT_R8_SINT, {16, 3, 3, 48}, 4},
    {VK_FORMAT_R8_SINT, {16, 3, 3, 16}, 4}, {VK_FORMAT_R8_SINT, {16, 3, 3, 48}, 4},
    {VK_FORMAT_R8_SINT, {16, 3, 3, 16}, 4}, {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},
    {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},  {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},
    {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},  {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},
    {VK_FORMAT_R8_SINT, {4, 3, 3, 16}, 4},
}};
inline constexpr std::array<std::int64_t, 4> inputShape{2, 12, 64, 64};
inline constexpr std::array<std::int64_t, 4> outputShape{2, 4, 64, 64};
inline constexpr VkFormat tensorFormat = VK_FORMAT_R8_SINT;
inline constexpr const char *graphEntryPoint = "graph_partition_0";
std::size_t graphConstantByteSize(const GraphConstantSpec &spec);
} // namespace fixture
