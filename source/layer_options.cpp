/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "layer_options.hpp"

#include "layer_setting_resolver.hpp"
#include "version.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string_view>
#include <utility>

#include <unistd.h>

namespace
{
std::string DefaultCaptureDirectoryName()
{
    static std::atomic<uint64_t> sequence{0};
    const auto now = std::chrono::system_clock::now();
    const auto epochMicroseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    const auto fraction = static_cast<uint64_t>(epochMicroseconds % 1'000'000);
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    (void)::gmtime_r(&seconds, &utc);

    char timestamp[32]{};
    (void)std::strftime(timestamp, sizeof(timestamp), "%Y%m%dT%H%M%S", &utc);
    char suffix[64]{};
    (void)std::snprintf(
        suffix, sizeof(suffix), ".%06lluZ-p%llu-%llu", static_cast<unsigned long long>(fraction),
        static_cast<unsigned long long>(::getpid()),
        static_cast<unsigned long long>(sequence.fetch_add(1, std::memory_order_relaxed)));
    return std::string("neural_statistics-") + timestamp + suffix;
}
} // namespace

LayerOptions::LayerOptions() : captureRoot_(getDefaultCaptureRoot()) {}

LayerOptions::LayerOptions(const VkInstanceCreateInfo *instCreateInfo) : LayerOptions()
{
    VkuLayerSettingSet layerSettingsSet = VK_NULL_HANDLE;
    const auto layerSettingsCreateInfo = vkuFindLayerSettingsCreateInfo(instCreateInfo);
    const VkResult createResult = vkuCreateLayerSettingSet(LGL_LAYER_NAME, layerSettingsCreateInfo,
                                                           nullptr, nullptr, &layerSettingsSet);
    if (createResult != VK_SUCCESS || layerSettingsSet == VK_NULL_HANDLE)
    {
        invalidate("failed to create the layer setting set");
        return;
    }
    vkuSetLayerSettingCompatibilityNamespace(layerSettingsSet, getLayerPrefix());

    const auto captureFolder = ResolveLayerSettingValue(layerSettingsSet, LGL_LAYER_NAME,
                                                        getLayerPrefix(), captureFolderKey);
    if (captureFolder.result != VK_SUCCESS)
    {
        invalidate("capture_folder must be a string path");
    }
    else if (captureFolder.present())
    {
        if (captureFolder.value.empty())
        {
            invalidate("capture_folder must identify the exact, non-empty capture root");
        }
        else
        {
            captureRoot_ = fs::path(captureFolder.value);
        }
    }

    const auto statisticsMode = ResolveLayerSettingValue(layerSettingsSet, LGL_LAYER_NAME,
                                                         getLayerPrefix(), statisticsModeKey);
    if (valid_ && statisticsMode.result != VK_SUCCESS)
    {
        invalidate("statistics_mode must be the enum value 0 or 1");
    }
    if (valid_)
    {
        const std::string_view value =
            statisticsMode.present() ? statisticsMode.value : std::string_view("0");
        if (value == "0")
        {
            statisticsModeIndex_ = 0;
            statsMode_ = VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS0_ARM;
        }
        else if (value == "1")
        {
            statisticsModeIndex_ = 1;
            statsMode_ = VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS1_ARM;
        }
        else
        {
            invalidate("statistics_mode must be exactly 0 or 1");
        }
    }

    const auto dispatchFilter = ResolveLayerSettingValue(layerSettingsSet, LGL_LAYER_NAME,
                                                         getLayerPrefix(), dispatchFilterKey);
    if (valid_ && dispatchFilter.result != VK_SUCCESS)
    {
        invalidate("dispatch_filter must be a string");
    }
    if (valid_)
    {
        const std::string_view value =
            dispatchFilter.present() ? dispatchFilter.value : std::string_view{};
        auto parsed = ParseDispatchFilter(value);
        if (!parsed)
        {
            invalidate("invalid dispatch_filter: " + parsed.error);
        }
        else
        {
            dispatchFilter_ = *parsed.filter;
            dispatchFilterText_ = dispatchFilter_.canonicalText();
        }
    }

    vkuDestroyLayerSettingSet(layerSettingsSet, nullptr);
}

bool LayerOptions::isValid() const noexcept
{
    return valid_;
}

const std::string &LayerOptions::getError() const noexcept
{
    return error_;
}

VkNeuralAcceleratorStatisticsModeARM LayerOptions::getStatsMode() const noexcept
{
    return statsMode_;
}

uint32_t LayerOptions::getStatsModeIndex() const noexcept
{
    return statisticsModeIndex_;
}

const fs::path &LayerOptions::getCaptureRoot() const noexcept
{
    return captureRoot_;
}

const DispatchFilter &LayerOptions::getDispatchFilter() const noexcept
{
    return dispatchFilter_;
}

const std::string &LayerOptions::getDispatchFilterText() const noexcept
{
    return dispatchFilterText_;
}

fs::path LayerOptions::getDefaultCaptureRoot()
{
#ifdef __ANDROID__
    return fs::path("/data/local/tmp") / DefaultCaptureDirectoryName();
#else
    return fs::current_path() / DefaultCaptureDirectoryName();
#endif
}

constexpr const char *LayerOptions::getLayerPrefix()
{
#ifdef __ANDROID__
    return "nxstats";
#else
    return "LAYER";
#endif
}

void LayerOptions::invalidate(std::string message)
{
    valid_ = false;
    error_ = std::move(message);
}
