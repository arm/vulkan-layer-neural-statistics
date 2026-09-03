/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "dispatch_filter.hpp"

#include <filesystem>
#include <string>

#include <vulkan/layer/vk_layer_settings.hpp>
#include <vulkan/vulkan_core.h>

namespace fs = std::filesystem;

class LayerOptions
{
  public:
    LayerOptions();
    explicit LayerOptions(const VkInstanceCreateInfo *instCreateInfo);
    ~LayerOptions() = default;

    bool isValid() const noexcept;
    const std::string &getError() const noexcept;
    VkNeuralAcceleratorStatisticsModeARM getStatsMode() const noexcept;
    uint32_t getStatsModeIndex() const noexcept;
    const fs::path &getCaptureRoot() const noexcept;
    const DispatchFilter &getDispatchFilter() const noexcept;
    const std::string &getDispatchFilterText() const noexcept;

  private:
    static fs::path getDefaultCaptureRoot();
    static constexpr const char *getLayerPrefix();
    void invalidate(std::string message);

    inline static constexpr const char *captureFolderKey = "capture_folder";
    inline static constexpr const char *statisticsModeKey = "statistics_mode";
    inline static constexpr const char *dispatchFilterKey = "dispatch_filter";

    bool valid_{true};
    std::string error_;
    uint32_t statisticsModeIndex_{0};
    VkNeuralAcceleratorStatisticsModeARM statsMode_{
        VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS0_ARM};
    fs::path captureRoot_;
    DispatchFilter dispatchFilter_{DispatchFilter::All()};
    std::string dispatchFilterText_;
};
