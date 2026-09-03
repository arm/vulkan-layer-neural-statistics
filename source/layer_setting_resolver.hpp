/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <optional>
#include <string>

#include <vulkan/layer/vk_layer_settings.hpp>
#include <vulkan/vulkan_core.h>

enum class LayerSettingSource
{
    Absent,
    Environment,
    File,
    Api,
};

struct LayerSettingValue
{
    LayerSettingSource source{LayerSettingSource::Absent};
    VkResult result{VK_SUCCESS};
    std::string value;

    bool present() const noexcept
    {
        return source != LayerSettingSource::Absent;
    }
};

LayerSettingValue SelectLayerSettingValue(std::optional<std::string> environment,
                                          std::optional<std::string> file,
                                          const LayerSettingValue &api);

LayerSettingValue ResolveLayerSettingValue(VkuLayerSettingSet layerSettingSet,
                                           const char *layerName,
                                           const char *compatibilityNamespace,
                                           const char *settingName);
