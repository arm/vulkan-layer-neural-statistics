/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "layer_setting_resolver.hpp"

#include "khronos/vulkan-utilities/src/layer/layer_settings_manager.hpp"
#include "khronos/vulkan-utilities/src/layer/layer_settings_util.hpp"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace
{
// Vulkan-Utilities does not expose source-specific presence queries through the
// public VkuLayerSettingSet API, which is required to distinguish absent values
// from explicitly empty environment/file values. This is the only bounded
// dependency on the vendored implementation contract: the handle created by
// this repository's pinned vkuCreateLayerSettingSet implementation owns a
// vl::LayerSettings object. No other layer code may inspect the opaque handle.
vl::LayerSettings &VendoredLayerSettings(VkuLayerSettingSet layerSettingSet) noexcept
{
    return *reinterpret_cast<vl::LayerSettings *>(layerSettingSet);
}

std::optional<std::string> ReadEnvironment(const std::string &name)
{
#if defined(__ANDROID__)
    const prop_info *property = __system_property_find(name.c_str());
    if (property == nullptr)
    {
        return std::nullopt;
    }

    std::string value;
#if __ANDROID_API__ >= 26
    __system_property_read_callback(
        property, [](void *cookie, const char *, const char *propertyValue, uint32_t)
        { static_cast<std::string *>(cookie)->assign(propertyValue); }, &value);
#else
    char buffer[PROP_VALUE_MAX] = {};
    __system_property_get(name.c_str(), buffer);
    value = buffer;
#endif
    return value;
#else
    const char *value = std::getenv(name.c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
#endif
}

std::optional<std::string> ReadEnvironmentSetting(const char *layerName,
                                                  const char *compatibilityNamespace,
                                                  const char *settingName)
{
    if (compatibilityNamespace != nullptr && compatibilityNamespace[0] != '\0')
    {
        const auto name = vl::GetEnvSettingName(layerName, compatibilityNamespace, settingName,
                                                vl::TRIM_NAMESPACE);
        if (auto value = ReadEnvironment(name))
        {
            return value;
        }
    }

    for (int trim = vl::TRIM_FIRST; trim <= vl::TRIM_LAST; ++trim)
    {
        const auto name =
            vl::GetEnvSettingName(layerName, nullptr, settingName, static_cast<vl::TrimMode>(trim));
        if (auto value = ReadEnvironment(name))
        {
            return value;
        }
    }
    return std::nullopt;
}
} // namespace

LayerSettingValue SelectLayerSettingValue(std::optional<std::string> environment,
                                          std::optional<std::string> file,
                                          const LayerSettingValue &api)
{
    if (environment)
    {
        return {LayerSettingSource::Environment, VK_SUCCESS, std::move(*environment)};
    }
    if (file)
    {
        return {LayerSettingSource::File, VK_SUCCESS, std::move(*file)};
    }
    return api;
}

LayerSettingValue ResolveLayerSettingValue(VkuLayerSettingSet layerSettingSet,
                                           const char *layerName,
                                           const char *compatibilityNamespace,
                                           const char *settingName)
{
    auto &settings = VendoredLayerSettings(layerSettingSet);

    std::optional<std::string> file;
    if (settings.HasFileSetting(settingName))
    {
        file = settings.GetFileSetting(settingName);
    }

    LayerSettingValue api;
    if (settings.HasAPISetting(settingName))
    {
        api.source = LayerSettingSource::Api;
        api.result = vkuGetLayerSettingValue(layerSettingSet, settingName, api.value);
    }

    return SelectLayerSettingValue(
        ReadEnvironmentSetting(layerName, compatibilityNamespace, settingName), std::move(file),
        std::move(api));
}
