/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_namespace.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace capture::names
{
namespace
{
bool IsAsciiAlphaNumeric(char value) noexcept
{
    const auto byte = static_cast<unsigned char>(value);
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9');
}

bool StartsWith(std::string_view value, std::string_view prefix) noexcept
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view value, std::string_view suffix) noexcept
{
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool IsIndexedDirectory(std::string_view key, std::string_view prefix) noexcept
{
    if (!StartsWith(key, prefix) || key.size() <= prefix.size() + 1 || key[prefix.size()] != '_')
    {
        return false;
    }
    return std::ranges::all_of(key.substr(prefix.size() + 1),
                               [](char value) { return value >= '0' && value <= '9'; });
}

bool ExtractPortableAscii(const std::filesystem::path &path, std::string &result)
{
    result = path.native();
    if (std::ranges::any_of(result,
                            [](char value) { return static_cast<unsigned char>(value) > 0x7f; }))
    {
        return false;
    }
    return true;
}
} // namespace

std::string PortableNameKey(std::string_view asciiName)
{
    return std::string(asciiName);
}

std::string MakeIndexedDirectoryName(std::string_view prefix, uint64_t id)
{
    std::ostringstream stream;
    stream << prefix << '_' << std::setw(kDirectoryIndexWidth) << std::setfill('0') << id;
    return stream.str();
}

bool IsReservedCaptureName(std::string_view portableKey) noexcept
{
    if (portableKey == kCaptureDocument || portableKey == kPipelineDocument ||
        portableKey == kSessionDocument || portableKey == kDispatchDocument)
    {
        return true;
    }
    if (StartsWith(portableKey, kInternalTemporaryPrefix) ||
        EndsWith(portableKey, kReservedTemporarySuffix))
    {
        return true;
    }
    return IsIndexedDirectory(portableKey, kPipelineDirectoryPrefix) ||
           IsIndexedDirectory(portableKey, kSessionDirectoryPrefix) ||
           IsIndexedDirectory(portableKey, kDispatchDirectoryPrefix);
}

ArtifactNameValidation ValidateArtifactFileName(const std::filesystem::path &fileName)
{
    ArtifactNameValidation result;
    if (fileName.empty() || fileName.is_absolute() || fileName != fileName.filename())
    {
        result.error = "artifact name must be one non-empty relative file name";
        return result;
    }

    std::string asciiName;
    if (!ExtractPortableAscii(fileName, asciiName))
    {
        result.error = "artifact name must contain only portable ASCII characters";
        return result;
    }
    if (asciiName.empty() || asciiName.size() > kMaximumArtifactNameBytes)
    {
        result.error = "artifact name length is outside the portable 1-120 byte range";
        return result;
    }
    if (asciiName == "." || asciiName == "..")
    {
        result.error = "artifact name must not be . or ..";
        return result;
    }
    if (!IsAsciiAlphaNumeric(asciiName.front()))
    {
        result.error = "artifact name must begin with an ASCII letter or digit";
        return result;
    }
    for (const char value : asciiName)
    {
        if (!IsAsciiAlphaNumeric(value) && value != '.' && value != '_' && value != '-')
        {
            result.error = "artifact name may contain only ASCII letters, digits, period, "
                           "underscore, and hyphen";
            return result;
        }
    }
    if (asciiName.back() == '.')
    {
        result.error = "artifact name must not end with a period";
        return result;
    }

    result.collisionKey = PortableNameKey(asciiName);
    if (IsReservedCaptureName(result.collisionKey))
    {
        result.error =
            "artifact name collides with the capture metadata, directory, or internal namespace";
        result.collisionKey.clear();
        return result;
    }

    result.valid = true;
    return result;
}
} // namespace capture::names
