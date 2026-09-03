/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "dispatch_filter.hpp"

#include <charconv>
#include <limits>

namespace
{
bool IsAsciiWhitespace(char value) noexcept
{
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
           value == '\v';
}

std::string_view Trim(std::string_view text) noexcept
{
    while (!text.empty() && IsAsciiWhitespace(text.front()))
    {
        text.remove_prefix(1);
    }
    while (!text.empty() && IsAsciiWhitespace(text.back()))
    {
        text.remove_suffix(1);
    }
    return text;
}

std::optional<uint64_t> ParseIndex(std::string_view text) noexcept
{
    if (text.empty())
    {
        return std::nullopt;
    }
    uint64_t value = 0;
    const char *begin = text.data();
    const char *end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);
    if (result.ec != std::errc{} || result.ptr != end)
    {
        return std::nullopt;
    }
    return value;
}
} // namespace

DispatchFilter::DispatchFilter(bool captureAll, uint64_t firstIndex, uint64_t lastIndex) noexcept
    : captureAll_(captureAll), first_(firstIndex), last_(lastIndex)
{
}

DispatchFilter DispatchFilter::All() noexcept
{
    return DispatchFilter(true, 0, std::numeric_limits<uint64_t>::max());
}

DispatchFilter DispatchFilter::Single(uint64_t index) noexcept
{
    return DispatchFilter(false, index, index);
}

DispatchFilter DispatchFilter::Range(uint64_t first, uint64_t last) noexcept
{
    return DispatchFilter(false, first, last);
}

bool DispatchFilter::matches(uint64_t executedDispatchIndex) const noexcept
{
    return captureAll_ || (executedDispatchIndex >= first_ && executedDispatchIndex <= last_);
}

bool DispatchFilter::capturesAll() const noexcept
{
    return captureAll_;
}

uint64_t DispatchFilter::first() const noexcept
{
    return first_;
}

uint64_t DispatchFilter::last() const noexcept
{
    return last_;
}

std::string DispatchFilter::canonicalText() const
{
    if (captureAll_)
    {
        return {};
    }
    if (first_ == last_)
    {
        return std::to_string(first_);
    }
    return std::to_string(first_) + "-" + std::to_string(last_);
}

DispatchFilterParseResult ParseDispatchFilter(std::string_view input)
{
    const std::string_view text = Trim(input);
    if (text.empty())
    {
        return {DispatchFilter::All(), {}};
    }

    for (const char value : text)
    {
        if (IsAsciiWhitespace(value))
        {
            return {std::nullopt,
                    "dispatch filter permits whitespace only around the complete expression"};
        }
    }

    const size_t separator = text.find('-');
    if (separator == std::string_view::npos)
    {
        const auto index = ParseIndex(text);
        if (!index.has_value())
        {
            return {
                std::nullopt,
                "dispatch filter must be a non-negative integer, an inclusive N-M range, or blank"};
        }
        return {DispatchFilter::Single(*index), {}};
    }

    if (text.find('-', separator + 1) != std::string_view::npos)
    {
        return {std::nullopt, "dispatch filter contains more than one range separator"};
    }

    const auto first = ParseIndex(text.substr(0, separator));
    const auto last = ParseIndex(text.substr(separator + 1));
    if (!first.has_value() || !last.has_value())
    {
        return {std::nullopt, "dispatch filter range endpoints must be non-negative integers"};
    }
    if (*first > *last)
    {
        return {std::nullopt, "dispatch filter range start must not exceed its end"};
    }
    return {DispatchFilter::Range(*first, *last), {}};
}
