/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

class DispatchFilter
{
  public:
    static DispatchFilter All() noexcept;
    static DispatchFilter Single(uint64_t index) noexcept;
    static DispatchFilter Range(uint64_t first, uint64_t last) noexcept;

    bool matches(uint64_t executedDispatchIndex) const noexcept;
    bool capturesAll() const noexcept;
    uint64_t first() const noexcept;
    uint64_t last() const noexcept;
    std::string canonicalText() const;

  private:
    DispatchFilter(bool captureAll, uint64_t firstIndex, uint64_t lastIndex) noexcept;

    bool captureAll_{true};
    uint64_t first_{0};
    uint64_t last_{0};
};

struct DispatchFilterParseResult
{
    std::optional<DispatchFilter> filter;
    std::string error;

    explicit operator bool() const noexcept
    {
        return filter.has_value();
    }
};

// Grammar: blank, N, or N-M. Only surrounding ASCII whitespace is accepted.
DispatchFilterParseResult ParseDispatchFilter(std::string_view text);
