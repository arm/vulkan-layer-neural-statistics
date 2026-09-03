/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <stdexcept>
#include <string>
namespace fixture
{
struct OptionError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
struct Options
{
    std::string eventsPath;
    std::string spirvReferencePath;
    unsigned graphCount{1};
    unsigned secondaryExecutions{0};
    unsigned resubmissions{1};
    unsigned dispatchRepeats{1};
    unsigned primaryCommandBuffersPerSubmit{1};
    unsigned distinctPrimarySubmissions{1};
    unsigned queueCount{1};
    unsigned deviceCount{1};
    bool distinctPrimaryStageSemaphore{false};
    bool secondDispatchConditionalFalse{false};
    bool immediateResubmission{false};
    unsigned reuseRetryMs{10};
    unsigned reuseTimeoutMs{5000};
    unsigned capacitySubmissions{1};
    std::string submitRoute{"legacy"};
    bool noncapturable{false};
    bool destroyAfterFence{false};
};
Options parseOptions(int argc, char **argv);
[[noreturn]] void printUsageAndExit(const char *program, int code);
} // namespace fixture
