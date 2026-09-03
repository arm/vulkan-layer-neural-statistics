/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "options.hpp"
#include <cstdlib>
#include <iostream>
#include <string_view>
namespace fixture
{
namespace
{
unsigned parseUnsigned(std::string_view text, std::string_view option, unsigned minimum,
                       unsigned maximum)
{
    if (text.empty())
        throw OptionError(std::string(option) + " requires an integer");
    unsigned long parsed = 0;
    try
    {
        std::size_t used = 0;
        parsed = std::stoul(std::string(text), &used, 10);
        if (used != text.size())
            throw OptionError("trailing characters");
    }
    catch (const OptionError &)
    {
        throw;
    }
    catch (const std::exception &)
    {
        throw OptionError(std::string(option) + " has invalid integer value " + std::string(text));
    }
    if (parsed < minimum || parsed > maximum)
        throw OptionError(std::string(option) + " is outside the supported range");
    return static_cast<unsigned>(parsed);
}
} // namespace
[[noreturn]] void printUsageAndExit(const char *program, int code)
{
    std::cerr << "usage: " << program << " [options]\n"
              << "  --graph-count 1|2|3 --secondary-executions N --resubmissions N\n"
              << "  --dispatch-repeats N --primary-command-buffers-per-submit N\n"
              << "  --distinct-primary-submissions N --distinct-primary-stage-semaphore\n"
              << "  --second-dispatch-conditional-false\n"
              << "  --queue-count 1|2 --device-count 1|2 --submit-route legacy|core|khr\n"
              << "  --immediate-resubmission --noncapturable --destroy-after-fence\n"
              << "  --reuse-retry-ms N --reuse-timeout-ms N --capacity-submissions 1|3\n"
              << "  --spirv-reference PATH --events PATH\n";
    std::exit(code);
}
Options parseOptions(int argc, char **argv)
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option(argv[index]);
        auto value = [&]() -> std::string_view
        {
            if (++index >= argc)
                throw OptionError(std::string(option) + " requires a value");
            return argv[index];
        };
        if (option == "--events")
            result.eventsPath = value();
        else if (option == "--spirv-reference")
            result.spirvReferencePath = value();
        else if (option == "--graph-count")
            result.graphCount = parseUnsigned(value(), option, 1, 3);
        else if (option == "--secondary-executions")
            result.secondaryExecutions = parseUnsigned(value(), option, 0, 4);
        else if (option == "--resubmissions")
            result.resubmissions = parseUnsigned(value(), option, 1, 256);
        else if (option == "--dispatch-repeats")
            result.dispatchRepeats = parseUnsigned(value(), option, 1, 256);
        else if (option == "--primary-command-buffers-per-submit")
            result.primaryCommandBuffersPerSubmit = parseUnsigned(value(), option, 1, 4);
        else if (option == "--distinct-primary-submissions")
            result.distinctPrimarySubmissions = parseUnsigned(value(), option, 1, 4);
        else if (option == "--queue-count")
            result.queueCount = parseUnsigned(value(), option, 1, 2);
        else if (option == "--device-count")
            result.deviceCount = parseUnsigned(value(), option, 1, 2);
        else if (option == "--reuse-retry-ms")
            result.reuseRetryMs = parseUnsigned(value(), option, 1, 60000);
        else if (option == "--reuse-timeout-ms")
            result.reuseTimeoutMs = parseUnsigned(value(), option, 1, 600000);
        else if (option == "--capacity-submissions")
            result.capacitySubmissions = parseUnsigned(value(), option, 1, 3);
        else if (option == "--distinct-primary-stage-semaphore")
            result.distinctPrimaryStageSemaphore = true;
        else if (option == "--second-dispatch-conditional-false")
            result.secondDispatchConditionalFalse = true;
        else if (option == "--immediate-resubmission")
            result.immediateResubmission = true;
        else if (option == "--noncapturable")
            result.noncapturable = true;
        else if (option == "--destroy-after-fence")
            result.destroyAfterFence = true;
        else if (option == "--submit-route")
            result.submitRoute = value();
        else if (option == "--help")
            printUsageAndExit(argv[0], 0);
        else
            throw OptionError("unknown option " + std::string(option));
    }
    if (result.submitRoute != "legacy" && result.submitRoute != "core" &&
        result.submitRoute != "khr")
        throw OptionError("--submit-route must be legacy, core, or khr");
    if (result.capacitySubmissions != 1 && result.capacitySubmissions != 3)
        throw OptionError("capacity turnover requires exactly three independent submissions");
    if (result.capacitySubmissions == 3 && result.graphCount != 3)
        throw OptionError("--capacity-submissions 3 requires --graph-count 3");
    if (result.graphCount > 1 && result.capacitySubmissions == 1 &&
        (result.secondaryExecutions != 0 || result.resubmissions != 1 ||
         result.dispatchRepeats != 1 || result.primaryCommandBuffersPerSubmit != 1 ||
         result.distinctPrimarySubmissions != 1 || result.destroyAfterFence))
    {
        throw OptionError("multi-graph mode supports only queue and submit-route controls");
    }
    if (result.dispatchRepeats > 1 && result.resubmissions > 1)
        throw OptionError("--dispatch-repeats and --resubmissions cannot both exceed one");
    if (result.immediateResubmission && result.dispatchRepeats == 1)
        throw OptionError("--immediate-resubmission requires --dispatch-repeats greater than one");
    if (result.distinctPrimaryStageSemaphore &&
        (result.distinctPrimarySubmissions != 2 || result.queueCount != 2 ||
         result.submitRoute == "legacy"))
        throw OptionError("distinct-primary stage semaphore requires two distinct submissions, two "
                          "queues, and core or KHR submit2");
    if (result.secondDispatchConditionalFalse && !result.distinctPrimaryStageSemaphore)
        throw OptionError("conditional-false second dispatch requires the distinct-primary "
                          "stage-semaphore topology");
    if (result.distinctPrimarySubmissions > 1 &&
        (result.graphCount != 1 || result.secondaryExecutions != 0 ||
         result.capacitySubmissions != 1 || result.resubmissions != 1 ||
         result.dispatchRepeats != 1 || result.primaryCommandBuffersPerSubmit != 1))
        throw OptionError(
            "distinct-primary mode requires one graph and no other command-buffer topology mode");
    if (result.primaryCommandBuffersPerSubmit > 1 &&
        (result.secondaryExecutions != 0 || result.resubmissions != 1 ||
         result.dispatchRepeats != 1 || result.graphCount != 1))
        throw OptionError(
            "duplicate-primary mode requires one graph, one submission, and no secondary wrapper");
    if (result.destroyAfterFence &&
        (result.graphCount != 1 || result.secondaryExecutions != 0 || result.resubmissions != 1 ||
         result.dispatchRepeats != 1 || result.distinctPrimarySubmissions != 1))
        throw OptionError(
            "pending-work destruction requires one graph and one ordinary primary submission");
    return result;
}
} // namespace fixture
