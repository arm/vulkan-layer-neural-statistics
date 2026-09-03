/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "events.hpp"
#include "options.hpp"
#include "vulkan_app.hpp"
#include <exception>
#include <iostream>
int main(int argc, char **argv)
{
    fixture::Options options;
    try
    {
        options = fixture::parseOptions(argc, argv);
    }
    catch (const fixture::OptionError &error)
    {
        std::cerr << "fixture option error: " << error.what() << '\n';
        return 2;
    }
    const std::int64_t started = fixture::nowNs();
    fixture::emitEvent("fixture_process_start");
    int exitCode = 0;
    try
    {
        fixture::VulkanApp app(options);
        app.run();
    }
    catch (const std::exception &error)
    {
        std::cerr << "fixture error: " << error.what() << '\n';
        exitCode = 255;
    }
    const std::int64_t ended = fixture::nowNs();
    fixture::emitEvent("fixture_process_end", 0, 0, exitCode);
    try
    {
        fixture::writeEventsFile(options.eventsPath, started, ended, exitCode);
    }
    catch (const std::exception &error)
    {
        std::cerr << "fixture events error: " << error.what() << '\n';
        return 255;
    }
    return exitCode;
}
