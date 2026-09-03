# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(VkLayerNeuralStatistics VERSION 1.0.0)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(FATAL_ERROR "VkLayerNeuralStatistics requires Linux or Android")
endif()

find_package(Threads REQUIRED)

# Common configuration
set(LGL_LOG_TAG "VkLayerNeuralStatistics")

option(LGL_CONFIG_TRACE "Enable Vulkan entrypoint logging")
option(LGL_CONFIG_OPTIMIZE_DISPATCH "Enable Vulkan entrypoint dispatch optimization" ON)
option(LGL_CONFIG_LOG "Enable general layer logging" ON)

set(NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY "" CACHE STRING
    "Test-only collector capacity override; leave empty for the production capacity of 64")
mark_as_advanced(NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY)
if(NOT NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY STREQUAL "")
    if(NOT NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY must be a positive integer")
    endif()
    add_compile_definitions(
        NEURAL_STATISTICS_COLLECTOR_CAPACITY=${NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY})
    message(WARNING
        "Building a test-only layer with collector capacity ${NEURAL_STATISTICS_TEST_COLLECTOR_CAPACITY}; do not ship this artifact")
endif()

include(../source_common/compiler_helper.cmake)
include(../cmake/clang-tools.cmake)

set(NEURAL_STATISTICS_GENERATED_VULKAN_UTILITY_SOURCES
    "${PROJECT_SOURCE_DIR}/../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_core.cpp"
    "${PROJECT_SOURCE_DIR}/../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_ext.cpp"
    "${PROJECT_SOURCE_DIR}/../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_khr.cpp"
    "${PROJECT_SOURCE_DIR}/../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_vendor.cpp")

function(neural_statistics_demote_generated_vulkan_utility_warnings target)
    set_property(
        SOURCE ${NEURAL_STATISTICS_GENERATED_VULKAN_UTILITY_SOURCES}
        TARGET_DIRECTORY ${target}
        APPEND PROPERTY COMPILE_OPTIONS
            "$<${is_gnu_fe}:-Wno-error=misleading-indentation>")
endfunction()

# Build steps
add_subdirectory(source)
add_subdirectory(../source_common/framework source_common/framework)
neural_statistics_demote_generated_vulkan_utility_warnings(lib_layer_framework)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
    neural_statistics_demote_generated_vulkan_utility_warnings(live_hook_tests)
endif()

option(NEURAL_STATISTICS_BUILD_E2E "Build the standalone Vulkan E2E fixture and host orchestrator tests" OFF)
if(NEURAL_STATISTICS_BUILD_E2E)
    add_subdirectory(e2e)
endif()
