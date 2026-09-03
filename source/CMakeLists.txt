# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

# Set output file names
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(VK_LAYER VkLayerNeuralStatistics_sym)
    set(VK_LAYER_STRIP libVkLayerNeuralStatistics.so)
else()
    set(VK_LAYER VkLayerNeuralStatistics)
endif()

# Set strings used by configure
set(LGL_LAYER_NAME_STR "VK_LAYER_LGL_neural_statistics")
set(LGL_LAYER_DESC_STR "VkLayerNeuralStatistics by LGL")
execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    OUTPUT_VARIABLE LGL_COMMIT_ID_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if (NOT LGL_COMMIT_ID_STR)
    set(LGL_COMMIT_ID_STR "unknown")
endif()

# Vulkan layer library
configure_file(
    version.hpp.in
    version.hpp
    ESCAPE_QUOTES @ONLY)

add_library(${VK_LAYER} SHARED)

# Layer source files
target_sources(${VK_LAYER} PRIVATE
    ../../source_common/framework/entry.cpp
    capture_filesystem.cpp
    capture_model.cpp
    capture_namespace.cpp
    capture_policy.cpp
    capture_serialization.cpp
    capture_store.cpp
    capture_writer.cpp
    dispatch_filter.cpp
    fault_injection.cpp
    gpu_collector.cpp
    device.cpp
    instance.cpp
    instance_creation_transaction.cpp
    resource_tracking.cpp
    submission_planner.cpp
    layer_device_functions.cpp
    layer_instance_functions.cpp
    layer_options.cpp
    layer_setting_resolver.cpp
)

# Layer settings utilities used in this layer
target_sources(${VK_LAYER} PRIVATE
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_manager.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_util.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings_helper.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings.cpp
)

target_include_directories(
    ${VK_LAYER} PRIVATE
        ./
        ../../source_common/
        ../../source_third_party/
        ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(
    ${VK_LAYER} SYSTEM PRIVATE
        ../../source_third_party/khronos/vulkan/include/
        ../../source_third_party/khronos/vulkan-utilities/include/)

lgl_set_build_options(${VK_LAYER})

# Patch compiler warnings to fix Vulkan Utilities build
target_compile_options(${VK_LAYER} PRIVATE
    $<${is_gnu_fe}:-Wno-double-promotion>
    $<${is_gnu_fe}:-Wno-shadow>
)

target_link_libraries(
    ${VK_LAYER}
        lib_layer_framework
        Threads::Threads
        $<$<PLATFORM_ID:Android>:log>)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(
        TARGET "${VK_LAYER}" POST_BUILD
        COMMAND ${CMAKE_STRIP}
        ARGS --strip-all -o ${VK_LAYER_STRIP} $<TARGET_FILE:${VK_LAYER}>
        COMMENT "Stripped lib${VK_LAYER}.so to ${VK_LAYER_STRIP}")
endif()

add_clang_tools()
