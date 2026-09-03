# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

set(FIXTURE_VULKAN_HEADERS
    "${PROJECT_SOURCE_DIR}/../source_third_party/khronos/vulkan/include")
if(NOT EXISTS "${FIXTURE_VULKAN_HEADERS}/vulkan/vulkan_core.h")
    message(FATAL_ERROR "Pinned Vulkan headers not found at ${FIXTURE_VULKAN_HEADERS}")
endif()

function(embed_fixture_binary input symbol output)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
    file(READ "${input}" binary_hex HEX)
    string(REGEX REPLACE "(..)" "0x\\1," binary_bytes "${binary_hex}")
    file(WRITE "${output}"
        "#include <cstddef>\n"
        "alignas(8) extern const unsigned char ${symbol}[] = {${binary_bytes}};\n"
        "extern const std::size_t ${symbol}_size = sizeof(${symbol});\n")
endfunction()

embed_fixture_binary(
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/graph_partition_0.spv"
    fixture_graph_spirv
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_spirv.cpp")
embed_fixture_binary(
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/graph_constants.bin"
    fixture_graph_constants
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_constants.cpp")

add_executable(neural_statistics_fixture
    main.cpp
    events.cpp
    options.cpp
    topology_plan.cpp
    vulkan_app.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_spirv.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_constants.cpp")
target_compile_features(neural_statistics_fixture PRIVATE cxx_std_20)
target_include_directories(neural_statistics_fixture PRIVATE "${FIXTURE_VULKAN_HEADERS}")
target_compile_definitions(neural_statistics_fixture PRIVATE VK_NO_PROTOTYPES)
target_link_libraries(neural_statistics_fixture PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
lgl_set_build_options(neural_statistics_fixture)
set_property(
    SOURCE vulkan_app.cpp
    APPEND PROPERTY COMPILE_OPTIONS
        "$<${is_gnu_fe}:-Wno-missing-field-initializers>")
install(TARGETS neural_statistics_fixture RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_executable(neural_statistics_fixture_behavior_tests
        behavior_tests.cpp
        events.cpp
        topology_plan.cpp)
    target_compile_features(neural_statistics_fixture_behavior_tests PRIVATE cxx_std_20)
    target_include_directories(neural_statistics_fixture_behavior_tests PRIVATE "${FIXTURE_VULKAN_HEADERS}")
    target_compile_definitions(neural_statistics_fixture_behavior_tests PRIVATE VK_NO_PROTOTYPES)
    lgl_set_build_options(neural_statistics_fixture_behavior_tests)
    add_test(NAME neural_statistics_fixture_behavior_tests COMMAND neural_statistics_fixture_behavior_tests)
endif()
