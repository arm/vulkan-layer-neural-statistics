# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

add_executable(capture_policy_tests
    capture_policy_tests.cpp
    ../source/capture_policy.cpp
    ../source/resource_tracking.cpp
)

target_include_directories(capture_policy_tests PRIVATE
    ../source
    ../../source_third_party/khronos/vulkan/include
)

target_compile_features(capture_policy_tests PRIVATE cxx_std_20)
add_test(NAME capture_policy_tests COMMAND capture_policy_tests)

set(CAPTURE_FOUNDATION_SOURCES
    ../source/capture_model.cpp
    ../source/capture_namespace.cpp
    ../source/dispatch_filter.cpp
    ../source/fault_injection.cpp
    ../source/instance_creation_transaction.cpp
    ../source/layer_options.cpp
    ../source/layer_setting_resolver.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_manager.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_util.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings_helper.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings.cpp
)

add_executable(capture_foundations_tests
    capture_foundations_tests.cpp
    ../source/capture_serialization.cpp
    ${CAPTURE_FOUNDATION_SOURCES}
)

target_include_directories(capture_foundations_tests PRIVATE
    ../source
    ${PROJECT_BINARY_DIR}/source
    ../../source_third_party
    ../../source_third_party/khronos/vulkan/include
    ../../source_third_party/khronos/vulkan-utilities/include
)

target_compile_definitions(capture_foundations_tests PRIVATE
    NEURAL_STATISTICS_ENABLE_FAULT_INJECTION
    NEURAL_STATISTICS_MANIFEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../manifest.json"
    NEURAL_STATISTICS_SETTINGS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../vk_layer_settings.txt"
)
target_compile_features(capture_foundations_tests PRIVATE cxx_std_20)
add_test(NAME capture_foundations_tests COMMAND capture_foundations_tests)

add_executable(capture_writer_tests
    capture_writer_tests.cpp
    ../source/capture_filesystem.cpp
    ../source/capture_model.cpp
    ../source/capture_namespace.cpp
    ../source/capture_serialization.cpp
    ../source/capture_store.cpp
    ../source/capture_writer.cpp
    ../source/dispatch_filter.cpp
    ../source/fault_injection.cpp
)

target_include_directories(capture_writer_tests PRIVATE
    ../source
    ../../source_third_party
)
target_compile_definitions(capture_writer_tests PRIVATE
    NEURAL_STATISTICS_ENABLE_FAULT_INJECTION
)
target_compile_features(capture_writer_tests PRIVATE cxx_std_20)
target_link_libraries(capture_writer_tests PRIVATE Threads::Threads)
add_test(NAME capture_writer_tests COMMAND capture_writer_tests)

add_executable(exception_safety_tests
    exception_safety_tests.cpp
    ${CAPTURE_FOUNDATION_SOURCES}
)

target_include_directories(exception_safety_tests PRIVATE
    ../source
    ${PROJECT_BINARY_DIR}/source
    ../../source_third_party
    ../../source_third_party/khronos/vulkan/include
    ../../source_third_party/khronos/vulkan-utilities/include
)

target_compile_definitions(exception_safety_tests PRIVATE
    NEURAL_STATISTICS_ENABLE_FAULT_INJECTION
)
target_compile_features(exception_safety_tests PRIVATE cxx_std_20)
add_test(NAME exception_safety_tests COMMAND exception_safety_tests)

add_executable(submission_planner_tests
    submission_planner_tests.cpp
    ../source/submission_planner.cpp
    ../source/dispatch_filter.cpp
    ../source/resource_tracking.cpp
)

target_include_directories(submission_planner_tests PRIVATE
    ../source
    ../../source_third_party/khronos/vulkan/include
)
target_compile_features(submission_planner_tests PRIVATE cxx_std_20)
target_link_libraries(submission_planner_tests PRIVATE Threads::Threads)
add_test(NAME submission_planner_tests COMMAND submission_planner_tests)

set(LIVE_HOOK_SOURCES
    ../source/capture_filesystem.cpp
    ../source/capture_model.cpp
    ../source/capture_namespace.cpp
    ../source/capture_policy.cpp
    ../source/capture_serialization.cpp
    ../source/capture_store.cpp
    ../source/capture_writer.cpp
    ../source/device.cpp
    ../source/dispatch_filter.cpp
    ../source/fault_injection.cpp
    ../source/gpu_collector.cpp
    ../source/instance.cpp
    ../source/instance_creation_transaction.cpp
    ../source/layer_device_functions.cpp
    ../source/layer_instance_functions.cpp
    ../source/layer_options.cpp
    ../source/layer_setting_resolver.cpp
    ../source/resource_tracking.cpp
    ../source/submission_planner.cpp
    ../../source_common/framework/device_functions.cpp
    ../../source_common/framework/instance_functions.cpp
    ../../source_common/framework/manual_functions.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_manager.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/layer_settings_util.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings_helper.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/layer/vk_layer_settings.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_core.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_ext.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_khr.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_manual.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_utils.cpp
    ../../source_third_party/khronos/vulkan-utilities/src/vulkan/vk_safe_struct_vendor.cpp
)

add_executable(live_hook_tests
    live_hook_tests.cpp
    ${LIVE_HOOK_SOURCES}
)

target_include_directories(live_hook_tests PRIVATE
    ../source
    ${PROJECT_BINARY_DIR}/source
    ../../source_common
    ../../source_third_party
    ../../source_third_party/khronos/vulkan/include
    ../../source_third_party/khronos/vulkan-utilities/include
)
target_compile_definitions(live_hook_tests PRIVATE
    NEURAL_STATISTICS_ENABLE_FAULT_INJECTION
    LGL_LOG_TAG="${LGL_LOG_TAG}"
    CONFIG_LOG=1
    CONFIG_OPTIMIZE_DISPATCH=1
    CONFIG_TRACE=0
)
target_compile_features(live_hook_tests PRIVATE cxx_std_20)
target_link_libraries(live_hook_tests PRIVATE Threads::Threads)
add_test(NAME live_hook_tests COMMAND live_hook_tests)
