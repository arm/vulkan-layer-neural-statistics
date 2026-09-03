# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

if(UNIX AND NOT ANDROID)
    add_subdirectory(fixture)
endif()

find_package(Python3 3.9 COMPONENTS Interpreter)
if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(
        NAME neural_statistics_e2e_tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p "test_*.py")
    set_tests_properties(neural_statistics_e2e_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
endif()
