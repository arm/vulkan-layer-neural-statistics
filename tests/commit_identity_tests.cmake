# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.19)
if(NOT DEFINED SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "SOURCE_DIR and TEST_BINARY_DIR are required")
endif()
include("${SOURCE_DIR}/cmake/commit_identity.cmake")
find_package(Git REQUIRED)

foreach(variable GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_OBJECT_DIRECTORY
                 GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE GIT_INDEX_FILE)
    unset(ENV{${variable}})
endforeach()

function(run_git output_variable)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -c commit.gpgsign=false -c core.hooksPath=
            -c user.name=CommitIdentityTest -c user.email=commit-identity@example.invalid
            ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Fixture Git command failed: ${ARGN}\n${error}")
    endif()
    set("${output_variable}" "${output}" PARENT_SCOPE)
endfunction()

function(expect_identity source expected label)
    neural_statistics_get_commit_identity("${source}" actual)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "${label}: expected '${expected}', got '${actual}'")
    endif()
    message(STATUS "Passed: ${label}")
endfunction()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef fixture_id)
set(scratch "${TEST_BINARY_DIR}/${fixture_id}")
file(MAKE_DIRECTORY "${scratch}")
set(parent "${scratch}/parent")
run_git(unused init --quiet "${parent}")
file(WRITE "${parent}/parent.txt" "parent repository\n")
run_git(unused -C "${parent}" add parent.txt)
run_git(unused -C "${parent}" commit --quiet -m "Parent fixture")
run_git(parent_identity -C "${parent}" rev-parse --short=12 HEAD)
expect_identity("${parent}" "${parent_identity}" "normal checkout")

set(layer "${parent}/layer")
file(MAKE_DIRECTORY "${layer}")
expect_identity("${layer}" "unknown" "source-only directory does not inherit parent identity")
set(ENV{GIT_DIR} "${parent}/.git")
set(ENV{GIT_WORK_TREE} "${parent}")
expect_identity("${layer}" "unknown" "environment cannot give source-only directory an identity")
unset(ENV{GIT_DIR})
unset(ENV{GIT_WORK_TREE})

run_git(unused init --quiet "${layer}")
expect_identity("${layer}" "unknown" "unborn nested repository does not inherit parent HEAD")
file(WRITE "${layer}/layer.txt" "first layer revision\n")
run_git(unused -C "${layer}" add layer.txt)
run_git(unused -C "${layer}" commit --quiet -m "First layer fixture")
run_git(first_identity -C "${layer}" rev-parse --short=12 HEAD)
expect_identity("${layer}" "${first_identity}" "nested independent repository uses its own HEAD")

file(WRITE "${layer}/layer.txt" "second layer revision\n")
run_git(unused -C "${layer}" add layer.txt)
run_git(unused -C "${layer}" commit --quiet -m "Second layer fixture")
run_git(second_identity -C "${layer}" rev-parse --short=12 HEAD)
set(linked "${parent}/linked-layer")
run_git(unused -C "${layer}" worktree add --quiet --detach "${linked}" "${first_identity}")
expect_identity("${linked}" "${first_identity}" "linked worktree gitfile uses its own detached HEAD")
expect_identity("${layer}" "${second_identity}" "main worktree retains its different HEAD")

set(relocated "${parent}/ci-layer")
file(COPY "${layer}/" DESTINATION "${relocated}")
expect_identity("${relocated}" "${second_identity}" "copy with Git metadata keeps the layer identity")
file(REMOVE_RECURSE "${relocated}/.git")
expect_identity("${relocated}" "unknown" "CI copy without Git metadata does not inherit framework identity")

set(ENV{GIT_DIR} "${parent}/.git")
set(ENV{GIT_WORK_TREE} "${parent}")
set(ENV{GIT_COMMON_DIR} "${parent}/.git")
expect_identity("${layer}" "${second_identity}" "foreign Git environment cannot redirect a checkout")
expect_identity("${linked}" "${first_identity}" "foreign Git environment cannot redirect a worktree")
unset(ENV{GIT_DIR})
unset(ENV{GIT_WORK_TREE})
unset(ENV{GIT_COMMON_DIR})

set(broken "${parent}/broken-layer")
file(MAKE_DIRECTORY "${broken}")
file(WRITE "${broken}/.git" "gitdir: missing-metadata\n")
expect_identity("${broken}" "unknown" "broken local gitfile never falls back to parent")

run_git(unused -C "${layer}" worktree remove "${linked}")
file(REMOVE_RECURSE "${scratch}")
message(STATUS "All layer commit identity tests passed")
