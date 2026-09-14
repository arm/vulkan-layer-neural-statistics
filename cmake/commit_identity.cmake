# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

function(neural_statistics_get_commit_identity source_dir output_variable)
    set(identity "unknown")
    get_filename_component(layer_root "${source_dir}" REALPATH)

    # An explicit metadata path supports normal checkouts and linked worktrees,
    # but never discovers an enclosing repository for a source-only directory.
    if(EXISTS "${layer_root}/.git")
        find_package(Git QUIET)
        if(Git_FOUND)
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                    --unset=GIT_DIR
                    --unset=GIT_WORK_TREE
                    --unset=GIT_COMMON_DIR
                    --unset=GIT_OBJECT_DIRECTORY
                    --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
                    --unset=GIT_NAMESPACE
                    "${GIT_EXECUTABLE}"
                    "--git-dir=${layer_root}/.git"
                    "--work-tree=${layer_root}"
                    rev-parse --verify --short=12 HEAD
                RESULT_VARIABLE git_result
                OUTPUT_VARIABLE git_identity
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(git_result EQUAL 0 AND git_identity MATCHES "^[0-9a-f]+$")
                set(identity "${git_identity}")
            endif()
        endif()
    endif()

    set("${output_variable}" "${identity}" PARENT_SCOPE)
endfunction()
