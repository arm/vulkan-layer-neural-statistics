/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_model.hpp"

#include <exception>
#include <new>
#include <stdexcept>

#include <vulkan/vulkan_core.h>

namespace capture::abi
{
// Configuration failures happen before downstream creation. Object-publication
// failures may happen after downstream creation, but before the object is
// returned to the application; the caller rolls that object back and returns an
// allowed VkResult. PostForwardCapture is reserved for work that the downstream
// implementation has already accepted and that cannot safely be reported as
// unexecuted or retried; capture is disabled and the accepted result is kept.
enum class BoundaryPhase
{
    Configuration,
    ObjectPublication,
    PostForwardCapture,
};

enum class Action
{
    ReturnError,
    DisableCaptureAndPreserveDownstreamResult,
};

struct Translation
{
    Action action{Action::ReturnError};
    VkResult result{VK_ERROR_INITIALIZATION_FAILED};
};

inline Translation TranslateException(std::exception_ptr exception, BoundaryPhase phase) noexcept
{
    if (phase == BoundaryPhase::PostForwardCapture)
    {
        return {Action::DisableCaptureAndPreserveDownstreamResult, VK_SUCCESS};
    }

    try
    {
        if (exception)
        {
            std::rethrow_exception(exception);
        }
    }
    catch (const std::bad_alloc &)
    {
        return {Action::ReturnError, VK_ERROR_OUT_OF_HOST_MEMORY};
    }
    catch (const CaptureTerminalError &)
    {
        return {Action::ReturnError, VK_ERROR_INITIALIZATION_FAILED};
    }
    catch (const std::overflow_error &)
    {
        return {Action::ReturnError, VK_ERROR_INITIALIZATION_FAILED};
    }
    catch (...)
    {
        return {Action::ReturnError, VK_ERROR_INITIALIZATION_FAILED};
    }

    return {Action::ReturnError, VK_ERROR_INITIALIZATION_FAILED};
}
} // namespace capture::abi
