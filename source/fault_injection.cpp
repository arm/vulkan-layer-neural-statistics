/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "fault_injection.hpp"

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)

#include <condition_variable>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>

namespace capture::fault
{
namespace
{
struct State
{
    Point point{Point::TrackPipelineAfterInsertion};
    Failure failure{Failure::BadAllocation};
    std::size_t remaining{0};
    bool enabled{false};
};

thread_local State localState;
State globalState;
std::mutex globalMutex;

struct PersistentAllocationState
{
    std::size_t remaining{0};
    bool enabled{false};
    bool failing{false};
};

PersistentAllocationState persistentAllocationState;
std::mutex persistentAllocationMutex;

struct PauseState
{
    Point point{Point::TrackPipelineAfterInsertion};
    bool enabled{false};
    bool reached{false};
    bool released{false};
};

PauseState pauseState;
std::mutex pauseMutex;
std::condition_variable pauseCondition;

class InjectedUnexpectedFailure final : public std::exception
{
  public:
    const char *what() const noexcept override
    {
        return "injected unexpected capture failure";
    }
};

[[noreturn]] void Throw(Failure failure)
{
    switch (failure)
    {
    case Failure::BadAllocation:
        throw std::bad_alloc{};
    case Failure::CounterExhaustion:
        throw std::overflow_error("injected capture counter exhaustion");
    case Failure::Unexpected:
        throw InjectedUnexpectedFailure{};
    }
    throw InjectedUnexpectedFailure{};
}

std::optional<Failure> Consume(State &state, Point point) noexcept
{
    if (!state.enabled || state.point != point)
    {
        return std::nullopt;
    }
    if (state.remaining > 1)
    {
        --state.remaining;
        return std::nullopt;
    }
    state.enabled = false;
    state.remaining = 0;
    return state.failure;
}
} // namespace

ScopedInjection::ScopedInjection(Point point, Failure failure, std::size_t occurrence) noexcept
    : previousPoint_(localState.point), previousFailure_(localState.failure),
      previousRemaining_(localState.remaining), previousEnabled_(localState.enabled)
{
    localState.point = point;
    localState.failure = failure;
    localState.remaining = occurrence == 0 ? 1 : occurrence;
    localState.enabled = true;
}

ScopedInjection::~ScopedInjection() noexcept
{
    localState.point = previousPoint_;
    localState.failure = previousFailure_;
    localState.remaining = previousRemaining_;
    localState.enabled = previousEnabled_;
}

ScopedGlobalInjection::ScopedGlobalInjection(Point point, Failure failure,
                                             std::size_t occurrence) noexcept
{
    std::lock_guard lock(globalMutex);
    previousPoint_ = globalState.point;
    previousFailure_ = globalState.failure;
    previousRemaining_ = globalState.remaining;
    previousEnabled_ = globalState.enabled;
    globalState.point = point;
    globalState.failure = failure;
    globalState.remaining = occurrence == 0 ? 1 : occurrence;
    globalState.enabled = true;
}

ScopedGlobalInjection::~ScopedGlobalInjection() noexcept
{
    std::lock_guard lock(globalMutex);
    globalState.point = previousPoint_;
    globalState.failure = previousFailure_;
    globalState.remaining = previousRemaining_;
    globalState.enabled = previousEnabled_;
}

ScopedGlobalPersistentAllocationFailure::ScopedGlobalPersistentAllocationFailure(
    std::size_t failFromOccurrence) noexcept
{
    std::lock_guard lock(persistentAllocationMutex);
    previousRemaining_ = persistentAllocationState.remaining;
    previousEnabled_ = persistentAllocationState.enabled;
    previousFailing_ = persistentAllocationState.failing;
    persistentAllocationState.remaining = failFromOccurrence == 0 ? 1 : failFromOccurrence;
    persistentAllocationState.enabled = true;
    persistentAllocationState.failing = false;
}

ScopedGlobalPersistentAllocationFailure::~ScopedGlobalPersistentAllocationFailure() noexcept
{
    std::lock_guard lock(persistentAllocationMutex);
    persistentAllocationState.remaining = previousRemaining_;
    persistentAllocationState.enabled = previousEnabled_;
    persistentAllocationState.failing = previousFailing_;
}

ScopedGlobalPause::ScopedGlobalPause(Point point) noexcept : point_(point)
{
    std::lock_guard lock(pauseMutex);
    pauseState = PauseState{point, true, false, false};
}

ScopedGlobalPause::~ScopedGlobalPause() noexcept
{
    release();
    std::lock_guard lock(pauseMutex);
    pauseState.enabled = false;
    pauseCondition.notify_all();
}

void ScopedGlobalPause::waitUntilReached()
{
    std::unique_lock lock(pauseMutex);
    pauseCondition.wait(lock, [] { return pauseState.reached || !pauseState.enabled; });
}

void ScopedGlobalPause::release() noexcept
{
    std::lock_guard lock(pauseMutex);
    if (pauseState.enabled && pauseState.point == point_)
    {
        pauseState.released = true;
        pauseCondition.notify_all();
    }
}

void Checkpoint(Point point)
{
    {
        std::unique_lock lock(pauseMutex);
        if (pauseState.enabled && pauseState.point == point)
        {
            pauseState.reached = true;
            pauseCondition.notify_all();
            pauseCondition.wait(lock, [] { return pauseState.released || !pauseState.enabled; });
        }
    }

    if (const auto failure = Consume(localState, point))
    {
        Throw(*failure);
    }

    std::optional<Failure> globalFailure;
    {
        std::lock_guard lock(globalMutex);
        globalFailure = Consume(globalState, point);
    }
    if (globalFailure)
    {
        Throw(*globalFailure);
    }
}

void AllocationCheckpoint()
{
    bool fail = false;
    {
        std::lock_guard lock(persistentAllocationMutex);
        if (!persistentAllocationState.enabled)
        {
            return;
        }
        if (persistentAllocationState.failing)
        {
            fail = true;
        }
        else if (persistentAllocationState.remaining > 1)
        {
            --persistentAllocationState.remaining;
        }
        else
        {
            persistentAllocationState.remaining = 0;
            persistentAllocationState.failing = true;
            fail = true;
        }
    }
    if (fail)
    {
        throw std::bad_alloc{};
    }
}
} // namespace capture::fault

#endif
