/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace capture
{
// A bounded, lossless FIFO. Producers block while the queue is full. close()
// rejects future pushes, wakes blocked producers/consumers, and permits queued
// entries to be drained before pop() returns false.
template <typename T> class BoundedQueue
{
  public:
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "BoundedQueue requires nothrow-movable entries so dequeue cannot lose a value");

    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("bounded queue capacity must be non-zero");
        }
    }

    BoundedQueue(const BoundedQueue &) = delete;
    BoundedQueue &operator=(const BoundedQueue &) = delete;

    bool push(T value)
    {
        std::unique_lock lock(mutex_);
        if (!closed_ && values_.size() >= capacity_)
        {
            ++blockedProducers_;
            blockedChanged_.notify_all();
            notFull_.wait(lock, [this] { return closed_ || values_.size() < capacity_; });
            --blockedProducers_;
            blockedChanged_.notify_all();
        }
        if (closed_)
        {
            return false;
        }
        values_.push_back(std::move(value));
        notEmpty_.notify_one();
        return true;
    }

    std::optional<T> pop()
    {
        std::unique_lock lock(mutex_);
        notEmpty_.wait(lock, [this] { return closed_ || !values_.empty(); });
        if (values_.empty())
        {
            return std::nullopt;
        }
        T value = std::move(values_.front());
        values_.pop_front();
        notFull_.notify_one();
        return value;
    }

    void close() noexcept
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
        notFull_.notify_all();
        notEmpty_.notify_all();
        blockedChanged_.notify_all();
    }

    bool waitForBlockedProducers(std::size_t count)
    {
        std::unique_lock lock(mutex_);
        blockedChanged_.wait(lock, [this, count] { return closed_ || blockedProducers_ >= count; });
        return blockedProducers_ >= count;
    }

    bool closed() const noexcept
    {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    std::size_t size() const noexcept
    {
        std::lock_guard lock(mutex_);
        return values_.size();
    }

    std::size_t blockedProducerCount() const noexcept
    {
        std::lock_guard lock(mutex_);
        return blockedProducers_;
    }

    std::size_t capacity() const noexcept
    {
        return capacity_;
    }

  private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::condition_variable blockedChanged_;
    std::deque<T> values_;
    std::size_t blockedProducers_{0};
    bool closed_{false};
};
} // namespace capture
