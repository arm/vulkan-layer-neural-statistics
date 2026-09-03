/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include "options.hpp"
#include <memory>
namespace fixture
{
class VulkanApp
{
  public:
    explicit VulkanApp(const Options &options);
    ~VulkanApp();
    VulkanApp(const VulkanApp &) = delete;
    VulkanApp &operator=(const VulkanApp &) = delete;
    void run();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fixture
