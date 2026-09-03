/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace capture
{
enum class FilePublicationMode
{
    CreateNew,
    Replace,
};

enum class FileSystemOperation
{
    ClaimRoot,
    CreateDirectories,
    Publish,
    OpenTemporaryFile,
    WriteTemporaryFile,
    RenameTemporaryFile,
    UnsupportedCapability,
};

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
using CaptureFileSystemObserver = void (*)(FileSystemOperation) noexcept;

// Test-only observer at the accepted filesystem-operation boundary. Production
// builds compile this facility out completely; tests use it to prove Vulkan
// producer threads never cross into capture publication operations.
void SetCaptureFileSystemObserver(CaptureFileSystemObserver observer) noexcept;
#endif

class CaptureFileSystemError final : public std::runtime_error
{
  public:
    CaptureFileSystemError(FileSystemOperation operation, std::filesystem::path path,
                           std::string message);

    FileSystemOperation operation() const noexcept;
    const std::filesystem::path &path() const noexcept;

  private:
    FileSystemOperation operation_;
    std::filesystem::path path_;
};

// A small rooted publication boundary. The final capture root is created
// exclusively and retained as a directory descriptor. Operations below it use
// descriptor-relative calls and refuse symlink path components. The class does
// not detect same-user renames, replacements, or edits of ordinary entries.
class CaptureFileSystem
{
  public:
    static CaptureFileSystem ClaimRoot(const std::filesystem::path &root);

    ~CaptureFileSystem() noexcept;
    CaptureFileSystem(CaptureFileSystem &&) noexcept;
    CaptureFileSystem &operator=(CaptureFileSystem &&) = delete;
    CaptureFileSystem(const CaptureFileSystem &) = delete;
    CaptureFileSystem &operator=(const CaptureFileSystem &) = delete;

    // Keep the exclusively claimed root after a valid initial capture document
    // has been published. Before this call destruction removes the root if it is
    // still empty.
    void preserveClaimedRoot() noexcept;

    void createDirectories(const std::filesystem::path &relativePath);
    void publish(const std::filesystem::path &relativePath, std::span<const uint8_t> contents,
                 FilePublicationMode mode);
    void publishText(const std::filesystem::path &relativePath, std::string_view contents,
                     FilePublicationMode mode);

  private:
    CaptureFileSystem(std::filesystem::path root, int rootDescriptor) noexcept;
    std::filesystem::path root_;
    int rootDescriptor_{-1};
    bool preserveRoot_{false};
};
} // namespace capture
