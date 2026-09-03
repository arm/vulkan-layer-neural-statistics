/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_filesystem.hpp"

#include "capture_namespace.hpp"
#include "fault_injection.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace capture
{
namespace
{
#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
std::atomic<CaptureFileSystemObserver> fileSystemObserver{nullptr};

void NotifyFileSystemObserver(FileSystemOperation operation) noexcept
{
    if (const auto observer = fileSystemObserver.load(std::memory_order_relaxed))
    {
        observer(operation);
    }
}
#else
void NotifyFileSystemObserver(FileSystemOperation) noexcept {}
#endif

std::string Describe(std::string_view action, const std::filesystem::path &path,
                     std::string_view detail)
{
    std::string result(action);
    result += " '";
    result += path.string();
    result += "'";
    if (!detail.empty())
    {
        result += ": ";
        result += detail;
    }
    return result;
}

[[noreturn]] void Throw(FileSystemOperation operation, const std::filesystem::path &path,
                        std::string_view action, const std::error_code &error)
{
    throw CaptureFileSystemError(operation, path, Describe(action, path, error.message()));
}

[[noreturn]] void ThrowErrno(FileSystemOperation operation, const std::filesystem::path &path,
                             std::string_view action, int error)
{
    Throw(operation, path, action, std::error_code(error, std::generic_category()));
}

[[noreturn]] void ThrowMessage(FileSystemOperation operation, const std::filesystem::path &path,
                               std::string_view message)
{
    throw CaptureFileSystemError(operation, path, Describe(message, path, {}));
}

std::filesystem::path AbsoluteLexicalRoot(const std::filesystem::path &root)
{
    if (root.empty())
    {
        ThrowMessage(FileSystemOperation::ClaimRoot, root, "capture root must not be empty");
    }

    std::error_code error;
    auto absolute = std::filesystem::absolute(root, error);
    if (error)
    {
        Throw(FileSystemOperation::ClaimRoot, root, "cannot make capture root absolute", error);
    }

    absolute = absolute.lexically_normal();
    if (absolute == absolute.root_path() || absolute.filename().empty())
    {
        ThrowMessage(FileSystemOperation::ClaimRoot, root,
                     "capture root must name a new child directory");
    }
    return absolute;
}

void ValidateRelativePath(const std::filesystem::path &path, FileSystemOperation operation,
                          const std::filesystem::path &reportPath)
{
    if (path.is_absolute())
    {
        ThrowMessage(operation, reportPath, "capture operation path must be root-relative");
    }
    for (const auto &component : path)
    {
        if (component.empty() || component == "." || component == "..")
        {
            ThrowMessage(operation, reportPath,
                         "capture operation path contains an invalid component");
        }
    }
}

class FileDescriptor
{
  public:
    explicit FileDescriptor(int value = -1) noexcept : value_(value) {}
    ~FileDescriptor() noexcept
    {
        if (value_ >= 0)
        {
            (void)::close(value_);
        }
    }

    FileDescriptor(FileDescriptor &&other) noexcept : value_(std::exchange(other.value_, -1)) {}

    FileDescriptor &operator=(FileDescriptor &&other) noexcept
    {
        if (this != &other)
        {
            if (value_ >= 0)
            {
                (void)::close(value_);
            }
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }

    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;

    int get() const noexcept
    {
        return value_;
    }
    int release() noexcept
    {
        return std::exchange(value_, -1);
    }

  private:
    int value_;
};

FileDescriptor DuplicateDirectory(int descriptor, FileSystemOperation operation,
                                  const std::filesystem::path &path)
{
    const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
    if (duplicate < 0)
    {
        ThrowErrno(operation, path, "cannot duplicate capture-root descriptor", errno);
    }
    return FileDescriptor(duplicate);
}

FileDescriptor OpenDirectoryAt(int parent, const std::filesystem::path &name,
                               FileSystemOperation operation,
                               const std::filesystem::path &reportPath)
{
    const int descriptor =
        ::openat(parent, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0)
    {
        ThrowErrno(operation, reportPath, "cannot open capture directory", errno);
    }
    return FileDescriptor(descriptor);
}

FileDescriptor OpenRelativeDirectory(int rootDescriptor, const std::filesystem::path &root,
                                     const std::filesystem::path &relativePath,
                                     FileSystemOperation operation)
{
    ValidateRelativePath(relativePath, operation, root / relativePath);
    FileDescriptor current = DuplicateDirectory(rootDescriptor, operation, root);
    std::filesystem::path currentPath;
    for (const auto &component : relativePath)
    {
        currentPath /= component;
        current = OpenDirectoryAt(current.get(), component, operation, root / currentPath);
    }
    return current;
}

void RenameNoReplace(int parent, const std::filesystem::path &temporary,
                     const std::filesystem::path &target, const std::filesystem::path &reportPath)
{
#if defined(SYS_renameat2)
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1U << 0)
#endif
    if (::syscall(SYS_renameat2, parent, temporary.c_str(), parent, target.c_str(),
                  RENAME_NOREPLACE) == 0)
    {
        return;
    }

    const int error = errno;
    if (error == ENOSYS || error == EINVAL || error == EOPNOTSUPP || error == ENOTSUP)
    {
        ThrowMessage(FileSystemOperation::UnsupportedCapability, reportPath,
                     "filesystem does not support atomic rename-no-replace publication");
    }
    ThrowErrno(FileSystemOperation::RenameTemporaryFile, reportPath,
               "cannot atomically publish new file", error);
#else
    (void)parent;
    (void)temporary;
    (void)target;
    ThrowMessage(FileSystemOperation::UnsupportedCapability, reportPath,
                 "platform does not expose atomic rename-no-replace publication");
#endif
}

std::string UniqueTemporaryName()
{
    static std::atomic<uint64_t> sequence{0};
    const auto tick =
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto count = sequence.fetch_add(1, std::memory_order_relaxed);

    std::string result(names::kInternalTemporaryPrefix);
    result += std::to_string(static_cast<uint64_t>(::getpid()));
    result.push_back('-');
    result += std::to_string(tick);
    result.push_back('-');
    result += std::to_string(count);
    result += names::kReservedTemporarySuffix;
    return result;
}
} // namespace

CaptureFileSystemError::CaptureFileSystemError(FileSystemOperation operation,
                                               std::filesystem::path path, std::string message)
    : std::runtime_error(std::move(message)), operation_(operation), path_(std::move(path))
{
}

FileSystemOperation CaptureFileSystemError::operation() const noexcept
{
    return operation_;
}

const std::filesystem::path &CaptureFileSystemError::path() const noexcept
{
    return path_;
}

CaptureFileSystem CaptureFileSystem::ClaimRoot(const std::filesystem::path &requestedRoot)
{
    NotifyFileSystemObserver(FileSystemOperation::ClaimRoot);
    auto root = AbsoluteLexicalRoot(requestedRoot);

    std::error_code error;
    std::filesystem::create_directories(root.parent_path(), error);
    if (error)
    {
        Throw(FileSystemOperation::ClaimRoot, root.parent_path(),
              "cannot create capture-root parent", error);
    }

    fault::Checkpoint(fault::Point::WriterClaimCreateRoot);
    if (::mkdir(root.c_str(), 0700) != 0)
    {
        ThrowErrno(FileSystemOperation::ClaimRoot, root, "cannot exclusively create capture root",
                   errno);
    }

    const int descriptor = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0)
    {
        const int openError = errno;
        (void)::rmdir(root.c_str());
        ThrowErrno(FileSystemOperation::ClaimRoot, root, "cannot open newly created capture root",
                   openError);
    }

    return CaptureFileSystem(std::move(root), descriptor);
}

CaptureFileSystem::CaptureFileSystem(std::filesystem::path root, int rootDescriptor) noexcept
    : root_(std::move(root)), rootDescriptor_(rootDescriptor)
{
}

CaptureFileSystem::~CaptureFileSystem() noexcept
{
    if (rootDescriptor_ >= 0)
    {
        (void)::close(rootDescriptor_);
        if (!preserveRoot_)
        {
            (void)::rmdir(root_.c_str());
        }
    }
}

CaptureFileSystem::CaptureFileSystem(CaptureFileSystem &&other) noexcept
    : root_(std::move(other.root_)), rootDescriptor_(std::exchange(other.rootDescriptor_, -1)),
      preserveRoot_(std::exchange(other.preserveRoot_, false))
{
}

void CaptureFileSystem::preserveClaimedRoot() noexcept
{
    preserveRoot_ = true;
}

void CaptureFileSystem::createDirectories(const std::filesystem::path &relativePath)
{
    NotifyFileSystemObserver(FileSystemOperation::CreateDirectories);
    ValidateRelativePath(relativePath, FileSystemOperation::CreateDirectories,
                         root_ / relativePath);
    FileDescriptor current =
        DuplicateDirectory(rootDescriptor_, FileSystemOperation::CreateDirectories, root_);
    std::filesystem::path currentPath;

    for (const auto &component : relativePath)
    {
        currentPath /= component;
        fault::Checkpoint(fault::Point::WriterCreateDirectory);
        if (::mkdirat(current.get(), component.c_str(), 0700) != 0 && errno != EEXIST)
        {
            ThrowErrno(FileSystemOperation::CreateDirectories, root_ / currentPath,
                       "cannot create capture directory", errno);
        }
        current = OpenDirectoryAt(current.get(), component, FileSystemOperation::CreateDirectories,
                                  root_ / currentPath);
    }
}

void CaptureFileSystem::publish(const std::filesystem::path &relativePath,
                                std::span<const uint8_t> contents, FilePublicationMode mode)
{
    NotifyFileSystemObserver(FileSystemOperation::Publish);
    ValidateRelativePath(relativePath, FileSystemOperation::OpenTemporaryFile,
                         root_ / relativePath);
    const auto targetName = relativePath.filename();
    if (targetName.empty())
    {
        ThrowMessage(FileSystemOperation::OpenTemporaryFile, root_ / relativePath,
                     "publication path must name a file");
    }

    const auto parentPath = relativePath.parent_path();
    FileDescriptor parent = OpenRelativeDirectory(rootDescriptor_, root_, parentPath,
                                                  FileSystemOperation::OpenTemporaryFile);

    constexpr std::size_t maximumAttempts = 32;
    for (std::size_t attempt = 0; attempt < maximumAttempts; ++attempt)
    {
        fault::Checkpoint(fault::Point::WriterOpenTemporaryFile);
        const std::filesystem::path temporaryName = UniqueTemporaryName();
        const int descriptor = ::openat(parent.get(), temporaryName.c_str(),
                                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (descriptor < 0)
        {
            if (errno == EEXIST)
            {
                continue;
            }
            ThrowErrno(FileSystemOperation::OpenTemporaryFile, root_ / parentPath / temporaryName,
                       "cannot create temporary capture file", errno);
        }

        FileDescriptor temporary(descriptor);
        try
        {
            fault::Checkpoint(fault::Point::WriterWriteTemporaryFile);
            std::size_t offset = 0;
            while (offset < contents.size())
            {
                const ssize_t written =
                    ::write(temporary.get(), contents.data() + offset, contents.size() - offset);
                if (written < 0)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }
                    ThrowErrno(FileSystemOperation::WriteTemporaryFile,
                               root_ / parentPath / temporaryName,
                               "cannot write temporary capture file", errno);
                }
                if (written == 0)
                {
                    ThrowMessage(FileSystemOperation::WriteTemporaryFile,
                                 root_ / parentPath / temporaryName,
                                 "temporary capture write made no progress");
                }
                offset += static_cast<std::size_t>(written);
            }

            if (::close(temporary.release()) != 0)
            {
                ThrowErrno(FileSystemOperation::WriteTemporaryFile,
                           root_ / parentPath / temporaryName,
                           "cannot close temporary capture file", errno);
            }

            fault::Checkpoint(fault::Point::WriterRenameTemporaryFile);
            if (mode == FilePublicationMode::CreateNew)
            {
                RenameNoReplace(parent.get(), temporaryName, targetName, root_ / relativePath);
            }
            else if (::renameat(parent.get(), temporaryName.c_str(), parent.get(),
                                targetName.c_str()) != 0)
            {
                ThrowErrno(FileSystemOperation::RenameTemporaryFile, root_ / relativePath,
                           "cannot atomically replace capture file", errno);
            }
            return;
        }
        catch (...)
        {
            (void)::unlinkat(parent.get(), temporaryName.c_str(), 0);
            throw;
        }
    }

    ThrowMessage(FileSystemOperation::OpenTemporaryFile, root_ / relativePath,
                 "could not allocate a unique temporary capture name");
}

void CaptureFileSystem::publishText(const std::filesystem::path &relativePath,
                                    std::string_view contents, FilePublicationMode mode)
{
    const auto *bytes = reinterpret_cast<const uint8_t *>(contents.data());
    publish(relativePath, std::span<const uint8_t>(bytes, contents.size()), mode);
}

#if defined(NEURAL_STATISTICS_ENABLE_FAULT_INJECTION)
void SetCaptureFileSystemObserver(CaptureFileSystemObserver observer) noexcept
{
    fileSystemObserver.store(observer, std::memory_order_relaxed);
}
#endif
} // namespace capture
