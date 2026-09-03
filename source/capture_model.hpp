/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "capture_namespace.hpp"
#include "dispatch_filter.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace capture
{
inline constexpr uint32_t kCaptureSchemaVersion = 2;
inline constexpr uint32_t kPipelineSchemaVersion = 2;
inline constexpr uint32_t kSessionSchemaVersion = 2;
inline constexpr uint32_t kDispatchSchemaVersion = 2;
inline constexpr uint32_t kFolderIndexWidth = names::kDirectoryIndexWidth;

template <typename Tag> class Id
{
  public:
    constexpr Id() noexcept = default;
    explicit constexpr Id(uint64_t value) noexcept : value_(value) {}

    constexpr uint64_t value() const noexcept
    {
        return value_;
    }
    auto operator<=>(const Id &) const = default;

  private:
    uint64_t value_{0};
};

struct LogicalDeviceTag;
struct ShaderModuleTag;
struct PipelineTag;
struct SessionTag;
struct DispatchTag;
using LogicalDeviceId = Id<LogicalDeviceTag>;
using ShaderModuleId = Id<ShaderModuleTag>;
using PipelineId = Id<PipelineTag>;
using SessionId = Id<SessionTag>;
using DispatchId = Id<DispatchTag>;

class MonotonicCounter
{
  public:
    explicit constexpr MonotonicCounter(uint64_t next = 0) noexcept : next_(next) {}

    uint64_t peek() const
    {
        if (exhausted_)
        {
            throw std::overflow_error("capture-local sequence exhausted");
        }
        return next_;
    }

    void commit() noexcept
    {
        if (next_ == std::numeric_limits<uint64_t>::max())
        {
            exhausted_ = true;
        }
        else
        {
            ++next_;
        }
    }

    uint64_t allocate()
    {
        const uint64_t value = peek();
        commit();
        return value;
    }

    uint64_t nextValue() const noexcept
    {
        return next_;
    }
    bool exhausted() const noexcept
    {
        return exhausted_;
    }

  private:
    uint64_t next_{0};
    bool exhausted_{false};
};

template <typename IdType> class IdAllocator
{
  public:
    explicit constexpr IdAllocator(uint64_t next = 0) noexcept : values_(next) {}

    IdType peek() const
    {
        return IdType(values_.peek());
    }
    void commit() noexcept
    {
        values_.commit();
    }
    IdType allocate()
    {
        const IdType value = peek();
        commit();
        return value;
    }
    uint64_t nextValue() const noexcept
    {
        return values_.nextValue();
    }
    bool exhausted() const noexcept
    {
        return values_.exhausted();
    }

  private:
    MonotonicCounter values_;
};

enum class CaptureStatus
{
    Incomplete,
    Complete,
    Error,
};

enum class CaptureTerminalFailure
{
    None,
    AllocationFailure,
    CounterExhaustion,
    UnexpectedFailure,
};

class CaptureTerminalError final : public std::exception
{
  public:
    const char *what() const noexcept override
    {
        return "capture model is in a terminal failure state";
    }
};

enum class ArtifactType
{
    DebugDatabase,
    StatisticsInfo,
    ShaderModule,
    DispatchStatistics,
};

enum class CaptureRootCollisionPolicy
{
    RejectExisting,
};

enum class CaptureRootDecision
{
    CreateNew,
    RejectEmptyPath,
    RejectExisting,
};

struct ArtifactDescriptor
{
    std::filesystem::path path;
    ArtifactType type{ArtifactType::DispatchStatistics};
    uint64_t size{0};
};

struct DeviceMetadata
{
    LogicalDeviceId id;
    std::string name;
    std::optional<uint32_t> vendorId;
    std::optional<uint32_t> deviceId;
    std::optional<uint32_t> driverVersion;
    std::optional<uint32_t> apiVersion;
};

struct CaptureMetadata
{
    std::string layerName{"VK_LAYER_LGL_neural_statistics"};
    std::string layerVersion{"1.0.0"};
    std::string commitIdentity{"unknown"};
    uint32_t layerImplementationVersion{1};
    uint32_t statisticsMode{0};
    std::string dispatchFilter;
    std::vector<DeviceMetadata> devices;
    std::vector<std::string> warnings;
};

struct ResourceBindingMetadata
{
    uint32_t descriptorSet{0};
    uint32_t binding{0};
    uint32_t arrayElement{0};
};

struct SpecializationMapEntryMetadata
{
    uint32_t constantId{0};
    uint32_t offset{0};
    uint64_t size{0};
};

struct PipelineShaderMetadata
{
    std::optional<ShaderModuleId> moduleId;
    bool spirvAvailable{false};
    std::string friendlyName;
    std::string entryPoint;
    std::vector<SpecializationMapEntryMetadata> specializationEntries;
    uint64_t specializationDataSize{0};
};

struct PipelineMetadata
{
    LogicalDeviceId deviceId;
    uint64_t flags{0};
    std::optional<uint64_t> layoutHandle;
    std::vector<ResourceBindingMetadata> resourceBindings;
    std::optional<std::string> vendorOptions;
    bool identifierOnly{false};
    bool foreignProcessingEngine{false};
    bool statisticsEnabled{true};
    PipelineShaderMetadata shader;
};

struct DispatchNode
{
    DispatchId id;
    SessionId sessionId;
    uint64_t executedIndex{0};
    std::optional<uint64_t> diagnosticCommandBufferHandle;
    std::vector<ArtifactDescriptor> artifacts;
};

struct SessionNode
{
    SessionId id;
    PipelineId pipelineId;
    LogicalDeviceId deviceId;
    std::string friendlyName;
    std::optional<uint64_t> diagnosticHandle;
    uint64_t flags{0};
    std::vector<DispatchId> dispatches;
};

struct PipelineNode
{
    PipelineId id;
    std::string friendlyName;
    std::optional<uint64_t> diagnosticHandle;
    PipelineMetadata metadata;
    std::vector<ArtifactDescriptor> artifacts;
    std::vector<SessionId> sessions;
};

struct DispatchReservation
{
    uint64_t executedIndex{0};
    std::optional<DispatchId> selectedDispatch;

    bool selected() const noexcept
    {
        return selectedDispatch.has_value();
    }
};

std::string_view ToString(CaptureStatus status) noexcept;
std::string_view ToString(ArtifactType type) noexcept;
std::string MakeIndexedFolderName(std::string_view prefix, uint64_t id);
std::string PipelineFolderName(PipelineId id);
std::string SessionFolderName(SessionId id);
std::string DispatchFolderName(DispatchId id);
std::filesystem::path PipelineRelativePath(PipelineId id);
std::filesystem::path SessionRelativePath(PipelineId pipelineId, SessionId sessionId);
std::filesystem::path DispatchRelativePath(PipelineId pipelineId, SessionId sessionId,
                                           DispatchId dispatchId);
CaptureRootDecision EvaluateCaptureRoot(
    const std::filesystem::path &captureRoot, bool alreadyExists,
    CaptureRootCollisionPolicy policy = CaptureRootCollisionPolicy::RejectExisting) noexcept;

// CaptureModel is externally synchronized and intended for a single owner.
// Mutations are transactional: IDs and executed indices are committed only with
// their visible mutation, and failed mutations leave all hierarchy links and
// materialized nodes unchanged. Allocation/counter failures and explicit
// complete/error status permanently disable future mutation while leaving the
// terminal model inspectable and serializable.
class CaptureModel
{
  public:
    CaptureModel(
        std::filesystem::path captureRoot, CaptureMetadata metadata,
        CaptureRootCollisionPolicy collisionPolicy = CaptureRootCollisionPolicy::RejectExisting);
    CaptureModel(const CaptureModel &) = delete;
    CaptureModel &operator=(const CaptureModel &) = delete;
    CaptureModel(CaptureModel &&) = delete;
    CaptureModel &operator=(CaptureModel &&) = delete;

    void trackPipeline(PipelineId pipelineId, PipelineMetadata metadata,
                       std::optional<uint64_t> diagnosticHandle = std::nullopt);
    void trackSession(SessionId sessionId, PipelineId pipelineId, uint64_t flags,
                      std::optional<uint64_t> diagnosticHandle = std::nullopt);
    void updateMetadata(CaptureMetadata metadata);

    void updatePipelineFriendlyName(PipelineId pipelineId, std::string friendlyName);
    void updatePipelineMetadata(PipelineId pipelineId, PipelineMetadata metadata);
    void updateSessionFriendlyName(SessionId sessionId, std::string friendlyName);

    void materializeExecutedDispatch(
        SessionId sessionId, DispatchReservation reservation,
        std::optional<uint64_t> diagnosticCommandBufferHandle = std::nullopt);

    void preparePipelineArtifactCommits(PipelineId pipelineId, std::size_t additionalCount);
    void commitPipelineArtifact(PipelineId pipelineId, ArtifactDescriptor artifact) noexcept;
    void prepareDispatchArtifactCommit(DispatchId dispatchId);
    void commitDispatchArtifact(DispatchId dispatchId, ArtifactDescriptor artifact) noexcept;

    // Convenience transactional mutation for model-only callers. The writer
    // uses prepare-before-publication and nothrow commit-after-publication.
    void addPipelineArtifact(PipelineId pipelineId, ArtifactDescriptor artifact);
    void addDispatchArtifact(DispatchId dispatchId, ArtifactDescriptor artifact);

    void markComplete();
    void markError(std::string message);

    const std::filesystem::path &captureRoot() const noexcept;
    const CaptureMetadata &metadata() const noexcept;
    CaptureStatus status() const noexcept;
    CaptureTerminalFailure terminalFailure() const noexcept;
    const std::string &errorMessage() const noexcept;
    CaptureRootCollisionPolicy collisionPolicy() const noexcept;

    const std::map<PipelineId, PipelineNode> &pipelines() const noexcept;
    const std::map<SessionId, SessionNode> &sessions() const noexcept;
    const std::map<DispatchId, DispatchNode> &dispatches() const noexcept;

    bool containsPipeline(PipelineId id) const noexcept;
    bool containsSession(SessionId id) const noexcept;
    bool isPipelineMaterialized(PipelineId id) const noexcept;
    bool isSessionMaterialized(SessionId id) const noexcept;
    const PipelineNode *findPipeline(PipelineId id) const noexcept;
    const SessionNode *findSession(SessionId id) const noexcept;
    const DispatchNode *findDispatch(DispatchId id) const noexcept;

    std::filesystem::path pipelinePath(PipelineId id) const;
    std::filesystem::path sessionPath(SessionId id) const;
    std::filesystem::path dispatchPath(DispatchId id) const;

    bool validateInvariants() const noexcept;

  private:
    PipelineNode &requirePipeline(PipelineId id);
    const PipelineNode &requirePipeline(PipelineId id) const;
    SessionNode &requireSession(SessionId id);
    const SessionNode &requireSession(SessionId id) const;
    void ensureMutable() const;
    void failTerminal(CaptureTerminalFailure failure) noexcept;

    std::filesystem::path captureRoot_;
    CaptureMetadata metadata_;
    CaptureRootCollisionPolicy collisionPolicy_;
    CaptureStatus status_{CaptureStatus::Incomplete};
    CaptureTerminalFailure terminalFailure_{CaptureTerminalFailure::None};
    std::string errorMessage_;

    std::map<PipelineId, PipelineNode> pendingPipelines_;
    std::map<SessionId, SessionNode> pendingSessions_;
    std::map<PipelineId, PipelineNode> pipelines_;
    std::map<SessionId, SessionNode> sessions_;
    std::map<DispatchId, DispatchNode> dispatches_;
};
} // namespace capture
