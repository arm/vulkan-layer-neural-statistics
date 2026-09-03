/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "bounded_queue.hpp"
#include "capture_filesystem.hpp"
#include "capture_model_harness.hpp"
#include "capture_serialization.hpp"
#include "capture_writer.hpp"
#include "fault_injection.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <latch>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace capture
{
template <typename T> struct TestSubmission
{
    TestSubmission(WriterResult<T> value)
        : accepted(value.ok()), error(value.ok() ? WriterError{} : value.error()),
          result(std::move(value))
    {
    }

    bool accepted{false};
    WriterError error;
    WriterResult<T> result;
};

class CaptureWriterHarness : public CaptureWriter
{
  public:
    explicit CaptureWriterHarness(CaptureWriterOptions options)
        : CaptureWriter(options), filter_(options.dispatchFilter)
    {
    }

    TestSubmission<PipelineId> trackPipeline(
        std::optional<uint64_t> diagnosticHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        const PipelineId id = pipelineIds_.peek();
        auto result = CaptureWriter::trackPipeline(id, {}, diagnosticHandle).wait();
        if (!result.ok())
        {
            return {WriterResult<PipelineId>::Failure(result.error())};
        }
        pipelineIds_.commit();
        return {WriterResult<PipelineId>::Success(id)};
    }

    TestSubmission<SessionId> trackSession(PipelineId pipelineId,
                                           std::optional<uint64_t> diagnosticHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        const SessionId id = sessionIds_.peek();
        auto result = CaptureWriter::trackSession(id, pipelineId, 0, diagnosticHandle).wait();
        if (!result.ok())
        {
            return {WriterResult<SessionId>::Failure(result.error())};
        }
        sessionIds_.commit();
        executedIndices_.emplace(id, 0);
        return {WriterResult<SessionId>::Success(id)};
    }

    TestSubmission<DispatchReservation> reserveDispatch(
        SessionId sessionId, std::optional<uint64_t> diagnosticCommandBufferHandle = std::nullopt)
    {
        std::lock_guard lock(identityMutex_);
        auto it = executedIndices_.find(sessionId);
        if (it == executedIndices_.end())
        {
            return {WriterResult<DispatchReservation>::Failure(
                {WriterErrorCode::InvalidRequest, "unknown test session", {}})};
        }
        DispatchReservation reservation{it->second, std::nullopt};
        if (filter_.matches(it->second))
        {
            reservation.selectedDispatch = dispatchIds_.peek();
        }
        auto result = CaptureWriter::materializeDispatch(sessionId, reservation,
                                                         diagnosticCommandBufferHandle)
                          .wait();
        if (!result.ok())
        {
            return {WriterResult<DispatchReservation>::Failure(result.error())};
        }
        ++it->second;
        if (reservation.selected())
        {
            dispatchIds_.commit();
        }
        return {WriterResult<DispatchReservation>::Success(reservation)};
    }

  private:
    DispatchFilter filter_;
    IdAllocator<PipelineId> pipelineIds_;
    IdAllocator<SessionId> sessionIds_;
    IdAllocator<DispatchId> dispatchIds_;
    std::map<SessionId, uint64_t> executedIndices_;
    std::mutex identityMutex_;
};
} // namespace capture

#define CaptureWriter CaptureWriterHarness
#define CaptureModel CaptureModelHarness

namespace
{
using Json = nlohmann::json;
using namespace std::chrono_literals;
int failures = 0;

void Check(bool condition, std::string_view message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class Sandbox
{
  public:
    explicit Sandbox(std::string_view label)
        : root_(std::filesystem::temp_directory_path() /
                (std::string("neural-statistics-writer-") + std::string(label) + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(sequence_.fetch_add(1))))
    {
        std::filesystem::create_directories(root_);
    }

    ~Sandbox() noexcept
    {
        std::error_code ignored;
        std::filesystem::remove_all(root_, ignored);
    }

    std::filesystem::path capture(std::string_view name = "capture") const
    {
        return root_ / std::string(name);
    }

    const std::filesystem::path &root() const noexcept
    {
        return root_;
    }

  private:
    inline static std::atomic<uint64_t> sequence_{0};
    std::filesystem::path root_;
};

capture::CaptureMetadata Metadata(uint32_t mode = 0)
{
    capture::CaptureMetadata metadata;
    metadata.statisticsMode = mode;
    metadata.dispatchFilter = "";
    metadata.devices.push_back(capture::DeviceMetadata{
        capture::LogicalDeviceId(0), "writer-test-device", 0x13B5, 7, 11, std::nullopt});
    return metadata;
}

capture::CaptureWriterOptions Options(const std::filesystem::path &root, std::size_t capacity = 8,
                                      DispatchFilter filter = DispatchFilter::All())
{
    return capture::CaptureWriterOptions{root, Metadata(), filter, capacity};
}

std::string ReadText(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadBytes(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    const std::vector<char> chars((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
    return std::vector<uint8_t>(chars.begin(), chars.end());
}

Json ReadJson(const std::filesystem::path &path)
{
    return Json::parse(ReadText(path));
}

template <typename T> capture::WriterResult<T> Wait(capture::WriterSubmission<T> submission)
{
    return submission.wait();
}

template <typename T> capture::WriterResult<T> Wait(capture::TestSubmission<T> submission)
{
    return std::move(submission.result);
}

std::span<const uint8_t> Bytes(std::initializer_list<uint8_t> values) noexcept
{
    return {values.begin(), values.size()};
}

bool ContainsKey(const Json &value, std::string_view key)
{
    if (value.is_object())
    {
        for (const auto &[name, child] : value.items())
        {
            if (name == key || ContainsKey(child, key))
            {
                return true;
            }
        }
    }
    else if (value.is_array())
    {
        for (const auto &child : value)
        {
            if (ContainsKey(child, key))
            {
                return true;
            }
        }
    }
    return false;
}

bool HasTemporaryFiles(const std::filesystem::path &root)
{
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator it(root, error), end; !error && it != end;
         it.increment(error))
    {
        if (it->path().filename().string().ends_with(".tmp"))
        {
            return true;
        }
    }
    return false;
}

void TestBoundedQueue()
{
    capture::BoundedQueue<int> queue(1);
    Check(queue.push(1), "bounded queue accepts first value");
    std::latch producerStarted(1);
    auto blocked = std::async(std::launch::async,
                              [&queue, &producerStarted]
                              {
                                  producerStarted.count_down();
                                  return queue.push(2);
                              });
    producerStarted.wait();
    Check(queue.waitForBlockedProducers(1),
          "bounded queue deterministically observes producer backpressure");
    const auto first = queue.pop();
    Check(first == 1, "bounded queue preserves first FIFO value");
    Check(blocked.get(), "blocked producer resumes without dropping its value");
    const auto second = queue.pop();
    Check(second == 2, "bounded queue preserves second FIFO value");

    queue.close();
    Check(!queue.push(3), "closed bounded queue rejects pushes");
    Check(!queue.pop().has_value(), "closed drained queue terminates consumers");

    capture::BoundedQueue<int> closing(1);
    Check(closing.push(10), "close-race queue accepts initial value");
    std::latch closingProducerStarted(1);
    auto rejected = std::async(std::launch::async,
                               [&closing, &closingProducerStarted]
                               {
                                   closingProducerStarted.count_down();
                                   return closing.push(11);
                               });
    closingProducerStarted.wait();
    Check(closing.waitForBlockedProducers(1),
          "close-race producer is known blocked before closure");
    closing.close();
    Check(!rejected.get(), "close wakes and rejects a known blocked producer");
    Check(closing.pop() == 10 && !closing.pop().has_value(),
          "close preserves already queued values while terminating after drain");

    capture::BoundedQueue<int> concurrent(4);
    constexpr int producerCount = 4;
    constexpr int valuesPerProducer = 50;
    std::vector<std::thread> producers;
    for (int producer = 0; producer < producerCount; ++producer)
    {
        producers.emplace_back(
            [producer, &concurrent]
            {
                for (int value = 0; value < valuesPerProducer; ++value)
                {
                    (void)concurrent.push(producer * valuesPerProducer + value);
                }
            });
    }
    std::set<int> observed;
    for (int index = 0; index < producerCount * valuesPerProducer; ++index)
    {
        observed.insert(*concurrent.pop());
    }
    for (auto &producer : producers)
    {
        producer.join();
    }
    Check(observed.size() == producerCount * valuesPerProducer,
          "bounded queue supports concurrent producers without drops");
}

void TestClaimAndLifecycle()
{
    Sandbox sandbox("claim");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.snapshot().state == capture::WriterState::NeverStarted,
          "writer begins in never-started state");
    const auto started = writer.start();
    if (!started.ok())
    {
        std::cerr << "writer start error: " << started.error().message << '\n';
    }
    Check(started.ok(), "writer claims a fresh exact root");
    if (!started.ok())
    {
        return;
    }
    Check(std::filesystem::is_directory(root), "fresh exact root is created");
    const Json initial = ReadJson(root / "capture.json");
    Check(initial.at("status") == "incomplete", "initial capture document is incomplete");
    Check(initial.at("pipelines").empty(), "initial capture has no materialized pipelines");
    Check(Wait(writer.finishComplete()).ok(), "normal terminal completion succeeds");
    Check(writer.shutdown().ok(), "normal shutdown joins the writer");
    const Json complete = ReadJson(root / "capture.json");
    Check(complete.at("status") == "complete", "normal shutdown publishes complete capture status");
    Check(!HasTemporaryFiles(root), "successful drain leaves no temporary files");
}

void TestInvalidMetadataRequestIsNonterminal()
{
    Sandbox sandbox("invalid-metadata-request");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), "invalid-metadata writer starts");
    const auto pipeline = Wait(writer.trackPipeline()).value();
    (void) pipeline;

    auto invalid = Metadata();
    invalid.devices.clear();
    const auto rejected = Wait(writer.updateMetadata(std::move(invalid)));
    Check(!rejected.ok() && rejected.error().code == capture::WriterErrorCode::InvalidRequest,
          "metadata removing a referenced device is rejected as an invalid request");

    const auto afterRejection = writer.snapshot();
    Check(afterRejection.state == capture::WriterState::Running, "invalid metadata does not terminalize the writer");

    auto valid = Metadata();
    valid.warnings.emplace_back("capture remains usable after invalid metadata");
    Check(Wait(writer.updateMetadata(std::move(valid))).ok(), "valid metadata is accepted after an invalid request");
    Check(Wait(writer.finishComplete()).ok(), "writer completes after invalid metadata rejection");
    Check(writer.shutdown().ok(), "writer with a rejected metadata request shuts down cleanly");
    const Json complete = ReadJson(root / "capture.json");
    Check(complete.at("status") == "complete"
              && complete.at("warnings") == Json::array({"capture remains usable after invalid metadata"}),
          "later valid metadata is published in the complete capture");
}

void TestFirstPublicationRetry()
{
    Sandbox sandbox("first-publication-retry");
    const auto root = sandbox.capture();
    capture::CaptureFileSystem fileSystem = capture::CaptureFileSystem::ClaimRoot(root);

    bool failedBeforeOpen = false;
    {
        capture::fault::ScopedInjection injection(capture::fault::Point::WriterOpenTemporaryFile,
                                                  capture::fault::Failure::BadAllocation);
        try
        {
            fileSystem.publishText("first.json", "first", capture::FilePublicationMode::Replace);
        }
        catch (const std::bad_alloc&)
        {
            failedBeforeOpen = true;
        }
    }
    Check(failedBeforeOpen, "first publication fails at the pre-open checkpoint");

    bool retrySucceeded = true;
    try
    {
        fileSystem.publishText("first.json", "retry", capture::FilePublicationMode::Replace);
    }
    catch (...)
    {
        retrySucceeded = false;
    }
    fileSystem.preserveClaimedRoot();
    Check(retrySucceeded, "failed first publication leaves the filesystem object reusable");
    Check(ReadText(root / "first.json") == "retry", "retried first publication writes the requested bytes");
    Check(!HasTemporaryFiles(root), "failed publication cleans up its unique temporary file");
}

void TestDirectoryCreationRetry()
{
    Sandbox sandbox("directory-creation-retry");
        const auto root = sandbox.capture();
        capture::CaptureFileSystem fileSystem = capture::CaptureFileSystem::ClaimRoot(root);

        bool sawBadAllocation = false;
        {
            capture::fault::ScopedInjection injection(capture::fault::Point::WriterCreateDirectory,
                                                  capture::fault::Failure::BadAllocation);
            try
            {
                fileSystem.createDirectories("child");
            }
            catch (const std::bad_alloc&)
            {
                sawBadAllocation = true;
            }
        }
    Check(sawBadAllocation, "directory creation fault is reported");
        Check(!std::filesystem::exists(root / "child"),
          "pre-creation failure leaves no partial directory");

        bool retrySucceeded = true;
        try
        {
            fileSystem.createDirectories("child");
        }
        catch (...)
        {
            retrySucceeded = false;
        }
        fileSystem.preserveClaimedRoot();
    Check(retrySucceeded && std::filesystem::is_directory(root / "child"),
          "directory creation remains retryable after a pre-creation failure");
}

void CheckCollision(const std::filesystem::path &root, std::string_view label)
{
    capture::CaptureWriter writer(Options(root));
    const auto result = writer.start();
    Check(!result.ok() && result.error().code == capture::WriterErrorCode::RootCollision,
          std::string("existing ") + std::string(label) + " is rejected");
}

void TestCollisionsAndRace()
{
    Sandbox sandbox("collisions");
    const auto empty = sandbox.capture("empty");
    std::filesystem::create_directory(empty);
    CheckCollision(empty, "empty directory");

    const auto nonempty = sandbox.capture("nonempty");
    std::filesystem::create_directory(nonempty);
    std::ofstream(nonempty / "sentinel.txt") << "keep";
    CheckCollision(nonempty, "nonempty directory");
    Check(ReadText(nonempty / "sentinel.txt") == "keep",
          "collision does not modify existing contents");

    const auto file = sandbox.capture("file");
    std::ofstream(file) << "keep";
    CheckCollision(file, "file");

    const auto target = sandbox.capture("symlink-target");
    const auto link = sandbox.capture("symlink");
    std::filesystem::create_directory(target);
    std::error_code symlinkError;
    std::filesystem::create_directory_symlink(target, link, symlinkError);
    if (!symlinkError)
    {
        CheckCollision(link, "symlink");
    }
    else
    {
        std::cout << "SKIP: symlink collision (host did not permit symlink creation)\n";
    }

    const auto racedRoot = sandbox.capture("race");
    capture::CaptureWriter first(Options(racedRoot));
    capture::CaptureWriter second(Options(racedRoot));
    auto firstStart = std::async(std::launch::async, [&first] { return first.start(); });
    auto secondStart = std::async(std::launch::async, [&second] { return second.start(); });
    const auto firstResult = firstStart.get();
    const auto secondResult = secondStart.get();
    Check(firstResult.ok() != secondResult.ok(), "two-writer root race has exactly one winner");
    Check(
        (firstResult.ok() || firstResult.error().code == capture::WriterErrorCode::RootCollision) &&
            (secondResult.ok() ||
             secondResult.error().code == capture::WriterErrorCode::RootCollision),
        "root-race loser reports a collision");
    if (firstResult.ok())
    {
        Check(first.shutdown().ok(), "first root-race winner shuts down");
    }
    if (secondResult.ok())
    {
        Check(second.shutdown().ok(), "second root-race winner shuts down");
    }
}

void TestRootRelativeNoFollow()
{
    Sandbox sandbox("root-relative-no-follow");
    const auto root = sandbox.capture();
    const auto outside = sandbox.capture("outside");
    std::filesystem::create_directory(outside);
    capture::CaptureFileSystem fileSystem = capture::CaptureFileSystem::ClaimRoot(root);

    std::error_code symlinkError;
    std::filesystem::create_directory_symlink(outside, root / "linked", symlinkError);
    if (symlinkError)
    {
        std::cout << "SKIP: root-relative no-follow test (host did not permit symlink creation)\n";
        return;
    }

    bool rejected = false;
    try
    {
        fileSystem.createDirectories("linked/nested");
    }
    catch (const capture::CaptureFileSystemError&)
    {
        rejected = true;
    }
    fileSystem.preserveClaimedRoot();
    Check(rejected, "root-relative directory traversal refuses a symlink component");
    Check(!std::filesystem::exists(outside / "nested"),
          "refused symlink traversal cannot create output outside the capture root");
}

struct Hierarchy
{
    capture::PipelineId pipeline;
    capture::SessionId session;
    capture::DispatchId dispatch;
};

Hierarchy MaterializeOne(capture::CaptureWriter &writer)
{
    const auto pipeline = Wait(writer.trackPipeline()).value();
    const auto session = Wait(writer.trackSession(pipeline)).value();
    const auto reservation = Wait(writer.reserveDispatch(session)).value();
    return {pipeline, session, *reservation.selectedDispatch};
}

void TestLayoutArtifactsNamesAndConcurrency()
{
    Sandbox sandbox("layout");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root, 2));
    Check(writer.start().ok(), "layout writer starts");

    const auto pipeline0 = Wait(writer.trackPipeline(uint64_t{0x10})).value();
    const auto pipeline1 = Wait(writer.trackPipeline()).value();
    const auto session0 = Wait(writer.trackSession(pipeline0, uint64_t{0x20})).value();
    const auto session1 = Wait(writer.trackSession(pipeline0)).value();
    const auto unselectedSession = Wait(writer.trackSession(pipeline1)).value();
    (void)unselectedSession;

    const std::vector<uint8_t> debugBytes{0, 1, 2, 3, 0xFE, 0xFF};
    Check(Wait(writer.writePipelineBinaryArtifact(pipeline0, "debug_database.bin",
                                                  capture::ArtifactType::DebugDatabase, debugBytes))
              .ok(),
          "static binary artifact queues before materialization");
    Check(Wait(writer.writePipelineTextArtifact(pipeline0, "neural_statistics_info.txt",
                                                capture::ArtifactType::StatisticsInfo,
                                                "line 1\nline 2\n"))
              .ok(),
          "static text artifact queues before materialization");
    Check(Wait(writer.writePipelineBinaryArtifact(
                   pipeline1, "never.bin", capture::ArtifactType::ShaderModule, Bytes({9, 9})))
              .ok(),
          "artifact may queue for a never-selected pipeline");
    Check(!std::filesystem::exists(root / "pipeline_000000") &&
              !std::filesystem::exists(root / "pipeline_000001"),
          "queued static artifacts do not eagerly materialize directories");

    const auto dispatch0 = *Wait(writer.reserveDispatch(session0)).value().selectedDispatch;
    Check(std::filesystem::is_regular_file(root / "pipeline_000000" / "debug_database.bin") &&
              ReadBytes(root / "pipeline_000000" / "debug_database.bin") == debugBytes,
          "lazy materialization writes queued binary artifact byte-exactly");
    Check(ReadText(root / "pipeline_000000" / "neural_statistics_info.txt") == "line 1\nline 2\n",
          "lazy materialization writes queued text artifact byte-exactly");
    Check(!std::filesystem::exists(root / "pipeline_000001"),
          "never-selected pipeline and session produce no directories");

    const std::vector<uint8_t> statistics{7, 6, 5, 4, 3};
    Check(Wait(writer.writeDispatchBinaryArtifact(dispatch0, "statistics_mode0.bin",
                                                  capture::ArtifactType::DispatchStatistics,
                                                  statistics))
              .ok(),
          "dispatch binary artifact publishes");
    Check(ReadBytes(root / "pipeline_000000" / "session_000000" / "dispatch_000000" /
                    "statistics_mode0.bin") == statistics,
          "dispatch artifact output is byte-exact");

    Check(Wait(writer.updatePipelineFriendlyName(pipeline0, "first name")).ok(),
          "first pipeline name update succeeds");
    auto firstName = writer.updatePipelineFriendlyName(pipeline0, "second name");
    auto secondName = writer.updatePipelineFriendlyName(pipeline0, "final name");
    Check(Wait(std::move(firstName)).ok() && Wait(std::move(secondName)).ok(),
          "queued friendly-name updates drain in FIFO order");
    Check(Wait(writer.updateSessionFriendlyName(session0, "session friendly")).ok(),
          "session name update succeeds");

    const auto dispatch1 = *Wait(writer.reserveDispatch(session1)).value().selectedDispatch;
    Check(dispatch1.value() == 1, "multiple sessions receive capture-wide dispatch IDs");
    const auto pipeline2 = Wait(writer.trackPipeline()).value();
    const auto session2 = Wait(writer.trackSession(pipeline2)).value();
    const auto dispatch2 = *Wait(writer.reserveDispatch(session2)).value().selectedDispatch;
    Check(dispatch2.value() == 2 && std::filesystem::is_directory(root / "pipeline_000002"),
          "multiple selected pipelines materialize independently");

    std::mutex idsMutex;
    std::vector<uint64_t> concurrentPipelineIds;
    std::vector<std::thread> producers;
    for (int producer = 0; producer < 4; ++producer)
    {
        producers.emplace_back(
            [&]
            {
                for (int index = 0; index < 8; ++index)
                {
                    const auto result = Wait(writer.trackPipeline());
                    if (result.ok())
                    {
                        std::lock_guard lock(idsMutex);
                        concurrentPipelineIds.push_back(result.value().value());
                    }
                }
            });
    }
    for (auto &producer : producers)
    {
        producer.join();
    }
    std::ranges::sort(concurrentPipelineIds);
    Check(concurrentPipelineIds.size() == 32 &&
              std::adjacent_find(concurrentPipelineIds.begin(), concurrentPipelineIds.end()) ==
                  concurrentPipelineIds.end(),
          "writer supports concurrent producers with unique lossless results");

    Check(Wait(writer.finishComplete()).ok(), "layout capture completes");
    Check(writer.shutdown().ok(), "layout capture joins");

    const auto pipelinePath = root / "pipeline_000000";
    const auto sessionPath = pipelinePath / "session_000000";
    const auto dispatchPath = sessionPath / "dispatch_000000";
    Check(std::filesystem::is_regular_file(root / "capture.json") &&
              std::filesystem::is_regular_file(pipelinePath / "pipeline.json") &&
              std::filesystem::is_regular_file(sessionPath / "session.json") &&
              std::filesystem::is_regular_file(dispatchPath / "dispatch.json"),
          "exact no-container hierarchy is published");
    Check(!std::filesystem::exists(root / "pipelines") &&
              !std::filesystem::exists(pipelinePath / "sessions") &&
              !std::filesystem::exists(sessionPath / "dispatches"),
          "container directories are absent");

    const Json pipelineJson = ReadJson(pipelinePath / "pipeline.json");
    const Json sessionJson = ReadJson(sessionPath / "session.json");
    Check(pipelineJson.at("friendly_name") == "final name" &&
              sessionJson.at("friendly_name") == "session friendly",
          "friendly-name updates rewrite metadata in FIFO order");
    Check(pipelineJson.at("id") == pipeline0.value() && sessionJson.at("id") == session0.value(),
          "friendly-name updates preserve ID-derived paths");
    Check(pipelineJson.at("artifacts").at(0).at("path") == "pipeline_000000/debug_database.bin" &&
              pipelineJson.at("artifacts").at(1).at("path") ==
                  "pipeline_000000/neural_statistics_info.txt",
          "artifact descriptors use deterministic canonical ordering");
    Check(!ContainsKey(ReadJson(root / "capture.json"), "hash") &&
              !ContainsKey(pipelineJson, "hash") &&
              !ContainsKey(ReadJson(dispatchPath / "dispatch.json"), "hash"),
          "writer emits no hashes");
    Check(!HasTemporaryFiles(root), "successful complex capture leaves no temporary files");
}

void TestArtifactNamespaceAndPortableCollisions()
{
    Sandbox sandbox("artifact-names");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), "artifact-name writer starts");
    const auto pipeline = Wait(writer.trackPipeline()).value();

    for (const std::filesystem::path &invalid : {
             std::filesystem::path("capture.json"),
             std::filesystem::path("session.json"),
             std::filesystem::path("dispatch.json"),
             std::filesystem::path("pipeline_000001"),
             std::filesystem::path("old-output.tmp"),
             std::filesystem::path(".capture-internal-future.tmp"),
             std::filesystem::path("stream:ads"),
             std::filesystem::path("trailing."),
             std::filesystem::path("trailing "),
             std::filesystem::path("has space.bin"),
             std::filesystem::path("a/b.bin"),
             std::filesystem::path("a\\b.bin"),
         })
    {
        const auto result = Wait(writer.writePipelineBinaryArtifact(
            pipeline, invalid.string(), capture::ArtifactType::DebugDatabase, Bytes({1})));
        Check(!result.ok() && result.error().code == capture::WriterErrorCode::InvalidRequest,
              std::string("invalid artifact name rejected: ") + invalid.string());
    }

    Check(Wait(writer.writePipelineBinaryArtifact(
                   pipeline, "Data.bin", capture::ArtifactType::DebugDatabase, Bytes({1, 2, 3})))
              .ok(),
          "portable artifact name is accepted");
    Check(Wait(writer.writePipelineBinaryArtifact(
                   pipeline, "data.BIN", capture::ArtifactType::DebugDatabase, Bytes({4, 5, 6}))).ok(),
          "distinct-case artifact name is accepted");

    const auto session = Wait(writer.trackSession(pipeline)).value();
    const auto dispatch = *Wait(writer.reserveDispatch(session)).value().selectedDispatch;
    Check(ReadBytes(root / "pipeline_000000" / "Data.bin") == std::vector<uint8_t>({1, 2, 3}),
          "accepted artifact remains distinct from reserved metadata publication");
    Check(Wait(writer.writeDispatchBinaryArtifact(
                   dispatch, "DISPATCH.JSON", capture::ArtifactType::DispatchStatistics, Bytes({9}))).ok(),
          "distinct-case dispatch artifact name is accepted");

    Check(Wait(writer.finishComplete()).ok() && writer.shutdown().ok(),
          "artifact-name writer completes cleanly");
    Check(ReadJson(root / "pipeline_000000" / "pipeline.json").at("artifacts").size() == 2,
          "distinct-case artifacts enter published descriptors independently");
}

std::vector<std::string> BuildDeterministicCapture(const std::filesystem::path &root)
{
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), "determinism writer starts");
    const auto hierarchy = MaterializeOne(writer);
    Check(Wait(writer.writePipelineBinaryArtifact(hierarchy.pipeline, "z.bin",
                                                  capture::ArtifactType::ShaderModule,
                                                  Bytes({3, 2, 1})))
              .ok(),
          "determinism z artifact writes");
    Check(Wait(writer.writePipelineBinaryArtifact(hierarchy.pipeline, "a.bin",
                                                  capture::ArtifactType::DebugDatabase,
                                                  Bytes({1, 2, 3})))
              .ok(),
          "determinism a artifact writes");
    Check(Wait(writer.writeDispatchTextArtifact(hierarchy.dispatch, "statistics_mode1.bin",
                                                capture::ArtifactType::DispatchStatistics, "bytes"))
              .ok(),
          "determinism dispatch artifact writes");
    Check(Wait(writer.finishComplete()).ok(), "determinism capture completes");
    Check(writer.shutdown().ok(), "determinism writer joins");
    return {
        ReadText(root / "capture.json"),
        ReadText(root / "pipeline_000000" / "pipeline.json"),
        ReadText(root / "pipeline_000000" / "session_000000" / "session.json"),
        ReadText(root / "pipeline_000000" / "session_000000" / "dispatch_000000" / "dispatch.json"),
    };
}

void TestDeterministicDocuments()
{
    Sandbox sandbox("deterministic");
    const auto first = BuildDeterministicCapture(sandbox.capture("first"));
    const auto second = BuildDeterministicCapture(sandbox.capture("second"));
    Check(first == second,
          "identical captures produce byte-identical deterministic JSON documents");
    for (const auto &document : first)
    {
        Check(!document.empty() && document.back() == '\n',
              "deterministic JSON ends with one newline");
    }
}

void TestExplicitErrorAndTerminalBehavior()
{
    Sandbox sandbox("terminal");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), "terminal writer starts");
    (void)MaterializeOne(writer);
    Check(Wait(writer.finishError("explicit writer error")).ok(),
          "explicit terminal error request succeeds");
    const auto rejected = writer.trackPipeline();
    Check(!rejected.accepted && (rejected.error.code == capture::WriterErrorCode::Terminal ||
                                 rejected.error.code == capture::WriterErrorCode::None),
          "enqueue after terminal error is rejected");
    const auto shutdown = writer.shutdown();
    Check(!shutdown.ok(), "shutdown reports explicit terminal error state");
    const Json errorJson = ReadJson(root / "capture.json");
    Check(errorJson.at("status") == "error" && errorJson.at("error") == "explicit writer error",
          "explicit terminal error is published in capture.json");

    capture::CaptureModel model("model-root", Metadata(), DispatchFilter::All());
    model.markComplete();
    bool rejectedMutation = false;
    try
    {
        (void)model.trackPipeline();
    }
    catch (const capture::CaptureTerminalError &)
    {
        rejectedMutation = true;
    }
    Check(rejectedMutation &&
              capture::SerializeCaptureJson(model).find("\"complete\"") != std::string::npos,
          "terminal model remains serializable but rejects future mutations");
}

void TestShutdownAndDestruction()
{
    Sandbox sandbox("shutdown");
    {
        capture::CaptureWriter neverStarted(Options(sandbox.capture("never")));
    }
    Check(!std::filesystem::exists(sandbox.capture("never")),
          "destroying a never-started writer has no filesystem effects");

    const auto automatic = sandbox.capture("automatic");
    {
        capture::CaptureWriter writer(Options(automatic, 2));
        Check(writer.start().ok(), "automatic-destruction writer starts");
        const auto hierarchy = MaterializeOne(writer);
        std::vector<capture::WriterSubmission<void>> pending;
        for (int index = 0; index < 20; ++index)
        {
            pending.push_back(writer.writeDispatchBinaryArtifact(
                hierarchy.dispatch, "artifact_" + std::to_string(index) + ".bin",
                capture::ArtifactType::DispatchStatistics,
                std::vector<uint8_t>(1024, static_cast<uint8_t>(index))));
        }
        Check(writer.snapshot().acceptedJobs >= pending.size(),
              "pending artifact jobs are accepted before destruction");
        // Do not wait on the submissions: destruction itself must close admission,
        // enqueue completion after the accepted work, drain, and join.
    }
    Check(ReadJson(automatic / "capture.json").at("status") == "complete",
          "started-writer destructor drains pending work and publishes complete status");
    Check(std::filesystem::is_regular_file(automatic / "pipeline_000000" / "session_000000" /
                                           "dispatch_000000" / "artifact_19.bin"),
          "destructor drains the final pending artifact without drops");
    Check(!HasTemporaryFiles(automatic), "destructor drain leaves no temporary files");
}

void RunInjectedFailure(capture::fault::Point point, std::string_view label,
                        bool failDirectoryCreation)
{
    Sandbox sandbox(std::string("fault-") + std::string(label));
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), std::string(label) + " fault writer starts");
    const auto pipeline = Wait(writer.trackPipeline()).value();
    const auto session = Wait(writer.trackSession(pipeline)).value();

    capture::WriterResult<void> operation = capture::WriterResult<void>::Success();
    if (failDirectoryCreation)
    {
        capture::fault::ScopedGlobalInjection injection(point, capture::fault::Failure::Unexpected);
        const auto reservation = Wait(writer.reserveDispatch(session));
        operation = reservation.ok() ? capture::WriterResult<void>::Success()
                                     : capture::WriterResult<void>::Failure(reservation.error());
    }
    else
    {
        const auto reservation = Wait(writer.reserveDispatch(session));
        Check(reservation.ok(), std::string(label) + " pre-fault dispatch materializes");
        capture::fault::ScopedGlobalInjection injection(point, capture::fault::Failure::Unexpected);
        operation = Wait(writer.writeDispatchBinaryArtifact(
            *reservation.value().selectedDispatch, std::string(label) + ".bin",
            capture::ArtifactType::DispatchStatistics, Bytes({1, 2, 3, 4})));
    }

    Check(!operation.ok(), std::string(label) + " injected write operation reports failure");
    const auto shutdown = writer.shutdown();
    Check(!shutdown.ok(), std::string(label) + " fatal writer state is reported at shutdown");
    Check(std::filesystem::is_regular_file(root / "capture.json"),
          std::string(label) + " failure preserves a capture document");
    const Json captureJson = ReadJson(root / "capture.json");
    Check(captureJson.at("status") == "error" || captureJson.at("status") == "incomplete",
          std::string(label) + " failure leaves valid error or crash-style incomplete status");
    const auto rejected = writer.trackPipeline();
    Check(!rejected.accepted, std::string(label) + " terminal writer rejects later enqueue");
}

void TestBlockedProducerAndTerminalLifecycle()
{
    {
        Sandbox sandbox("fatal-blocked-producer");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root, 1));
        Check(writer.start().ok(), "fatal-blocked-producer writer starts");
        const auto hierarchy = MaterializeOne(writer);

        capture::fault::ScopedGlobalInjection injection(
            capture::fault::Point::WriterWriteTemporaryFile, capture::fault::Failure::Unexpected);
        capture::fault::ScopedGlobalPause pause(capture::fault::Point::WriterWriteTemporaryFile);
        auto first = writer.writeDispatchBinaryArtifact(
            hierarchy.dispatch, "first.bin", capture::ArtifactType::DispatchStatistics, Bytes({1}));
        pause.waitUntilReached();
        auto queued = writer.writeDispatchBinaryArtifact(hierarchy.dispatch, "queued.bin",
                                                         capture::ArtifactType::DispatchStatistics,
                                                         Bytes({2}));

        std::latch blockedStarted(1);
        auto blocked = std::async(std::launch::async,
                                  [&]
                                  {
                                      blockedStarted.count_down();
                                      return writer.writeDispatchBinaryArtifact(
                                          hierarchy.dispatch, "blocked.bin",
                                          capture::ArtifactType::DispatchStatistics, Bytes({3}));
                                  });
        blockedStarted.wait();
        Check(writer.waitForBlockedSubmitters(1),
              "fatal-failure producer is known blocked on the full queue");
        pause.release();

        const auto blockedSubmission = blocked.get();
        Check(!blockedSubmission.accepted,
              "fatal writer failure wakes and rejects a known blocked producer");
        Check(!Wait(std::move(first)).ok() && !Wait(std::move(queued)).ok(),
              "fatal writer failure rejects both active and queued jobs");
        Check(!writer.shutdown().ok(), "fatal blocked-producer writer reports terminal failure");
    }

    {
        Sandbox sandbox("terminal-admission-race");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root, 1));
        Check(writer.start().ok(), "terminal-admission-race writer starts");
        const auto hierarchy = MaterializeOne(writer);

        capture::fault::ScopedGlobalPause pause(capture::fault::Point::WriterWriteTemporaryFile);
        auto first = writer.writeDispatchBinaryArtifact(
            hierarchy.dispatch, "first.bin", capture::ArtifactType::DispatchStatistics, Bytes({1}));
        pause.waitUntilReached();
        auto queued = writer.writeDispatchBinaryArtifact(hierarchy.dispatch, "queued.bin",
                                                         capture::ArtifactType::DispatchStatistics,
                                                         Bytes({2}));

        std::latch blockedStarted(1);
        auto blocked = std::async(std::launch::async,
                                  [&]
                                  {
                                      blockedStarted.count_down();
                                      return writer.writeDispatchBinaryArtifact(
                                          hierarchy.dispatch, "blocked.bin",
                                          capture::ArtifactType::DispatchStatistics, Bytes({3}));
                                  });
        blockedStarted.wait();
        Check(writer.waitForBlockedSubmitters(1),
              "terminal-race producer is known blocked before terminal admission");

        std::latch terminalStarted(1);
        auto terminal = std::async(std::launch::async,
                                   [&]
                                   {
                                       terminalStarted.count_down();
                                       return writer.finishComplete();
                                   });
        terminalStarted.wait();
        pause.release();

        auto blockedSubmission = blocked.get();
        auto terminalSubmission = terminal.get();
        Check(Wait(std::move(first)).ok() && Wait(std::move(queued)).ok() &&
                  Wait(std::move(blockedSubmission)).ok(),
              "work admitted before terminal request drains losslessly in total order");
        Check(Wait(std::move(terminalSubmission)).ok(),
              "terminal request is admitted after the previously blocked producer");
        Check(writer.shutdown().ok(), "terminal admission race joins cleanly");
    }

    {
        Sandbox sandbox("concurrent-terminal-shutdown");
        capture::CaptureWriter writer(Options(sandbox.capture(), 2));
        Check(writer.start().ok(), "concurrent terminal/shutdown writer starts");
        (void)MaterializeOne(writer);
        std::barrier begin(3);
        auto terminal = std::async(std::launch::async,
                                   [&]
                                   {
                                       begin.arrive_and_wait();
                                       return writer.finishComplete().wait();
                                   });
        auto shutdown = std::async(std::launch::async,
                                   [&]
                                   {
                                       begin.arrive_and_wait();
                                       return writer.shutdown();
                                   });
        begin.arrive_and_wait();
        const auto terminalResult = terminal.get();
        const auto shutdownResult = shutdown.get();
        Check(shutdownResult.ok(), "concurrent shutdown has one synchronized lifecycle owner");
        Check(terminalResult.ok() ||
                  terminalResult.error().code == capture::WriterErrorCode::Terminal,
              "concurrent explicit terminal request either wins or receives a deterministic "
              "terminal rejection");
    }
}

void TestPendingArtifactFailureCoherence()
{
    constexpr std::array<std::string_view, 3> names{"a.bin", "b.bin", "c.bin"};
    for (std::size_t failIndex = 0; failIndex < names.size(); ++failIndex)
    {
        Sandbox sandbox("pending-coherence-" + std::to_string(failIndex));
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root));
        Check(writer.start().ok(), "pending-coherence writer starts");
        const auto pipeline = Wait(writer.trackPipeline()).value();
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            Check(
                Wait(writer.writePipelineBinaryArtifact(
                         pipeline, std::string(names[index]), capture::ArtifactType::DebugDatabase,
                         std::vector<uint8_t>(index + 1, static_cast<uint8_t>(index + 1))))
                    .ok(),
                "pending artifact intent queues without entering the model");
        }
        const auto session = Wait(writer.trackSession(pipeline)).value();
        capture::WriterResult<capture::DispatchReservation> reservation =
            capture::WriterResult<capture::DispatchReservation>::Failure({});
        {
            capture::fault::ScopedGlobalInjection injection(
                capture::fault::Point::WriterRenameTemporaryFile,
                capture::fault::Failure::Unexpected, failIndex + 1);
            reservation = Wait(writer.reserveDispatch(session));
        }
        Check(!reservation.ok(), "injected middle pending-artifact publication fails");
        Check(!writer.shutdown().ok(), "pending-artifact publication failure is terminal");

        const auto pipelinePath = root / "pipeline_000000";
        const Json pipelineJson = ReadJson(pipelinePath / "pipeline.json");
        Check(pipelineJson.at("artifacts").size() == failIndex,
              "error metadata contains exactly the successfully published artifact descriptors");
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            const auto path = pipelinePath / std::string(names[index]);
            if (index < failIndex)
            {
                Check(std::filesystem::is_regular_file(path) &&
                          std::filesystem::file_size(path) == index + 1,
                      "committed pending descriptor resolves to an exact-size regular file");
                Check(pipelineJson.at("artifacts").at(index).at("path") ==
                              "pipeline_000000/" + std::string(names[index]) &&
                          pipelineJson.at("artifacts").at(index).at("size") == index + 1,
                      "committed pending descriptor matches its published file");
            }
            else
            {
                Check(!std::filesystem::exists(path),
                      "failed and later pending artifacts leave no silent orphan output");
            }
        }
        Check(ReadJson(root / "capture.json").at("status") == "error",
              "middle pending-artifact failure publishes coherent terminal error metadata");
    }
}

void TestConstructionClaimGuard()
{
    {
        Sandbox sandbox("invalid-construction");
        const auto root = sandbox.capture();
        auto options = Options(root);
        options.metadata.statisticsMode = 2;
        capture::CaptureWriter writer(std::move(options));
        const auto result = writer.start();
        Check(!result.ok() && result.error().code == capture::WriterErrorCode::InvalidRequest,
              "model-construction validation failure is reported distinctly");
        Check(!std::filesystem::exists(root),
              "model-construction failure occurs before capture-root claim");
    }

    {
        Sandbox sandbox("initial-document-failure");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root));
        capture::WriterResult<void> result = capture::WriterResult<void>::Success();
        {
            capture::fault::ScopedGlobalInjection injection(
                capture::fault::Point::WriterRenameTemporaryFile,
                capture::fault::Failure::Unexpected);
            result = writer.start();
        }
        Check(!result.ok() && result.error().code != capture::WriterErrorCode::InvalidRequest,
              "initial capture-document publication failure is distinct from model validation");
        Check(!std::filesystem::exists(root),
              "initial-publication failure removes the still-empty claimed root");
    }
}

void TestShutdownAllocationFailures()
{
    {
        Sandbox sandbox("terminal-job-allocation");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root));
        Check(writer.start().ok(), "terminal-job allocation writer starts");
        capture::fault::ScopedGlobalInjection injection(
            capture::fault::Point::WriterTerminalJobCreation,
            capture::fault::Failure::BadAllocation);
        Check(writer.shutdown().ok(),
              "terminal-job allocation failure falls back to automatic drain and completion");
        Check(ReadJson(root / "capture.json").at("status") == "complete",
              "fallback shutdown still publishes complete status");
    }

    {
        Sandbox sandbox("future-retrieval-allocation");
        capture::CaptureWriter writer(Options(sandbox.capture()));
        Check(writer.start().ok(), "future-retrieval writer starts");
        capture::fault::ScopedGlobalInjection injection(
            capture::fault::Point::WriterFutureRetrieval, capture::fault::Failure::BadAllocation);
        const auto submission = writer.trackPipeline();
        Check(!submission.accepted &&
                  submission.error.code == capture::WriterErrorCode::InternalFailure,
              "future retrieval allocation failure is translated at the no-throw submission "
              "boundary");
        Check(writer.shutdown().ok(), "future-retrieval failure does not poison writer shutdown");
    }

    {
        Sandbox sandbox("destructor-automatic-allocation");
        const auto root = sandbox.capture();
        capture::fault::ScopedGlobalInjection injection(
            capture::fault::Point::WriterDestructorAutomaticCompletion,
            capture::fault::Failure::BadAllocation);
        {
            capture::CaptureWriter writer(Options(root));
            Check(writer.start().ok(), "destructor-allocation writer starts");
            (void)MaterializeOne(writer);
        }
        const Json document = ReadJson(root / "capture.json");
        Check(document.at("status") == "incomplete" || document.at("status") == "error",
              "destructor allocation failure is contained and leaves a valid non-complete capture");
    }
}

void TestPersistentWriterAllocationFailures()
{
    {
        Sandbox sandbox("persistent-startup-allocation");
        capture::CaptureWriter writer(Options(sandbox.capture()));
        capture::fault::ScopedGlobalInjection primaryFailure(
            capture::fault::Point::WriterClaimCreateRoot, capture::fault::Failure::BadAllocation);
        capture::fault::ScopedGlobalPersistentAllocationFailure persistentFailure;
        const auto started = writer.start();
        Check(!started.ok() && started.error().code == capture::WriterErrorCode::OutOfHostMemory,
              "startup bad_alloc plus persistent translation failure returns code-only "
              "out-of-host-memory");
        const auto snapshot = writer.snapshot();
        Check(snapshot.state == capture::WriterState::Error &&
                  snapshot.terminalErrorCode == capture::WriterErrorCode::OutOfHostMemory,
              "startup waiter is notified with deterministic nonallocating terminal state");
        const auto shutdown = writer.shutdown();
        Check(!shutdown.ok() && shutdown.error().code == capture::WriterErrorCode::OutOfHostMemory,
              "startup persistent-allocation failure joins without deadlock or terminate");
    }

    {
        Sandbox sandbox("persistent-job-allocation");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root));
        Check(writer.start().ok(), "persistent accepted-job allocation writer starts");
        const auto hierarchy = MaterializeOne(writer);
        capture::WriterResult<void> result = capture::WriterResult<void>::Success();
        capture::WriterResult<void> shutdown = capture::WriterResult<void>::Success();
        {
            capture::fault::ScopedGlobalInjection primaryFailure(
                capture::fault::Point::WriterWriteTemporaryFile,
                capture::fault::Failure::BadAllocation);
            // Rich translation succeeds at the first allocation checkpoint;
            // every later delivery/publication checkpoint fails persistently.
            capture::fault::ScopedGlobalPersistentAllocationFailure persistentFailure(2);
            result = Wait(writer.writeDispatchBinaryArtifact(
                hierarchy.dispatch, "persistent.bin", capture::ArtifactType::DispatchStatistics,
                Bytes({1, 2, 3})));
            const auto snapshot = writer.snapshot();
            Check(snapshot.state == capture::WriterState::Error &&
                      snapshot.terminalErrorCode == capture::WriterErrorCode::OutOfHostMemory,
                  "accepted-job publication failure retains canonical out-of-host-memory state");
            const auto rejected = writer.trackPipeline();
            Check(!rejected.accepted &&
                      rejected.error.code == capture::WriterErrorCode::OutOfHostMemory,
                  "accepted-job persistent failure closes admission with a code-only rejection");
            shutdown = writer.shutdown();
        }
        Check(!result.ok() && result.error().code == capture::WriterErrorCode::OutOfHostMemory,
              "accepted job receives deterministic code-only failure when rich delivery allocation "
              "fails");
        Check(!shutdown.ok() && shutdown.error().code == capture::WriterErrorCode::OutOfHostMemory,
              "accepted-job persistent failure closes the queue and joins cleanly");
    }

    {
        Sandbox sandbox("snapshot-allocation");
        const auto root = sandbox.capture();
        capture::CaptureWriter writer(Options(root));
        Check(writer.start().ok(), "snapshot-allocation writer starts");
        const auto hierarchy = MaterializeOne(writer);
        {
            capture::fault::ScopedGlobalInjection primaryFailure(
                capture::fault::Point::WriterWriteTemporaryFile,
                capture::fault::Failure::Unexpected);
            Check(!Wait(writer.writeDispatchBinaryArtifact(
                            hierarchy.dispatch, "snapshot.bin",
                            capture::ArtifactType::DispatchStatistics, Bytes({4, 5, 6})))
                       .ok(),
                  "snapshot-allocation writer reaches a rich terminal failure");
        }
        const auto richError = writer.terminalError();
        Check(richError.code == capture::WriterErrorCode::InternalFailure &&
                  !richError.message.empty(),
              "rich terminal diagnostics are retained when allocation succeeds");

        capture::WriterResult<void> shutdown = capture::WriterResult<void>::Success();
        {
            capture::fault::ScopedGlobalPersistentAllocationFailure persistentFailure;
            const auto snapshot = writer.snapshot();
            Check(snapshot.state == capture::WriterState::Error &&
                      snapshot.terminalErrorCode == richError.code &&
                      snapshot.completedJobs == writer.snapshot().completedJobs,
                  "nonallocating snapshot remains available while diagnostic allocation "
                  "persistently fails");
            bool diagnosticsFailed = false;
            try
            {
                (void)writer.terminalError();
            }
            catch (const std::bad_alloc &)
            {
                diagnosticsFailed = true;
            }
            Check(diagnosticsFailed,
                  "separate rich terminal diagnostic acquisition remains explicitly fallible");
            shutdown = writer.shutdown();
        }
        Check(!shutdown.ok() && shutdown.error().code == richError.code,
              "shutdown degrades rich terminal diagnostics to their canonical code while "
              "allocation fails");
    }
}

void TestFinalCompletionFailure()
{
    Sandbox sandbox("final-completion-failure");
    const auto root = sandbox.capture();
    capture::CaptureWriter writer(Options(root));
    Check(writer.start().ok(), "final-completion fault writer starts");
    capture::WriterResult<void> completion = capture::WriterResult<void>::Success();
    {
        // finishComplete first republishes the valid incomplete document, then
        // publishes complete. Fail only the latter rename.
        capture::fault::ScopedGlobalInjection injection(
            capture::fault::Point::WriterRenameTemporaryFile, capture::fault::Failure::Unexpected,
            2);
        completion = Wait(writer.finishComplete());
    }
    Check(!completion.ok(), "failed final complete publication reports an error");
    Check(!writer.shutdown().ok(),
          "failed final complete publication leaves terminal writer error");
    const Json captureJson = ReadJson(root / "capture.json");
    Check(captureJson.at("status") == "error",
          "failed final complete publication is recovered as terminal error, not complete");
}

void TestInjectedFailures()
{
    RunInjectedFailure(capture::fault::Point::WriterCreateDirectory, "directory", true);
    RunInjectedFailure(capture::fault::Point::WriterOpenTemporaryFile, "open", false);
    RunInjectedFailure(capture::fault::Point::WriterWriteTemporaryFile, "write", false);
    RunInjectedFailure(capture::fault::Point::WriterRenameTemporaryFile, "rename", false);
    TestFinalCompletionFailure();
}
} // namespace

int main(int argc, char **argv)
{
    const auto run = [&](std::string_view name, auto operation)
    {
        if (argc == 1 || name == argv[1])
        {
            operation();
        }
    };
    run("bounded-queue", TestBoundedQueue);
    run("claim-lifecycle", TestClaimAndLifecycle);
    run("invalid-metadata", TestInvalidMetadataRequestIsNonterminal);
    run("first-publication-retry", TestFirstPublicationRetry);
    run("directory-creation-retry", TestDirectoryCreationRetry);
    run("collisions", TestCollisionsAndRace);
    run("no-follow", TestRootRelativeNoFollow);
    run("layout", TestLayoutArtifactsNamesAndConcurrency);
    run("artifact-names", TestArtifactNamespaceAndPortableCollisions);
    run("determinism", TestDeterministicDocuments);
    run("explicit-terminal", TestExplicitErrorAndTerminalBehavior);
    run("shutdown", TestShutdownAndDestruction);
    run("blocked-lifecycle", TestBlockedProducerAndTerminalLifecycle);
    run("pending-coherence", TestPendingArtifactFailureCoherence);
    run("claim-guard", TestConstructionClaimGuard);
    run("shutdown-allocation", TestShutdownAllocationFailures);
    run("persistent-allocation", TestPersistentWriterAllocationFailures);
    run("injected-failures", TestInjectedFailures);

    if (failures != 0)
    {
        std::cerr << failures << " capture writer test(s) failed\n";
        return 1;
    }
    std::cout << "All capture writer tests passed\n";
    return 0;
}
