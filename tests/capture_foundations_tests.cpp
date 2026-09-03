/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_model.hpp"
#include "capture_model_harness.hpp"
#include "capture_serialization.hpp"
#include "dispatch_filter.hpp"
#include "layer_options.hpp"
#include "layer_setting_resolver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#define CaptureModel CaptureModelHarness

namespace
{
using Json = nlohmann::json;
int failures = 0;

std::optional<std::string> ReadEnvironment(std::string_view name)
{
    const char *value = std::getenv(std::string(name).c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
}

void SetEnvironment(std::string_view name, const std::optional<std::string> &value)
{
    if (value)
    {
        setenv(std::string(name).c_str(), value->c_str(), 1);
    }
    else
    {
        unsetenv(std::string(name).c_str());
    }
}

class ScopedEnvironment
{
  public:
    explicit ScopedEnvironment(std::string name)
        : name_(std::move(name)), original_(ReadEnvironment(name_))
    {
    }

    ~ScopedEnvironment() noexcept
    {
        try
        {
            SetEnvironment(name_, original_);
        }
        catch (...)
        {
        }
    }
    void set(std::optional<std::string> value)
    {
        SetEnvironment(name_, value);
    }

  private:
    std::string name_;
    std::optional<std::string> original_;
};

class SettingsSandbox
{
  public:
    SettingsSandbox()
        : root_(std::filesystem::temp_directory_path() /
                ("neural-statistics-settings-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))),
          settingsPath_("VK_LAYER_SETTINGS_PATH"), captureFolder_("VK_LAYER_CAPTURE_FOLDER"),
          statisticsMode_("VK_LAYER_STATISTICS_MODE"), dispatchFilter_("VK_LAYER_DISPATCH_FILTER")
    {
        std::filesystem::create_directories(root_);
        settingsPath_.set(root_.string());
        captureFolder_.set(std::nullopt);
        statisticsMode_.set(std::nullopt);
        dispatchFilter_.set(std::nullopt);
    }

    ~SettingsSandbox() noexcept
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    void write(std::string_view contents)
    {
        std::ofstream file(root_ / "vk_layer_settings.txt", std::ios::trunc);
        file << contents;
    }

    ScopedEnvironment &captureFolder()
    {
        return captureFolder_;
    }
    ScopedEnvironment &statisticsMode()
    {
        return statisticsMode_;
    }
    ScopedEnvironment &dispatchFilter()
    {
        return dispatchFilter_;
    }

  private:
    std::filesystem::path root_;
    ScopedEnvironment settingsPath_;
    ScopedEnvironment captureFolder_;
    ScopedEnvironment statisticsMode_;
    ScopedEnvironment dispatchFilter_;
};

void Check(bool condition, std::string_view message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void CheckValid(std::string_view text, bool capturesAll, uint64_t first, uint64_t last,
                std::string_view canonical)
{
    const auto parsed = ParseDispatchFilter(text);
    Check(static_cast<bool>(parsed), std::string("valid filter rejected: ") + std::string(text));
    if (!parsed)
    {
        return;
    }
    Check(parsed.filter->capturesAll() == capturesAll, "capture-all classification");
    Check(parsed.filter->first() == first, "filter first endpoint");
    Check(parsed.filter->last() == last, "filter last endpoint");
    Check(parsed.filter->canonicalText() == canonical, "filter canonical text");
}

void CheckInvalid(std::string_view text)
{
    const auto parsed = ParseDispatchFilter(text);
    Check(!parsed, std::string("invalid filter accepted: ") + std::string(text));
    Check(!parsed.error.empty(), "invalid filter reports an error");
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

void TestDispatchFilter()
{
    CheckValid("", true, 0, std::numeric_limits<uint64_t>::max(), "");
    CheckValid(" \t\r\n", true, 0, std::numeric_limits<uint64_t>::max(), "");
    CheckValid("0", false, 0, 0, "0");
    CheckValid("17", false, 17, 17, "17");
    CheckValid("3-8", false, 3, 8, "3-8");
    CheckValid("  3-8\t", false, 3, 8, "3-8");
    CheckValid("18446744073709551615", false, std::numeric_limits<uint64_t>::max(),
               std::numeric_limits<uint64_t>::max(), "18446744073709551615");

    const auto inclusive = ParseDispatchFilter("3-5");
    Check(inclusive.filter->matches(3), "range includes first endpoint");
    Check(inclusive.filter->matches(5), "range includes last endpoint");
    Check(!inclusive.filter->matches(2) && !inclusive.filter->matches(6),
          "range excludes outside values");

    for (const std::string_view invalid : {
             "-1",
             "4-3",
             "18446744073709551616",
             "1-18446744073709551616",
             "abc",
             "1,2",
             "1;2",
             "1-2-3",
             "1--2",
             "1-",
             "-2",
             "1 -2",
             "1- 2",
             "1 - 2",
             "1 2",
             "+1",
             "1/2",
         })
    {
        CheckInvalid(invalid);
    }
}

void TestIdsAndPaths()
{
    capture::IdAllocator<capture::PipelineId> pipelines;
    capture::IdAllocator<capture::SessionId> sessions;
    capture::IdAllocator<capture::DispatchId> dispatches;
    Check(pipelines.allocate().value() == 0 && pipelines.allocate().value() == 1,
          "pipeline IDs increment stably");
    Check(sessions.allocate().value() == 0 && sessions.allocate().value() == 1,
          "session sequence is independent");
    Check(dispatches.allocate().value() == 0, "dispatch sequence is independent");

    capture::IdAllocator<capture::PipelineId> finalId(std::numeric_limits<uint64_t>::max());
    Check(finalId.allocate().value() == std::numeric_limits<uint64_t>::max(),
          "the final capture-local ID is allocated without wrapping");
    bool exhausted = false;
    try
    {
        (void)finalId.allocate();
    }
    catch (const std::overflow_error &)
    {
        exhausted = true;
    }
    Check(exhausted, "capture-local ID exhaustion throws instead of wrapping");

    capture::MonotonicCounter finalIndex(std::numeric_limits<uint64_t>::max());
    Check(finalIndex.allocate() == std::numeric_limits<uint64_t>::max(),
          "the final executed index is allocated without wrapping");
    exhausted = false;
    try
    {
        (void)finalIndex.allocate();
    }
    catch (const std::overflow_error &)
    {
        exhausted = true;
    }
    Check(exhausted, "executed-index exhaustion throws instead of wrapping");

    static_assert(!std::is_copy_constructible_v<capture::CaptureModel>);
    static_assert(!std::is_copy_assignable_v<capture::CaptureModel>);
    static_assert(!std::is_move_constructible_v<capture::CaptureModel>);
    static_assert(!std::is_move_assignable_v<capture::CaptureModel>);

    Check(capture::PipelineFolderName(capture::PipelineId(0)) == "pipeline_000000",
          "pipeline folder is zero padded");
    Check(capture::SessionFolderName(capture::SessionId(42)) == "session_000042",
          "session folder is zero padded");
    Check(capture::DispatchFolderName(capture::DispatchId(999999)) == "dispatch_999999",
          "dispatch folder width is deterministic");
    Check(capture::DispatchFolderName(capture::DispatchId(1000000)) == "dispatch_1000000",
          "folder IDs are not truncated after six digits");

    const auto hierarchy = capture::DispatchRelativePath(
        capture::PipelineId(0), capture::SessionId(1), capture::DispatchId(2));
    Check(hierarchy.generic_string() == "pipeline_000000/session_000001/dispatch_000002",
          "hierarchy has no container directories");
}

capture::CaptureMetadata Metadata(uint32_t mode = 0, std::string filter = {})
{
    capture::CaptureMetadata metadata;
    metadata.statisticsMode = mode;
    metadata.dispatchFilter = std::move(filter);
    metadata.devices.push_back(capture::DeviceMetadata{capture::LogicalDeviceId(0), "test-device",
                                                       0x13B5, std::nullopt, std::nullopt,
                                                       std::nullopt});
    return metadata;
}

void TestLazyModelAndReservation()
{
    const auto parsed = ParseDispatchFilter("1-2");
    capture::CaptureModel model("capture-root", Metadata(1, "1-2"), *parsed.filter);

    const auto pipeline0 = model.trackPipeline(uint64_t{0x100});
    const auto pipeline1 = model.trackPipeline();
    const auto session0 = model.trackSession(pipeline0, uint64_t{0x200});
    const auto session1 = model.trackSession(pipeline0);
    const auto session2 = model.trackSession(pipeline1, uint64_t{0x300});

    Check(model.pipelines().empty() && model.sessions().empty() && model.dispatches().empty(),
          "tracking does not materialize capture nodes");

    const auto unselected = model.reserveExecutedDispatch(session0, uint64_t{0xA});
    Check(unselected.executedIndex == 0 && !unselected.selected(),
          "first per-session occurrence is reserved but filtered out");
    Check(model.pipelines().empty(), "unselected dispatch keeps pipeline and session lazy");

    const auto selected0 = model.reserveExecutedDispatch(session0, uint64_t{0xB});
    Check(selected0.executedIndex == 1 && selected0.selected(),
          "second occurrence is selected with zero-based indexing");
    Check(model.pipelines().size() == 1 && model.sessions().size() == 1 &&
              model.dispatches().size() == 1,
          "selected dispatch materializes its pipeline and session");
    Check(model.findPipeline(pipeline1) == nullptr && model.findSession(session1) == nullptr,
          "unselected tracked identities remain absent from capture model");

    const auto selected1 = model.reserveExecutedDispatch(session0, uint64_t{0xC});
    Check(selected1.executedIndex == 2 && selected1.selected(),
          "inclusive range selects final endpoint");
    const auto filtered = model.reserveExecutedDispatch(session0, uint64_t{0xD});
    Check(filtered.executedIndex == 3 && !filtered.selected(),
          "range stops after inclusive endpoint");

    const auto otherSession0 = model.reserveExecutedDispatch(session1, uint64_t{0xE});
    Check(otherSession0.executedIndex == 0, "each session starts its own executed index at zero");
    const auto otherSession1 = model.reserveExecutedDispatch(session1, uint64_t{0xF});
    Check(otherSession1.executedIndex == 1 && otherSession1.selected(),
          "second session selects independently of first session count");

    model.reserveExecutedDispatch(session2, uint64_t{0x10});
    const auto pipeline1Selected = model.reserveExecutedDispatch(session2, uint64_t{0x11});
    Check(pipeline1Selected.selected() && model.pipelines().size() == 2 &&
              model.sessions().size() == 3,
          "multiple pipelines and sessions materialize independently");

    Check(selected0.selectedDispatch->value() == 0 && selected1.selectedDispatch->value() == 1 &&
              otherSession1.selectedDispatch->value() == 2 &&
              pipeline1Selected.selectedDispatch->value() == 3,
          "selected dispatch IDs use a stable capture-wide sequence");

    const auto *dispatch0 = model.findDispatch(*selected0.selectedDispatch);
    const auto *dispatch1 = model.findDispatch(*selected1.selectedDispatch);
    Check(dispatch0->diagnosticCommandBufferHandle == uint64_t{0xB} &&
              dispatch1->diagnosticCommandBufferHandle == uint64_t{0xC},
          "command buffers are optional diagnostics, not indexing identities");
    Check(dispatch0->executedIndex == 1 && dispatch1->executedIndex == 2,
          "executed indices advance independently of command-buffer identity");
}

void TestNamesHandlesArtifactsAndSerialization()
{
    capture::CaptureModel model("capture-root", Metadata(), DispatchFilter::All());
    const auto pipeline = model.trackPipeline();
    const auto session = model.trackSession(pipeline, uint64_t{0x222});
    model.addPipelineArtifact(
        pipeline, {"pipeline_000000/debug_database.bin", capture::ArtifactType::DebugDatabase, 64});

    const auto pipelinePathBefore = model.pipelinePath(pipeline);
    const auto sessionPathBefore = model.sessionPath(session);
    model.updatePipelineFriendlyName(pipeline, "friendly/pipeline name");
    model.updateSessionFriendlyName(session, "renamed session");
    const auto reservation = model.reserveExecutedDispatch(session);
    model.addDispatchArtifact(*reservation.selectedDispatch,
                              {"pipeline_000000/session_000000/dispatch_000000/statistics.bin",
                               capture::ArtifactType::DispatchStatistics, 128});

    model.updatePipelineFriendlyName(pipeline, "later pipeline name");
    model.updateSessionFriendlyName(session, "later session name");
    Check(model.pipelinePath(pipeline) == pipelinePathBefore &&
              model.sessionPath(session) == sessionPathBefore,
          "friendly-name updates never change established ID paths");
    Check(!model.findPipeline(pipeline)->diagnosticHandle.has_value(),
          "pipeline diagnostic handle is optional");
    Check(model.findSession(session)->diagnosticHandle == uint64_t{0x222},
          "session diagnostic handle is retained when supplied");
    Check(!model.findDispatch(*reservation.selectedDispatch)
               ->diagnosticCommandBufferHandle.has_value(),
          "dispatch command-buffer diagnostic is optional");

    const std::string captureJson0 = capture::SerializeCaptureJson(model);
    const std::string captureJson1 = capture::SerializeCaptureJson(model);
    Check(captureJson0 == captureJson1, "capture serialization is deterministic");
    const Json captureJson = Json::parse(captureJson0);
    Check(captureJson.at("schema_version") == capture::kCaptureSchemaVersion,
          "capture schema version is serialized");
    Check(captureJson.at("status") == "incomplete" && captureJson.at("error").is_null(),
          "capture starts incomplete without an error");
    Check(captureJson.at("devices").at(0).at("id") == 0 &&
              captureJson.at("devices").at(0).at("name") == "test-device",
          "capture document serializes logical device metadata");
    Check(captureJson.at("pipelines").at(0).at("path") == "pipeline_000000/pipeline.json",
          "capture references pipeline document deterministically");

    const Json pipelineJson = Json::parse(capture::SerializePipelineJson(model, pipeline));
    const Json sessionJson = Json::parse(capture::SerializeSessionJson(model, session));
    const Json dispatchJson =
        Json::parse(capture::SerializeDispatchJson(model, *reservation.selectedDispatch));
    Check(pipelineJson.at("friendly_name") == "later pipeline name" &&
              pipelineJson.at("device_id") == 0,
          "materialized pipeline metadata receives friendly-name and device identity updates");
    Check(sessionJson.at("friendly_name") == "later session name" &&
              sessionJson.at("device_id") == 0,
          "materialized session metadata receives friendly-name updates");
    Check(dispatchJson.at("executed_index") == 0,
          "dispatch serializes zero-based occurrence index");
    Check(dispatchJson.at("artifacts").at(0).at("type") == "dispatch_statistics" &&
              dispatchJson.at("artifacts").at(0).at("size") == 128,
          "artifact descriptors serialize path, type, and size");
    Check(!ContainsKey(pipelineJson, "hash") && !ContainsKey(dispatchJson, "hash"),
          "artifact schema contains no hashes");

    model.markError("writer failed");
    const Json errorJson = Json::parse(capture::SerializeCaptureJson(model));
    Check(errorJson.at("status") == "error" && errorJson.at("error") == "writer failed",
          "capture supports error completion state");

    capture::CaptureModel completeModel("capture-root", Metadata(), DispatchFilter::All());
    completeModel.markComplete();
    const Json completeJson = Json::parse(capture::SerializeCaptureJson(completeModel));
    Check(completeJson.at("status") == "complete" && completeJson.at("error").is_null(),
          "capture supports complete terminal state without an error");
}

void PopulateArtifactOrderingModel(capture::CaptureModel &model, bool reverse)
{
    const auto pipeline = model.trackPipeline();
    const auto session = model.trackSession(pipeline);
    model.updatePipelineFriendlyName(pipeline, "pipeline \"quoted\" \\ slash\nline");
    model.updateSessionFriendlyName(session, "session\tname \"quoted\"");
    const auto dispatch = model.reserveExecutedDispatch(session);

    const capture::ArtifactDescriptor pipelineA{"artifacts/zeta \"quoted\".bin",
                                                capture::ArtifactType::StatisticsInfo, 3};
    const capture::ArtifactDescriptor pipelineB{"artifacts/alpha\nline.bin",
                                                capture::ArtifactType::DebugDatabase, 9};
    const capture::ArtifactDescriptor dispatchA{"dispatch/zeta name.bin",
                                                capture::ArtifactType::DispatchStatistics, 12};
    const capture::ArtifactDescriptor dispatchB{"dispatch/alpha \"quoted\".bin",
                                                capture::ArtifactType::StatisticsInfo, 7};

    if (reverse)
    {
        model.addPipelineArtifact(pipeline, pipelineB);
        model.addPipelineArtifact(pipeline, pipelineA);
        model.addDispatchArtifact(*dispatch.selectedDispatch, dispatchB);
        model.addDispatchArtifact(*dispatch.selectedDispatch, dispatchA);
    }
    else
    {
        model.addPipelineArtifact(pipeline, pipelineA);
        model.addPipelineArtifact(pipeline, pipelineB);
        model.addDispatchArtifact(*dispatch.selectedDispatch, dispatchA);
        model.addDispatchArtifact(*dispatch.selectedDispatch, dispatchB);
    }
    model.markError("writer \"failed\" at \\path\nnext line");
}

void TestCanonicalArtifactOrderingAndEscaping()
{
    capture::CaptureModel forward("capture-root", Metadata(), DispatchFilter::All());
    capture::CaptureModel reverse("capture-root", Metadata(), DispatchFilter::All());
    PopulateArtifactOrderingModel(forward, false);
    PopulateArtifactOrderingModel(reverse, true);

    const auto pipeline = capture::PipelineId(0);
    const auto session = capture::SessionId(0);
    const auto dispatch = capture::DispatchId(0);
    Check(capture::SerializePipelineJson(forward, pipeline) ==
              capture::SerializePipelineJson(reverse, pipeline),
          "pipeline JSON is byte-identical for opposite artifact insertion orders");
    Check(capture::SerializeDispatchJson(forward, dispatch) ==
              capture::SerializeDispatchJson(reverse, dispatch),
          "dispatch JSON is byte-identical for opposite artifact insertion orders");

    const Json captureJson = Json::parse(capture::SerializeCaptureJson(forward));
    const Json pipelineJson = Json::parse(capture::SerializePipelineJson(forward, pipeline));
    const Json sessionJson = Json::parse(capture::SerializeSessionJson(forward, session));
    const Json dispatchJson = Json::parse(capture::SerializeDispatchJson(forward, dispatch));
    Check(captureJson.at("error") == "writer \"failed\" at \\path\nnext line",
          "capture errors round-trip JSON special characters");
    Check(pipelineJson.at("friendly_name") == "pipeline \"quoted\" \\ slash\nline",
          "pipeline names round-trip JSON special characters");
    Check(sessionJson.at("friendly_name") == "session\tname \"quoted\"",
          "session names round-trip JSON special characters");
    Check(pipelineJson.at("artifacts").at(0).at("path") == "artifacts/alpha\nline.bin" &&
              pipelineJson.at("artifacts").at(1).at("path") == "artifacts/zeta \"quoted\".bin",
          "pipeline artifacts use canonical path/type/size order and preserve special characters");
    Check(dispatchJson.at("artifacts").at(0).at("path") == "dispatch/alpha \"quoted\".bin" &&
              dispatchJson.at("artifacts").at(1).at("path") == "dispatch/zeta name.bin",
          "dispatch artifacts use canonical path/type/size order and portable path serialization");
}

void TestPortableArtifactNamespace()
{
    using capture::names::PortableNameKey;
    using capture::names::ValidateArtifactFileName;

    for (const std::string_view valid : {
             "debug_database.bin",
             "statistics-mode_1.bin",
             "A0.txt",
         })
    {
        const auto result = ValidateArtifactFileName(std::filesystem::path(valid));
        Check(result.valid && !result.collisionKey.empty(),
              std::string("portable artifact name accepted: ") + std::string(valid));
    }

    for (const std::string_view invalid : {
             "",
             ".",
             "..",
             "a/b.bin",
             "a\\b.bin",
             "stream:ads",
             "control\x01.bin",
             "name ",
             "name.",
             "has space.bin",
             "é.bin",
             "capture.json",
             "session.json",
             "dispatch.json",
             "pipeline_000000",
             "dispatch_7",
             "artifact.tmp",
             ".capture-internal-owned.tmp",
         })
    {
        const auto result = ValidateArtifactFileName(std::filesystem::path(invalid));
        Check(!result.valid, std::string("non-portable or reserved artifact name rejected: ") +
                                 std::string(invalid));
    }

    Check(PortableNameKey("Data.BIN") != PortableNameKey("data.bin"),
          "artifact collision keys preserve filesystem case sensitivity");
}

void TestCollisionPolicyAndValidation()
{
    using namespace capture;
    Check(EvaluateCaptureRoot({}, false) == CaptureRootDecision::RejectEmptyPath,
          "empty capture root is rejected");
    Check(EvaluateCaptureRoot("new-root", false) == CaptureRootDecision::CreateNew,
          "nonexistent exact capture root may be created");
    Check(EvaluateCaptureRoot("existing-root", true) == CaptureRootDecision::RejectExisting,
          "existing capture root is never merged or overwritten");

    bool threw = false;
    try
    {
        CaptureModel invalid({}, Metadata(), DispatchFilter::All());
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    Check(threw, "model validates non-empty exact capture root");

    threw = false;
    try
    {
        CaptureModel invalid("root", Metadata(2), DispatchFilter::All());
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    Check(threw, "model validates statistics mode invariant");
}

LayerOptions MakeLayerOptions(const char *captureFolder, const char *statisticsMode,
                              const char *dispatchFilter)
{
    constexpr const char *layerName = "VK_LAYER_LGL_neural_statistics";
    const VkLayerSettingEXT settings[]{
        {layerName, "capture_folder", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &captureFolder},
        {layerName, "statistics_mode", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &statisticsMode},
        {layerName, "dispatch_filter", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &dispatchFilter},
    };
    const VkLayerSettingsCreateInfoEXT settingsInfo{
        VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        nullptr,
        static_cast<uint32_t>(std::size(settings)),
        settings,
    };
    const VkInstanceCreateInfo instanceInfo{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, &settingsInfo, 0, nullptr, 0, nullptr, 0, nullptr,
    };
    return LayerOptions(&instanceInfo);
}

void TestDefaultCaptureRootNaming()
{
    const LayerOptions first;
    const LayerOptions second;
    const auto firstRoot = first.getCaptureRoot();
    const auto secondRoot = second.getCaptureRoot();

#ifdef __ANDROID__
    Check(firstRoot.parent_path() == std::filesystem::path("/data/local/tmp"),
          "Android default capture uses the documented parent directory");
#else
    Check(firstRoot.parent_path() == std::filesystem::current_path(),
          "Linux default capture uses the current working directory");
#endif
    Check(firstRoot.filename().string().starts_with("neural_statistics-"),
          "default capture directory includes the neural-statistics timestamp prefix");
    Check(firstRoot != secondRoot,
          "successive default capture roots do not collide within one process");
}

void TestLayerSettingSelection()
{
    LayerSettingValue api{LayerSettingSource::Api, VK_SUCCESS, "api"};
    auto selected = SelectLayerSettingValue(std::string(""), std::string("file"), api);
    Check(selected.source == LayerSettingSource::Environment && selected.value.empty(),
          "an explicit empty environment value overrides file and API values");
    selected = SelectLayerSettingValue(std::nullopt, std::string(""), api);
    Check(selected.source == LayerSettingSource::File && selected.value.empty(),
          "an explicit empty file value overrides an API value");
    selected = SelectLayerSettingValue(std::nullopt, std::nullopt, api);
    Check(selected.source == LayerSettingSource::Api && selected.value == "api",
          "API settings are selected when environment and file values are absent");
}

void TestLayerOptionsApiTransport()
{
    SettingsSandbox sandbox;

    const auto valid = MakeLayerOptions("exact-capture-root", "1", " 2-4 ");
    Check(valid.isValid(), "API settings accept the final three-setting contract");
    Check(valid.getCaptureRoot() == std::filesystem::path("exact-capture-root"),
          "API capture_folder is the exact root");
    Check(valid.getStatsModeIndex() == 1 &&
              valid.getStatsMode() == VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS1_ARM,
          "API enum mode 1 maps to the Vulkan mode");
    Check(valid.getDispatchFilterText() == "2-4" && valid.getDispatchFilter().matches(2) &&
              valid.getDispatchFilter().matches(4) && !valid.getDispatchFilter().matches(5),
          "API dispatch_filter stores the production matcher and canonical text");

    const auto blankFilter = MakeLayerOptions("root", "0", " \t\r\n ");
    Check(blankFilter.isValid() && blankFilter.getDispatchFilter().capturesAll(),
          "blank API dispatch_filter means capture-all");
    const auto invalidMode = MakeLayerOptions("root", "2", "");
    Check(!invalidMode.isValid() && !invalidMode.getError().empty(),
          "API settings reject an invalid statistics mode");
    const auto emptyMode = MakeLayerOptions("root", "", "");
    Check(!emptyMode.isValid() && !emptyMode.getError().empty(),
          "API settings reject explicitly empty statistics_mode");
    const auto invalidFilter = MakeLayerOptions("root", "0", "1 - 2");
    Check(!invalidFilter.isValid() && !invalidFilter.getError().empty(),
          "API settings reject malformed dispatch_filter");
    const auto invalidRoot = MakeLayerOptions("", "0", "");
    Check(!invalidRoot.isValid() && !invalidRoot.getError().empty(),
          "API settings reject explicitly empty capture_folder");
}

void TestLayerOptionsFileTransport()
{
    SettingsSandbox sandbox;
    sandbox.write("lgl_neural_statistics.capture_folder = file-root\n"
                  "lgl_neural_statistics.statistics_mode = 1\n"
                  "lgl_neural_statistics.dispatch_filter =    \n");
    const LayerOptions blankFilter(nullptr);
    Check(blankFilter.isValid() &&
              blankFilter.getCaptureRoot() == std::filesystem::path("file-root") &&
              blankFilter.getStatsModeIndex() == 1 && blankFilter.getDispatchFilter().capturesAll(),
          "the real settings-file loader preserves blank dispatch_filter as capture-all");

    std::ifstream checkedIn(NEURAL_STATISTICS_SETTINGS_PATH);
    const std::string checkedInText((std::istreambuf_iterator<char>(checkedIn)),
                                    std::istreambuf_iterator<char>());
    sandbox.write(checkedInText);
    const LayerOptions sample(nullptr);
    Check(sample.isValid() && sample.getDispatchFilter().capturesAll(),
          "the checked-in blank-filter sample is valid through the real settings-file path");

    sandbox.write("lgl_neural_statistics.capture_folder =\n"
                  "lgl_neural_statistics.statistics_mode = 0\n"
                  "lgl_neural_statistics.dispatch_filter =\n");
    const LayerOptions emptyRoot(nullptr);
    Check(!emptyRoot.isValid() && !emptyRoot.getError().empty(),
          "settings files reject explicitly empty capture_folder");

    sandbox.write("lgl_neural_statistics.capture_folder = file-root\n"
                  "lgl_neural_statistics.statistics_mode =\n"
                  "lgl_neural_statistics.dispatch_filter =\n");
    const LayerOptions emptyMode(nullptr);
    Check(!emptyMode.isValid() && !emptyMode.getError().empty(),
          "settings files reject explicitly empty statistics_mode");

    sandbox.write("lgl_neural_statistics.capture_folder = file-root\n"
                  "lgl_neural_statistics.statistics_mode = 0\n"
                  "lgl_neural_statistics.dispatch_filter = 7\n");
    const auto fileOverridesApi = MakeLayerOptions("api-root", "1", "9");
    Check(fileOverridesApi.isValid() &&
              fileOverridesApi.getCaptureRoot() == std::filesystem::path("file-root") &&
              fileOverridesApi.getStatsModeIndex() == 0 &&
              fileOverridesApi.getDispatchFilterText() == "7",
          "settings-file values preserve their precedence over API settings");
}

void TestLayerOptionsEnvironmentTransport()
{
    SettingsSandbox sandbox;
    sandbox.write("lgl_neural_statistics.capture_folder = file-root\n"
                  "lgl_neural_statistics.statistics_mode = 0\n"
                  "lgl_neural_statistics.dispatch_filter = 5-8\n");

    sandbox.captureFolder().set(std::string("environment-root"));
    sandbox.statisticsMode().set(std::string("1"));
    sandbox.dispatchFilter().set(std::string(""));
    const auto emptyDispatch = ReadEnvironment("VK_LAYER_DISPATCH_FILTER");
    const bool supportsEmptyEnvironment = emptyDispatch.has_value() && emptyDispatch->empty();
    if (supportsEmptyEnvironment)
    {
        const LayerOptions environmentOverrides(nullptr);
        Check(
            environmentOverrides.isValid() &&
                environmentOverrides.getCaptureRoot() ==
                    std::filesystem::path("environment-root") &&
                environmentOverrides.getStatsModeIndex() == 1 &&
                environmentOverrides.getDispatchFilter().capturesAll(),
            "environment values, including explicit empty dispatch_filter, override file settings");
    }

    sandbox.dispatchFilter().set(std::string(" \t "));
    const LayerOptions whitespaceFilter(nullptr);
    Check(whitespaceFilter.isValid() && whitespaceFilter.getDispatchFilter().capturesAll(),
          "whitespace environment dispatch_filter means capture-all");

    if (supportsEmptyEnvironment)
    {
        sandbox.captureFolder().set(std::string(""));
        const LayerOptions emptyRoot(nullptr);
        Check(!emptyRoot.isValid() && !emptyRoot.getError().empty(),
              "environment settings reject explicitly empty capture_folder");

        sandbox.captureFolder().set(std::string("environment-root"));
        sandbox.statisticsMode().set(std::string(""));
        const LayerOptions emptyMode(nullptr);
        Check(!emptyMode.isValid() && !emptyMode.getError().empty(),
              "environment settings reject explicitly empty statistics_mode");
    }
}

bool ValidateVkConfig12EnumSetting(const Json &setting)
{
    if (!setting.is_object() || setting.value("type", "") != "ENUM")
    {
        return false;
    }

    static const std::vector<std::string_view> allowedProperties{
        "key",      "env",      "type",      "label",   "description",
        "detailed", "url",      "platforms", "status",  "deprecated_by_key",
        "view",     "expanded", "flags",     "default", "dependence",
        "settings", "messages"};
    for (const auto &[name, _] : setting.items())
    {
        if (std::ranges::find(allowedProperties, name) == allowedProperties.end())
        {
            return false;
        }
    }

    for (const std::string_view required :
         {"key", "label", "description", "type", "default", "flags"})
    {
        if (!setting.contains(required) ||
            (required != "flags" && !setting.at(required).is_string()))
        {
            return false;
        }
    }
    if (!setting.at("flags").is_array() || setting.at("flags").empty())
    {
        return false;
    }

    std::map<std::string, bool> keys;
    for (const auto &flag : setting.at("flags"))
    {
        if (!flag.is_object() || !flag.contains("key") || !flag.at("key").is_string() ||
            !flag.contains("label") || !flag.at("label").is_string() ||
            !flag.contains("description") || !flag.at("description").is_string())
        {
            return false;
        }
        if (!keys.emplace(flag.at("key").get<std::string>(), true).second)
        {
            return false;
        }
    }
    return keys.contains(setting.at("default").get<std::string>());
}

void TestManifestContract()
{
    std::ifstream input(NEURAL_STATISTICS_MANIFEST_PATH);
    Check(input.good(), "manifest can be opened");
    if (!input.good())
    {
        return;
    }
    Json manifest;
    input >> manifest;
    Check(manifest.at("file_format_version") == "1.2.0",
          "settings manifest uses schema version 1.2.0");
    const auto &settings = manifest.at("layer").at("features").at("settings");
    Check(settings.size() == 3, "manifest exposes exactly three settings");

    std::map<std::string, Json> byKey;
    for (const auto &setting : settings)
    {
        byKey.emplace(setting.at("key").get<std::string>(), setting);
    }
    Check(byKey.contains("capture_folder") && byKey.contains("statistics_mode") &&
              byKey.contains("dispatch_filter"),
          "manifest exposes the final setting keys");
    Check(byKey.at("capture_folder").at("type") == "SAVE_FOLDER",
          "capture folder uses VkConfig folder selector");
    Check(byKey.at("dispatch_filter").at("type") == "STRING", "dispatch filter uses text control");
    const auto &statisticsMode = byKey.at("statistics_mode");
    Check(ValidateVkConfig12EnumSetting(statisticsMode),
          "statistics mode satisfies the focused VkConfig 1.2 ENUM schema contract");
    const auto &modes = statisticsMode.at("flags");
    Check(modes.size() == 2 && modes.at(0).at("key") == "0" && modes.at(1).at("key") == "1" &&
              statisticsMode.at("default") == "0",
          "statistics enum contains only string modes 0 and 1 with string default 0");

    Json incompatible = statisticsMode;
    incompatible["enum"] = incompatible.at("flags");
    incompatible.erase("flags");
    Check(!ValidateVkConfig12EnumSetting(incompatible),
          "the manifest validator rejects the loader-incompatible enum property shape");
    Check(!byKey.contains("debug_database") && !byKey.contains("statistics_info"),
          "artifact toggles are not user-visible");
}
} // namespace

int main()
{
    TestDispatchFilter();
    TestIdsAndPaths();
    TestLazyModelAndReservation();
    TestNamesHandlesArtifactsAndSerialization();
    TestCanonicalArtifactOrderingAndEscaping();
    TestPortableArtifactNamespace();
    TestCollisionPolicyAndValidation();
    TestDefaultCaptureRootNaming();
    TestLayerSettingSelection();
    TestLayerOptionsApiTransport();
    TestLayerOptionsFileTransport();
    TestLayerOptionsEnvironmentTransport();
    TestManifestContract();

    if (failures != 0)
    {
        std::cerr << failures << " capture foundation test(s) failed\n";
        return 1;
    }
    std::cout << "All capture foundation tests passed\n";
    return 0;
}
