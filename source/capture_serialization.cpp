/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "capture_serialization.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

#include <nlohmann/json.hpp>

namespace capture
{
namespace
{
using Json = nlohmann::ordered_json;

Json OptionalHandle(const std::optional<uint64_t> &handle)
{
    return handle.has_value() ? Json(*handle) : Json(nullptr);
}

Json ArtifactJson(const ArtifactDescriptor &artifact)
{
    return Json{
        {"path", artifact.path.generic_string()},
        {"type", ToString(artifact.type)},
        {"size", artifact.size},
    };
}

Json ArtifactsJson(const std::vector<ArtifactDescriptor> &artifacts)
{
    std::vector<const ArtifactDescriptor *> sorted;
    sorted.reserve(artifacts.size());
    for (const auto &artifact : artifacts)
    {
        sorted.push_back(&artifact);
    }
    std::ranges::sort(
        sorted, {}, [](const ArtifactDescriptor *artifact)
        { return std::tuple(artifact->path.generic_string(), artifact->type, artifact->size); });

    Json result = Json::array();
    for (const auto *artifact : sorted)
    {
        result.push_back(ArtifactJson(*artifact));
    }
    return result;
}

template <typename IdType> std::vector<IdType> SortedIds(const std::vector<IdType> &ids)
{
    std::vector<IdType> result = ids;
    std::ranges::sort(result);
    return result;
}

std::string Dump(const Json &json)
{
    return json.dump(4) + '\n';
}
} // namespace

std::string SerializeCaptureJson(const CaptureModel &model)
{
    return SerializeCaptureJson(model, model.status(), model.errorMessage());
}

std::string SerializeCaptureJson(const CaptureModel &model, CaptureStatus status,
                                 std::string_view errorMessage)
{
    Json pipelines = Json::array();
    for (const auto &[id, _] : model.pipelines())
    {
        pipelines.push_back(Json{
            {"id", id.value()},
            {"path", (PipelineRelativePath(id) / names::kPipelineDocument).generic_string()},
        });
    }

    const auto &metadata = model.metadata();
    Json devices = Json::array();
    for (const auto &device : metadata.devices)
    {
        devices.push_back(Json{
            {"id", device.id.value()},
            {"name", device.name},
            {"vendor_id", device.vendorId.has_value() ? Json(*device.vendorId) : Json(nullptr)},
            {"device_id", device.deviceId.has_value() ? Json(*device.deviceId) : Json(nullptr)},
            {"driver_version",
             device.driverVersion.has_value() ? Json(*device.driverVersion) : Json(nullptr)},
            {"api_version",
             device.apiVersion.has_value() ? Json(*device.apiVersion) : Json(nullptr)},
        });
    }
    Json warnings = Json::array();
    for (const auto &warning : metadata.warnings)
    {
        warnings.push_back(warning);
    }
    Json json{
        {"schema_version", kCaptureSchemaVersion},
        {"status", ToString(status)},
        {"error", status == CaptureStatus::Error ? Json(errorMessage) : Json(nullptr)},
        {"capture",
         Json{
             {"layer_name", metadata.layerName},
             {"layer_version", metadata.layerVersion},
             {"commit_identity", metadata.commitIdentity},
             {"layer_implementation_version", metadata.layerImplementationVersion},
             {"statistics_mode", metadata.statisticsMode},
             {"dispatch_filter", metadata.dispatchFilter},
         }},
        {"devices", std::move(devices)},
        {"warnings", std::move(warnings)},
        {"pipelines", std::move(pipelines)},
    };
    return Dump(json);
}

std::string SerializePipelineJson(const CaptureModel &model, PipelineId pipelineId)
{
    const PipelineNode *pipeline = model.findPipeline(pipelineId);
    if (pipeline == nullptr)
    {
        throw std::out_of_range("pipeline is not materialized");
    }

    Json sessions = Json::array();
    for (const SessionId id : SortedIds(pipeline->sessions))
    {
        sessions.push_back(Json{
            {"id", id.value()},
            {"path",
             (SessionRelativePath(pipelineId, id) / names::kSessionDocument).generic_string()},
        });
    }
    Json resources = Json::array();
    for (const auto &resource : pipeline->metadata.resourceBindings)
    {
        resources.push_back(Json{
            {"descriptor_set", resource.descriptorSet},
            {"binding", resource.binding},
            {"array_element", resource.arrayElement},
        });
    }
    Json specializationEntries = Json::array();
    for (const auto &entry : pipeline->metadata.shader.specializationEntries)
    {
        specializationEntries.push_back(Json{
            {"constant_id", entry.constantId},
            {"offset", entry.offset},
            {"size", entry.size},
        });
    }
    return Dump(Json{
        {"schema_version", kPipelineSchemaVersion},
        {"id", pipelineId.value()},
        {"device_id", pipeline->metadata.deviceId.value()},
        {"friendly_name", pipeline->friendlyName},
        {"diagnostic_handle", OptionalHandle(pipeline->diagnosticHandle)},
        {"flags", pipeline->metadata.flags},
        {"pipeline_layout_handle", OptionalHandle(pipeline->metadata.layoutHandle)},
        {"resource_bindings", std::move(resources)},
        {"vendor_options", pipeline->metadata.vendorOptions.has_value()
                               ? Json(*pipeline->metadata.vendorOptions)
                               : Json(nullptr)},
        {"identifier_only", pipeline->metadata.identifierOnly},
        {"foreign_processing_engine", pipeline->metadata.foreignProcessingEngine},
        {"statistics_enabled", pipeline->metadata.statisticsEnabled},
        {"shader",
         Json{
             {"module_id", pipeline->metadata.shader.moduleId.has_value()
                               ? Json(pipeline->metadata.shader.moduleId->value())
                               : Json(nullptr)},
             {"spirv_available", pipeline->metadata.shader.spirvAvailable},
             {"friendly_name", pipeline->metadata.shader.friendlyName},
             {"entry_point", pipeline->metadata.shader.entryPoint},
             {"specialization_entries", std::move(specializationEntries)},
             {"specialization_data_size", pipeline->metadata.shader.specializationDataSize},
         }},
        {"artifacts", ArtifactsJson(pipeline->artifacts)},
        {"sessions", std::move(sessions)},
    });
}

std::string SerializeSessionJson(const CaptureModel &model, SessionId sessionId)
{
    const SessionNode *session = model.findSession(sessionId);
    if (session == nullptr)
    {
        throw std::out_of_range("session is not materialized");
    }

    Json dispatches = Json::array();
    for (const DispatchId id : SortedIds(session->dispatches))
    {
        dispatches.push_back(Json{
            {"id", id.value()},
            {"path",
             (DispatchRelativePath(session->pipelineId, sessionId, id) / names::kDispatchDocument)
                 .generic_string()},
        });
    }
    return Dump(Json{
        {"schema_version", kSessionSchemaVersion},
        {"id", sessionId.value()},
        {"pipeline_id", session->pipelineId.value()},
        {"device_id", session->deviceId.value()},
        {"friendly_name", session->friendlyName},
        {"diagnostic_handle", OptionalHandle(session->diagnosticHandle)},
        {"flags", session->flags},
        {"dispatches", std::move(dispatches)},
    });
}

std::string SerializeDispatchJson(const CaptureModel &model, DispatchId dispatchId)
{
    const DispatchNode *dispatch = model.findDispatch(dispatchId);
    if (dispatch == nullptr)
    {
        throw std::out_of_range("unknown dispatch ID");
    }
    return Dump(Json{
        {"schema_version", kDispatchSchemaVersion},
        {"id", dispatchId.value()},
        {"session_id", dispatch->sessionId.value()},
        {"executed_index", dispatch->executedIndex},
        {"diagnostic_command_buffer_handle",
         OptionalHandle(dispatch->diagnosticCommandBufferHandle)},
        {"artifacts", ArtifactsJson(dispatch->artifacts)},
    });
}
} // namespace capture
