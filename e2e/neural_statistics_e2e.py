#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

"""Target-side orchestrator and exact structural validator for neural-statistics E2E runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from typing import Any

SCHEMA_VERSION = 2
MATRIX_SCHEMA_VERSION = 1
CASE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?$")
BAD_LOG_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"device[- ]?loss|VK_ERROR_DEVICE_LOST",
        r"gpu fault|data abort|assert(?:ion)? failed",
        r"segmentation fault|segfault|SIGSEGV|SIGABRT|abort trap",
    )
]
FORBIDDEN_NAMES = {"pipelines", "sessions", "dispatches"}
OLD_FLAT_RE = re.compile(r"pipeline_0x[0-9a-f]+.*\.bin$", re.IGNORECASE)
KNOWN_ARTIFACT_TYPES = {"debug_database", "statistics_info", "shader_module", "dispatch_statistics"}
STATIC_ARTIFACT_ORDER = ("debug_database", "statistics_info", "shader_module")
STATIC_ARTIFACT_TYPES = set(STATIC_ARTIFACT_ORDER)
INTERNAL_TEMP_RE = re.compile(
    r"^\.capture-internal-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}\.tmp$")
ORCHESTRATOR_ENV_KEYS = {
    "VK_LAYER_CAPTURE_FOLDER",
    "VK_LAYER_STATISTICS_MODE",
    "VK_LAYER_DISPATCH_FILTER",
    "VK_LAYER_TEST_POST_GRAPH_COPY_COUNT",
    "VK_INSTANCE_LAYERS",
    "VK_LAYER_PATH",
    "VK_ADD_LAYER_PATH",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LOADER_LAYERS_ENABLE",
    "VK_LOADER_LAYERS_DISABLE",
    "LD_LIBRARY_PATH",
}
CAPTURE_KEYS = {"schema_version", "status", "error", "capture", "devices", "warnings", "pipelines"}
CAPTURE_SETTINGS_KEYS = {
    "layer_name", "layer_version", "commit_identity", "layer_implementation_version",
    "statistics_mode", "dispatch_filter",
}
DEVICE_KEYS = {"id", "name", "vendor_id", "device_id", "driver_version", "api_version"}
PIPELINE_KEYS = {
    "schema_version", "id", "friendly_name", "diagnostic_handle", "flags", "pipeline_layout_handle",
    "resource_bindings", "vendor_options", "identifier_only", "foreign_processing_engine",
    "device_id", "statistics_enabled", "shader", "artifacts", "sessions",
}
SHADER_KEYS = {
    "module_id", "spirv_available", "friendly_name", "entry_point",
    "specialization_entries", "specialization_data_size",
}
SESSION_KEYS = {"schema_version", "id", "pipeline_id", "device_id", "friendly_name", "diagnostic_handle", "flags", "dispatches"}
DISPATCH_KEYS = {"schema_version", "id", "session_id", "executed_index", "diagnostic_command_buffer_handle", "artifacts"}
ARTIFACT_KEYS = {"path", "type", "size"}
REFERENCE_KEYS = {"id", "path"}


class E2EError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise E2EError(f"cannot parse JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        class MatrixVariables(dict[str, str]):
            def __missing__(self, key: str) -> str:
                if key in {"CAPTURE_ROOT", "RUN_DIR"}:
                    return "{" + key + "}"
                raise KeyError(key)
        try:
            return value.format_map(MatrixVariables(variables))
        except KeyError as exc:
            raise E2EError(f"undefined matrix variable {exc.args[0]!r} in {value!r}") from exc
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    return value


def runtime_substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(variables)
        except KeyError as exc:
            raise E2EError(f"undefined runtime variable {exc.args[0]!r} in {value!r}") from exc
    if isinstance(value, list):
        return [runtime_substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: runtime_substitute(item, variables) for key, item in value.items()}
    return value


def validate_case_name(name: str) -> None:
    if not CASE_NAME_RE.fullmatch(name):
        raise E2EError(
            f"unsafe case name {name!r}; expected 1-80 lowercase ASCII letters, digits, and interior hyphens")


def parse_matrix(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise E2EError("matrix must be an object with schema_version 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise E2EError("matrix cases must be a non-empty array")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise E2EError("every case must have a string name")
        validate_case_name(case["name"])
        if case["name"] in seen:
            raise E2EError(f"duplicate case name {case['name']}")
        seen.add(case["name"])
        if "environment" in case:
            environment = case["environment"]
            if isinstance(environment, dict):
                reserved = sorted(str(key) for key in environment if str(key) in ORCHESTRATOR_ENV_KEYS)
                if reserved:
                    raise E2EError("case environment overrides orchestrator-owned keys: " + ", ".join(reserved))
            raise E2EError("arbitrary per-case environment overrides are not supported")
    return data


def select_cases(matrix: dict[str, Any], names: list[str], tags: list[str]) -> list[dict[str, Any]]:
    cases = matrix["cases"]
    by_name = {case["name"]: case for case in cases}
    missing = sorted(set(names) - by_name.keys())
    if missing:
        raise E2EError("unknown cases: " + ", ".join(missing))
    selected = [by_name[name] for name in names] if names else list(cases)
    if tags:
        selected = [case for case in selected if set(tags).issubset(set(case.get("tags", [])))]
    if not selected:
        raise E2EError("case selection is empty")
    return selected


def _expect(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(document: dict[str, Any], expected: set[str], errors: list[str], label: str) -> None:
    actual = set(document)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{label}: missing fields {missing}")
    if extra:
        errors.append(f"{label}: unknown fields {extra}")


def _reject_forbidden_json_fields(value: Any, errors: list[str], label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = str(key).lower()
            if "hash" in lower:
                errors.append(f"{label}: hash field is forbidden: {key}")
            if lower in {"constants", "graph_constants", "data_graph_constants", "constant_data"}:
                errors.append(f"{label}: data-graph constants field is forbidden: {key}")
            _reject_forbidden_json_fields(child, errors, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_json_fields(child, errors, f"{label}[{index}]")


def _relative_posix(value: Any, errors: list[str], label: str) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: path must be a nonempty string")
        return None
    if "\x00" in value or "\\" in value or value.startswith("/") or value.startswith("//"):
        errors.append(f"{label}: path is not a canonical relative POSIX path: {value!r}")
        return None
    if re.match(r"^[A-Za-z]:", value):
        errors.append(f"{label}: drive-qualified path is forbidden: {value!r}")
        return None
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        errors.append(f"{label}: empty, dot, or parent path component is forbidden: {value!r}")
        return None
    canonical = PurePosixPath(*components).as_posix()
    if canonical != value or PurePosixPath(value).is_absolute():
        errors.append(f"{label}: path is not canonical: {value!r}")
        return None
    return canonical


def _resolved_regular(root: Path, relative: str, errors: list[str], label: str, directory: bool = False) -> Path | None:
    current = root
    for component in relative.split("/"):
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            errors.append(f"{label}: path does not exist: {relative}: {exc}")
            return None
        if stat.S_ISLNK(mode):
            errors.append(f"{label}: symlink is forbidden: {relative}")
            return None
    try:
        resolved = current.resolve(strict=True)
        if os.path.commonpath([str(root), str(resolved)]) != str(root):
            errors.append(f"{label}: resolved path escapes capture root: {relative}")
            return None
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: cannot resolve path {relative}: {exc}")
        return None
    mode = current.lstat().st_mode
    if directory and not stat.S_ISDIR(mode):
        errors.append(f"{label}: expected ordinary directory: {relative}")
        return None
    if not directory and not stat.S_ISREG(mode):
        errors.append(f"{label}: expected ordinary regular file: {relative}")
        return None
    return current


def _inventory(root: Path, errors: list[str]) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    internal_temporaries: set[str] = set()
    for current, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dir_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                errors.append(f"symlink is forbidden: {relative}")
                dir_names.remove(name)
            elif not stat.S_ISDIR(mode):
                errors.append(f"non-directory entry in directory list: {relative}")
                dir_names.remove(name)
            else:
                directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                errors.append(f"symlink is forbidden: {relative}")
            elif not stat.S_ISREG(mode):
                errors.append(f"non-regular public output: {relative}")
            else:
                files.add(relative)
            _expect(name not in FORBIDDEN_NAMES, errors, f"forbidden container name: {relative}")
            if INTERNAL_TEMP_RE.fullmatch(name):
                internal_temporaries.add(relative)
            else:
                _expect(not name.endswith(".tmp"), errors, f"temporary file remains: {relative}")
                _expect(not name.startswith(".capture-internal-"), errors,
                        f"internal publication file remains: {relative}")
            _expect(not OLD_FLAT_RE.match(name), errors, f"old flat artifact remains: {relative}")
    return files, directories, internal_temporaries


def _read_document(root: Path, relative: str, errors: list[str], expected_keys: set[str], label: str) -> dict[str, Any] | None:
    path = _resolved_regular(root, relative, errors, label)
    if path is None:
        return None
    try:
        value = load_json(path)
    except E2EError as exc:
        errors.append(str(exc))
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}: root is not an object")
        return None
    _require_keys(value, expected_keys, errors, label)
    _reject_forbidden_json_fields(value, errors, label)
    _expect(value.get("schema_version") == SCHEMA_VERSION, errors,
            f"{label}: schema_version is not {SCHEMA_VERSION}")
    return value


def _validate_reference_list(document: dict[str, Any], key: str, errors: list[str], label: str) -> list[tuple[int, str]]:
    refs = document.get(key)
    if not isinstance(refs, list):
        errors.append(f"{label}.{key}: expected array")
        return []
    result: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()
    for index, ref in enumerate(refs):
        ref_label = f"{label}.{key}[{index}]"
        if not isinstance(ref, dict):
            errors.append(f"{ref_label}: reference is not an object")
            continue
        _require_keys(ref, REFERENCE_KEYS, errors, ref_label)
        ref_id = ref.get("id")
        relative = _relative_posix(ref.get("path"), errors, ref_label + ".path")
        if not _is_int(ref_id) or ref_id < 0:
            errors.append(f"{ref_label}.id: expected nonnegative integer")
            continue
        if ref_id in seen_ids:
            errors.append(f"{ref_label}: duplicate referenced ID {ref_id}")
        seen_ids.add(ref_id)
        if relative is None:
            continue
        if relative in seen_paths:
            errors.append(f"{ref_label}: duplicate referenced path {relative}")
        seen_paths.add(relative)
        result.append((ref_id, relative))
    return result


def _validate_handle(value: Any, errors: list[str], label: str) -> None:
    _expect(value is None or (_is_int(value) and value >= 0), errors, f"{label}: expected null or nonnegative integer")


def _validate_artifacts(root: Path, owner_dir: str, document: dict[str, Any], errors: list[str],
                        expected_files: set[str], artifact_classes: set[str], artifact_paths: list[tuple[str, str]],
                        raw_mode: int | None, pipeline_shader_module_id: int | None = None,
                        static_type_counts: dict[str, int] | None = None,
                        allow_missing_dispatch_statistics: bool = False,
                        allow_missing_static_suffix: bool = False) -> dict[str, int]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append(f"{owner_dir}: artifacts is not an array")
        return {}
    seen_descriptors: set[tuple[str, str, int]] = set()
    seen_paths: set[str] = set()
    type_counts: dict[str, int] = {}
    static_type_sequence: list[str] = []
    for index, descriptor in enumerate(artifacts):
        label = f"{owner_dir}.artifacts[{index}]"
        if not isinstance(descriptor, dict):
            errors.append(f"{label}: descriptor is not an object")
            continue
        _require_keys(descriptor, ARTIFACT_KEYS, errors, label)
        relative = _relative_posix(descriptor.get("path"), errors, label + ".path")
        artifact_type = descriptor.get("type")
        size = descriptor.get("size")
        if artifact_type not in KNOWN_ARTIFACT_TYPES:
            errors.append(f"{label}: unknown artifact type {artifact_type!r}")
            continue
        if not _is_int(size) or size < 0:
            errors.append(f"{label}: artifact size must be a nonnegative integer")
            continue
        if relative is None:
            continue
        name = PurePosixPath(relative).name
        if size == 0 and not (artifact_type == "statistics_info" and name == "neural_statistics_info.txt"):
            errors.append(f"{label}: only a text statistics-info artifact may be empty")
        identity = (relative, artifact_type, size)
        if identity in seen_descriptors:
            errors.append(f"{label}: duplicate artifact descriptor")
        seen_descriptors.add(identity)
        if relative in seen_paths:
            errors.append(f"{label}: duplicate artifact path {relative}")
        seen_paths.add(relative)
        type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
        if artifact_type in STATIC_ARTIFACT_TYPES:
            static_type_sequence.append(artifact_type)
        artifact_classes.add(artifact_type)
        artifact_paths.append((artifact_type, relative))
        expected_files.add(relative)
        artifact = _resolved_regular(root, relative, errors, label)
        if artifact is not None:
            actual_size = artifact.stat().st_size
            _expect(actual_size == size, errors, f"{label}: size mismatch: expected {size}, got {actual_size}")
        artifact_parent = PurePosixPath(relative).parent
        _expect(artifact_parent == PurePosixPath(owner_dir), errors,
                f"{label}: artifact must be an immediate child of its owner directory")
        if artifact_type == "debug_database":
            _expect(name == "debug_database.bin", errors, f"{label}: wrong debug database filename")
        elif artifact_type == "statistics_info":
            _expect(name in {"neural_statistics_info.txt", "neural_statistics_info.bin"}, errors,
                    f"{label}: wrong statistics-info filename")
        elif artifact_type == "shader_module":
            expected_name = None if pipeline_shader_module_id is None else f"shader_module_{pipeline_shader_module_id}.spv"
            _expect(expected_name is not None and name == expected_name, errors,
                    f"{label}: shader module path is not ID-derived")
        elif artifact_type == "dispatch_statistics":
            expected_name = None if raw_mode is None else f"statistics_mode{raw_mode}.bin"
            _expect(expected_name is not None and name == expected_name, errors,
                    f"{label}: raw statistics filename/type does not match capture mode")
    if raw_mode is None:
        required_counts = static_type_counts or {artifact_type: 1 for artifact_type in STATIC_ARTIFACT_TYPES}
        expected_static_sequence = [
            artifact_type for artifact_type in STATIC_ARTIFACT_ORDER
            if required_counts.get(artifact_type, 1) == 1
        ]
        if allow_missing_static_suffix:
            expected_prefix = expected_static_sequence[:len(static_type_sequence)]
            _expect(static_type_sequence == expected_prefix, errors,
                    f"{owner_dir}: static artifacts must be a published prefix of "
                    f"{expected_static_sequence}, got {static_type_sequence}")
        else:
            for artifact_type in STATIC_ARTIFACT_TYPES:
                expected_count = required_counts.get(artifact_type, 1)
                _expect(type_counts.get(artifact_type, 0) == expected_count, errors,
                        f"{owner_dir}: expected exactly {expected_count} {artifact_type} artifact(s)")
            _expect(static_type_sequence == expected_static_sequence, errors,
                    f"{owner_dir}: static artifacts are not in producer publication order")
        _expect(type_counts.get("dispatch_statistics", 0) == 0, errors,
                f"{owner_dir}: pipeline must not contain dispatch statistics")
    else:
        expected_count = type_counts.get("dispatch_statistics", 0)
        valid_count = expected_count in ({0, 1} if allow_missing_dispatch_statistics else {1})
        _expect(set(type_counts).issubset({"dispatch_statistics"}) and valid_count, errors,
                f"{owner_dir}: dispatch must contain " +
                ("at most one" if allow_missing_dispatch_statistics else "exactly one") +
                " mode-matching raw statistics artifact")
    return type_counts


def validate_capture(root: Path, expected: dict[str, Any], statistics_mode: int | None = None) -> dict[str, Any]:
    errors: list[str] = []
    artifact_classes: set[str] = set()
    artifact_paths: list[tuple[str, str]] = []
    statistics_nonzero: dict[int, bool] = {}
    statistics_bytes: dict[int, bytes] = {}
    documents = 0
    if not root.exists():
        return {"ok": False, "errors": [f"capture root does not exist: {root}"], "artifact_classes": []}
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        return {"ok": False, "errors": [f"cannot inspect capture root: {exc}"], "artifact_classes": []}
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        return {"ok": False, "errors": ["capture root must be an ordinary directory, not a symlink"], "artifact_classes": []}
    root = root.resolve(strict=True)
    actual_files, actual_directories, internal_temporaries = _inventory(root, errors)
    expected_files: set[str] = {"capture.json"}

    capture = _read_document(root, "capture.json", errors, CAPTURE_KEYS, "capture.json")
    if capture is None:
        for relative in sorted(internal_temporaries):
            errors.append(f"internal publication file remains without a valid capture status: {relative}")
        return {"ok": False, "errors": errors, "artifact_classes": [], "documents": documents,
                "files": len(actual_files)}
    documents += 1
    status = capture.get("status")
    _expect(status in ("incomplete", "complete", "error"), errors,
            f"capture.json: invalid status {status!r}")
    allowed_internal_temporaries = internal_temporaries if status in ("incomplete", "error") else set()
    for relative in sorted(internal_temporaries - allowed_internal_temporaries):
        context = "a complete capture" if status == "complete" else "a capture without a valid terminal status"
        errors.append(f"internal publication file remains in {context}: {relative}")
    error_value = capture.get("error")
    if status == "error":
        _expect(isinstance(error_value, str) and bool(error_value), errors,
                "capture.json: error status requires a nonempty error string")
    else:
        _expect(error_value is None, errors, "capture.json: non-error status requires null error")
    expected_status = expected.get("capture_status")
    if expected_status is not None:
        _expect(status == expected_status, errors, f"capture status: expected {expected_status}, got {status}")
    error_contains = expected.get("error_contains")
    if error_contains:
        _expect(isinstance(error_value, str) and error_contains in error_value, errors,
                f"capture error does not contain {error_contains!r}: {error_value!r}")

    settings = capture.get("capture")
    if not isinstance(settings, dict):
        errors.append("capture.json.capture: expected object")
        settings = {}
    else:
        _require_keys(settings, CAPTURE_SETTINGS_KEYS, errors, "capture.json.capture")
    _expect(isinstance(settings.get("layer_name"), str) and bool(settings.get("layer_name")), errors,
            "capture.json.capture.layer_name: expected nonempty string")
    for field in ("layer_version", "commit_identity", "dispatch_filter"):
        _expect(isinstance(settings.get(field), str), errors, f"capture.json.capture.{field}: expected string")
    _expect(_is_int(settings.get("layer_implementation_version")) and settings.get("layer_implementation_version") >= 0,
            errors, "capture.json.capture.layer_implementation_version: expected nonnegative integer")
    capture_mode = settings.get("statistics_mode")
    _expect(capture_mode in {0, 1} and not isinstance(capture_mode, bool), errors,
            "capture.json.capture.statistics_mode: expected 0 or 1")
    if statistics_mode is not None:
        _expect(capture_mode == statistics_mode, errors,
                f"capture statistics mode: expected {statistics_mode}, got {capture_mode}")

    devices = capture.get("devices")
    device_ids: set[int] = set()
    if not isinstance(devices, list):
        errors.append("capture.json.devices: expected array")
        devices = []
    for index, device in enumerate(devices):
        label = f"capture.json.devices[{index}]"
        if not isinstance(device, dict):
            errors.append(f"{label}: expected object")
            continue
        _require_keys(device, DEVICE_KEYS, errors, label)
        logical_id = device.get("id")
        _expect(_is_int(logical_id) and logical_id >= 0, errors,
                f"{label}.id: expected nonnegative integer")
        if _is_int(logical_id) and logical_id >= 0:
            _expect(logical_id not in device_ids, errors, f"{label}: duplicate logical device ID {logical_id}")
            device_ids.add(logical_id)
        _expect(isinstance(device.get("name"), str) and bool(device.get("name")), errors,
                f"{label}.name: expected nonempty string")
        for field in DEVICE_KEYS - {"id", "name"}:
            _expect(device.get(field) is None or (_is_int(device.get(field)) and device.get(field) >= 0), errors,
                    f"{label}.{field}: expected null or nonnegative integer")
    expected_device_count = expected.get("device_count")
    if expected_device_count is not None:
        _expect(len(device_ids) == expected_device_count, errors,
                f"logical device count: expected {expected_device_count}, got {len(device_ids)}")
    warnings = capture.get("warnings")
    warnings_valid = isinstance(warnings, list) and all(isinstance(item, str) for item in warnings)
    _expect(warnings_valid, errors, "capture.json.warnings: expected array of strings")
    warning_values = warnings if warnings_valid else []

    pipeline_ids: set[int] = set()
    session_ids: set[int] = set()
    dispatch_ids: set[int] = set()
    actual_pipelines: list[dict[str, Any]] = []
    for pipeline_id, pipeline_relative in _validate_reference_list(capture, "pipelines", errors, "capture.json"):
        expected_pipeline_relative = f"pipeline_{pipeline_id:06d}/pipeline.json"
        _expect(pipeline_relative == expected_pipeline_relative, errors,
                f"pipeline reference {pipeline_id}: expected {expected_pipeline_relative}, got {pipeline_relative}")
        if pipeline_id in pipeline_ids:
            errors.append(f"duplicate global pipeline ID {pipeline_id}")
        pipeline_ids.add(pipeline_id)
        expected_files.add(pipeline_relative)
        pipeline = _read_document(root, pipeline_relative, errors, PIPELINE_KEYS, pipeline_relative)
        if pipeline is None:
            continue
        documents += 1
        _expect(pipeline.get("id") == pipeline_id, errors, f"{pipeline_relative}: ID/reference mismatch")
        pipeline_device_id = pipeline.get("device_id")
        _expect(_is_int(pipeline_device_id) and pipeline_device_id in device_ids, errors,
                f"{pipeline_relative}: device_id does not reference a captured logical device")
        _expect(isinstance(pipeline.get("friendly_name"), str), errors, f"{pipeline_relative}: malformed friendly_name")
        _validate_handle(pipeline.get("diagnostic_handle"), errors, f"{pipeline_relative}.diagnostic_handle")
        _validate_handle(pipeline.get("pipeline_layout_handle"), errors, f"{pipeline_relative}.pipeline_layout_handle")
        _expect(_is_int(pipeline.get("flags")) and pipeline.get("flags") >= 0, errors, f"{pipeline_relative}: malformed flags")
        _expect(pipeline.get("vendor_options") is None or isinstance(pipeline.get("vendor_options"), str), errors,
                f"{pipeline_relative}: malformed vendor_options")
        for field in ("identifier_only", "foreign_processing_engine", "statistics_enabled"):
            _expect(isinstance(pipeline.get(field), bool), errors, f"{pipeline_relative}: malformed {field}")
        bindings = pipeline.get("resource_bindings")
        if not isinstance(bindings, list):
            errors.append(f"{pipeline_relative}.resource_bindings: expected array")
        else:
            seen_bindings: set[tuple[int, int, int]] = set()
            for index, binding in enumerate(bindings):
                label = f"{pipeline_relative}.resource_bindings[{index}]"
                if not isinstance(binding, dict):
                    errors.append(f"{label}: expected object")
                    continue
                _require_keys(binding, {"descriptor_set", "binding", "array_element"}, errors, label)
                values = tuple(binding.get(key) for key in ("descriptor_set", "binding", "array_element"))
                _expect(all(_is_int(value) and value >= 0 for value in values), errors, f"{label}: malformed binding")
                if values in seen_bindings:
                    errors.append(f"{label}: duplicate binding descriptor")
                seen_bindings.add(values)
        shader = pipeline.get("shader")
        module_id: int | None = None
        if not isinstance(shader, dict):
            errors.append(f"{pipeline_relative}.shader: expected object")
            shader = {}
        else:
            _require_keys(shader, SHADER_KEYS, errors, f"{pipeline_relative}.shader")
        if shader.get("module_id") is not None:
            _expect(_is_int(shader.get("module_id")) and shader.get("module_id") >= 0, errors,
                    f"{pipeline_relative}.shader.module_id: malformed")
            if _is_int(shader.get("module_id")) and shader.get("module_id") >= 0:
                module_id = shader["module_id"]
        spirv_available = shader.get("spirv_available")
        _expect(isinstance(spirv_available, bool), errors,
                f"{pipeline_relative}.shader.spirv_available: malformed")
        if spirv_available is True:
            _expect(module_id is not None, errors,
                    f"{pipeline_relative}.shader: available SPIR-V requires a module_id")
        for field in ("friendly_name", "entry_point"):
            _expect(isinstance(shader.get(field), str), errors, f"{pipeline_relative}.shader.{field}: malformed")
        _expect(_is_int(shader.get("specialization_data_size")) and shader.get("specialization_data_size") >= 0,
                errors, f"{pipeline_relative}.shader.specialization_data_size: malformed")
        entries = shader.get("specialization_entries")
        if not isinstance(entries, list):
            errors.append(f"{pipeline_relative}.shader.specialization_entries: expected array")
        else:
            seen_entries: set[tuple[int, int, int]] = set()
            for index, entry in enumerate(entries):
                label = f"{pipeline_relative}.shader.specialization_entries[{index}]"
                if not isinstance(entry, dict):
                    errors.append(f"{label}: expected object")
                    continue
                _require_keys(entry, {"constant_id", "offset", "size"}, errors, label)
                values = tuple(entry.get(key) for key in ("constant_id", "offset", "size"))
                _expect(all(_is_int(value) and value >= 0 for value in values), errors, f"{label}: malformed entry")
                if values in seen_entries:
                    errors.append(f"{label}: duplicate specialization entry")
                seen_entries.add(values)
        pipeline_dir = str(PurePosixPath(pipeline_relative).parent)
        debug_warning = f"debug database was unavailable for pipeline {pipeline_id}"
        statistics_info_warning = f"neural statistics info was unavailable for pipeline {pipeline_id}"
        for warning in (debug_warning, statistics_info_warning):
            _expect(warning_values.count(warning) <= 1, errors,
                    f"capture.json.warnings: duplicate omission warning {warning!r}")
        static_type_counts = {
            "debug_database": 0 if debug_warning in warning_values else 1,
            "statistics_info": 0 if statistics_info_warning in warning_values else 1,
            "shader_module": 1 if spirv_available is True else 0,
        }
        _validate_artifacts(root, pipeline_dir, pipeline, errors, expected_files, artifact_classes, artifact_paths,
                            raw_mode=None, pipeline_shader_module_id=module_id,
                            static_type_counts=static_type_counts,
                            allow_missing_static_suffix=status == "error")

        actual_sessions: list[dict[str, Any]] = []
        for session_id, session_relative in _validate_reference_list(pipeline, "sessions", errors, pipeline_relative):
            expected_session_relative = f"{pipeline_dir}/session_{session_id:06d}/session.json"
            _expect(session_relative == expected_session_relative, errors,
                    f"session reference {session_id}: expected {expected_session_relative}, got {session_relative}")
            if session_id in session_ids:
                errors.append(f"duplicate global session ID {session_id}")
            session_ids.add(session_id)
            expected_files.add(session_relative)
            session = _read_document(root, session_relative, errors, SESSION_KEYS, session_relative)
            if session is None:
                continue
            documents += 1
            _expect(session.get("id") == session_id, errors, f"{session_relative}: ID/reference mismatch")
            _expect(session.get("pipeline_id") == pipeline_id, errors, f"{session_relative}: pipeline_id mismatch")
            _expect(session.get("device_id") == pipeline_device_id, errors,
                    f"{session_relative}: device_id does not match its pipeline")
            _expect(isinstance(session.get("friendly_name"), str), errors, f"{session_relative}: malformed friendly_name")
            _validate_handle(session.get("diagnostic_handle"), errors, f"{session_relative}.diagnostic_handle")
            _expect(_is_int(session.get("flags")) and session.get("flags") >= 0, errors,
                    f"{session_relative}: malformed flags")
            session_dir = str(PurePosixPath(session_relative).parent)
            actual_dispatches: list[dict[str, int]] = []
            for dispatch_id, dispatch_relative in _validate_reference_list(session, "dispatches", errors, session_relative):
                expected_dispatch_relative = f"{session_dir}/dispatch_{dispatch_id:06d}/dispatch.json"
                _expect(dispatch_relative == expected_dispatch_relative, errors,
                        f"dispatch reference {dispatch_id}: expected {expected_dispatch_relative}, got {dispatch_relative}")
                if dispatch_id in dispatch_ids:
                    errors.append(f"duplicate global dispatch ID {dispatch_id}")
                dispatch_ids.add(dispatch_id)
                expected_files.add(dispatch_relative)
                dispatch = _read_document(root, dispatch_relative, errors, DISPATCH_KEYS, dispatch_relative)
                if dispatch is None:
                    continue
                documents += 1
                _expect(dispatch.get("id") == dispatch_id, errors, f"{dispatch_relative}: ID/reference mismatch")
                _expect(dispatch.get("session_id") == session_id, errors, f"{dispatch_relative}: session_id mismatch")
                executed_index = dispatch.get("executed_index")
                _expect(_is_int(executed_index) and executed_index >= 0, errors,
                        f"{dispatch_relative}: executed_index must be nonnegative integer")
                _validate_handle(dispatch.get("diagnostic_command_buffer_handle"), errors,
                                 f"{dispatch_relative}.diagnostic_command_buffer_handle")
                dispatch_dir = str(PurePosixPath(dispatch_relative).parent)
                dispatch_type_counts = _validate_artifacts(
                    root, dispatch_dir, dispatch, errors, expected_files, artifact_classes,
                    artifact_paths, raw_mode=capture_mode if capture_mode in {0, 1} else None,
                    allow_missing_dispatch_statistics=status == "error")
                if capture_mode in {0, 1} and dispatch_type_counts.get("dispatch_statistics") == 1:
                    raw_relative = f"{dispatch_dir}/statistics_mode{capture_mode}.bin"
                    raw_artifact = _resolved_regular(
                        root, raw_relative, errors, f"dispatch {dispatch_id} statistics content")
                    if raw_artifact is not None:
                        try:
                            raw_bytes = raw_artifact.read_bytes()
                            statistics_bytes[dispatch_id] = raw_bytes
                            statistics_nonzero[dispatch_id] = any(raw_bytes)
                        except OSError as exc:
                            errors.append(f"cannot inspect dispatch statistics content {raw_relative}: {exc}")
                actual_dispatches.append({"id": dispatch_id, "executed_index": executed_index})
            actual_sessions.append({"id": session_id, "dispatches": actual_dispatches})
        actual_pipelines.append({"id": pipeline_id, "sessions": actual_sessions})

    for warning in warning_values:
        unavailable = re.fullmatch(
            r"(?:debug database|neural statistics info) was unavailable for pipeline ([0-9]+)", warning)
        if unavailable is not None:
            warned_pipeline_id = int(unavailable.group(1))
            _expect(warned_pipeline_id in pipeline_ids, errors,
                    f"capture.json.warnings: omission warning references unknown pipeline {warned_pipeline_id}")

    expected_pipelines = expected.get("pipelines")
    if expected_pipelines is not None:
        _expect(actual_pipelines == expected_pipelines, errors,
                f"hierarchy mismatch: expected {expected_pipelines!r}, got {actual_pipelines!r}")
    required_classes = set(expected.get("artifact_classes", []))
    _expect(required_classes.issubset(artifact_classes), errors,
            f"missing artifact classes: {sorted(required_classes - artifact_classes)}")
    seen_content_expectations: set[int] = set()
    for index, content_expectation in enumerate(expected.get("statistics_contents", [])):
        label = f"statistics_contents[{index}]"
        if not isinstance(content_expectation, dict) or set(content_expectation) != {"dispatch_id", "content"}:
            errors.append(f"{label}: expected dispatch_id/content object")
            continue
        dispatch_id = content_expectation.get("dispatch_id")
        content = content_expectation.get("content")
        if not _is_int(dispatch_id) or dispatch_id < 0 or content not in {"zero", "nonzero"}:
            errors.append(f"{label}: dispatch_id must be nonnegative and content must be zero or nonzero")
            continue
        if dispatch_id in seen_content_expectations:
            errors.append(f"{label}: duplicate dispatch content expectation")
            continue
        seen_content_expectations.add(dispatch_id)
        actual_nonzero = statistics_nonzero.get(dispatch_id)
        _expect(actual_nonzero is not None, errors, f"{label}: dispatch {dispatch_id} statistics were not found")
        if actual_nonzero is not None:
            expected_nonzero = content == "nonzero"
            _expect(actual_nonzero == expected_nonzero, errors,
                    f"dispatch {dispatch_id} statistics content: expected {content}, got "
                    f"{'nonzero' if actual_nonzero else 'zero'}")

    seen_statistics_pairs: set[tuple[int, int]] = set()
    for index, raw_pair in enumerate(expected.get("statistics_differ", [])):
        label = f"statistics_differ[{index}]"
        if (not isinstance(raw_pair, list) or len(raw_pair) != 2 or
                not all(_is_int(dispatch_id) and dispatch_id >= 0 for dispatch_id in raw_pair) or
                raw_pair[0] == raw_pair[1]):
            errors.append(f"{label}: expected two distinct nonnegative dispatch IDs")
            continue
        first_id, second_id = raw_pair
        pair = tuple(sorted((first_id, second_id)))
        if pair in seen_statistics_pairs:
            errors.append(f"{label}: duplicate dispatch comparison {pair}")
            continue
        seen_statistics_pairs.add(pair)
        first = statistics_bytes.get(first_id)
        second = statistics_bytes.get(second_id)
        _expect(first is not None, errors, f"{label}: dispatch {first_id} statistics were not found")
        _expect(second is not None, errors, f"{label}: dispatch {second_id} statistics were not found")
        if first is not None and second is not None:
            _expect(first != second, errors,
                    f"dispatch statistics must differ: {first_id} and {second_id} are byte-identical")

    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    public_files = actual_files - allowed_internal_temporaries
    _expect(public_files == expected_files, errors,
            f"public file set mismatch: unexpected={sorted(public_files - expected_files)}, missing={sorted(expected_files - public_files)}")
    _expect(actual_directories == expected_directories, errors,
            f"public directory set mismatch: unexpected={sorted(actual_directories - expected_directories)}, missing={sorted(expected_directories - actual_directories)}")
    if expected.get("pipelines") == []:
        _expect(not any(path.startswith("pipeline_") for path in actual_files | actual_directories), errors,
                "lazy hierarchy miss must leave no pipeline/session/dispatch output")

    spirv_comparison: dict[str, Any] | None = None
    spirv = expected.get("spirv_file")
    if spirv:
        reference = Path(spirv)
        try:
            mode = reference.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise OSError("reference is not an ordinary regular file")
            reference_hash = sha256(reference)
            reference_bytes = reference.read_bytes()
            shader_paths = [relative for artifact_type, relative in artifact_paths if artifact_type == "shader_module"]
            _expect(len(shader_paths) == 1, errors,
                    f"SPIR-V comparison requires exactly one captured shader module, got {len(shader_paths)}")
            captures = []
            for relative in shader_paths:
                capture_path = root / relative
                capture_hash = sha256(capture_path)
                matches = capture_path.read_bytes() == reference_bytes
                _expect(matches, errors, f"SPIR-V bytes differ: {relative}")
                captures.append({"path": relative, "sha256": capture_hash, "matches": matches})
            spirv_comparison = {
                "reference": str(reference), "reference_sha256": reference_hash,
                "reference_size": len(reference_bytes), "captures": captures,
            }
        except OSError as exc:
            errors.append(f"cannot read SPIR-V reference {reference}: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": status,
        "pipelines": actual_pipelines,
        "artifact_classes": sorted(artifact_classes),
        "documents": documents,
        "files": len(actual_files),
        "spirv_comparison": spirv_comparison,
    }


def _matching_events(events: list[Any], requirement: dict[str, Any]) -> list[dict[str, Any]]:
    name = requirement.get("event")
    where = requirement.get("where", {})
    return [event for event in events if isinstance(event, dict) and event.get("event") == name
            and all(event.get(key) == value for key, value in where.items())]


def validate_events(events: list[Any], expected_events: list[Any], expected_order: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    assertions: list[dict[str, Any]] = []
    for raw in expected_events:
        requirement = {"event": raw} if isinstance(raw, str) else raw
        if not isinstance(requirement, dict) or not isinstance(requirement.get("event"), str):
            errors.append(f"invalid expected event requirement: {raw!r}")
            continue
        matches = _matching_events(events, requirement)
        minimum = int(requirement.get("min_count", requirement.get("count", 1)))
        maximum = int(requirement.get("max_count", requirement.get("count", minimum)))
        ok = minimum <= len(matches) <= maximum
        if not ok:
            errors.append(f"event {requirement!r}: expected count {minimum}..{maximum}, got {len(matches)}")
        assertions.append({"requirement": requirement, "observed": len(matches), "ok": ok})
    cursor = -1
    for raw in expected_order:
        requirement = {"event": raw} if isinstance(raw, str) else raw
        found = None
        if isinstance(requirement, dict):
            for index in range(cursor + 1, len(events)):
                if _matching_events([events[index]], requirement):
                    found = index
                    break
        if found is None:
            errors.append(f"ordered event not found after index {cursor}: {requirement!r}")
            assertions.append({"order": requirement, "ok": False})
        else:
            cursor = found
            assertions.append({"order": requirement, "index": found, "ok": True})
    return errors, assertions


def write_reports(summary: dict[str, Any], summary_path: Path, report_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["Neural Statistics E2E Report", "=" * 29, ""]
    lines.append(f"Overall: {'PASS' if summary['ok'] else 'FAIL'}")
    lines.append(f"Matrix: {summary['matrix']}")
    lines.append("")
    for result in summary["cases"]:
        lines.append(f"[{result['result'].upper()}] {result['name']}")
        lines.append(f"  exit={result.get('exit_code')} timeout={result.get('timed_out')} elapsed_ms={result.get('elapsed_ms')}")
        lines.append(f"  run={result.get('run_dir')}")
        for assertion in result.get("event_assertions", []):
            lines.append(f"  behavioral_assertion: {json.dumps(assertion, sort_keys=True)}")
        comparison = (result.get("validation") or {}).get("spirv_comparison")
        if comparison:
            lines.append(f"  spirv_comparison: {json.dumps(comparison, sort_keys=True)}")
        for error in result.get("errors", []):
            lines.append(f"  error: {error}")
        for event in result.get("events", []):
            lines.append(f"  event: {json.dumps(event, sort_keys=True)}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_work_root(work_root: Path) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    mode = work_root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise E2EError("work root must be an ordinary directory, not a symlink")
    return work_root.resolve(strict=True)


def _create_run_dir(work_root: Path, name: str) -> Path:
    validate_case_name(name)
    root = _safe_work_root(work_root)
    candidate = root / f"{name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    if os.path.commonpath([str(root), str(candidate)]) != str(root):
        raise E2EError(f"run directory escapes work root: {candidate}")
    candidate.mkdir(parents=False, exist_ok=False)
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root or os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise E2EError(f"resolved run directory escapes work root: {resolved}")
    return resolved


def _process_group_alive(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, shell=False, timeout=10)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_case(case: dict[str, Any], fixture: Path, work_root: Path, variables: dict[str, str], default_timeout: float) -> dict[str, Any]:
    case = substitute(case, variables)
    name = case["name"]
    validate_case_name(name)
    run_dir = _create_run_dir(work_root, name)
    capture_root = run_dir / "capture"
    if case.get("precreate_capture_root"):
        capture_root.mkdir()
    replacements = {"CAPTURE_ROOT": str(capture_root), "RUN_DIR": str(run_dir)}
    case = runtime_substitute(case, replacements)
    fixture_args = [str(arg) for arg in case.get("fixture", {}).get("args", [])]
    command = [str(fixture)] + fixture_args

    env = os.environ.copy()
    layer = case.get("layer", {})
    env["VK_INSTANCE_LAYERS"] = str(layer.get("instance_layers", "VK_LAYER_LGL_neural_statistics"))
    env["VK_LAYER_CAPTURE_FOLDER"] = str(capture_root)
    env["VK_LAYER_STATISTICS_MODE"] = str(layer.get("statistics_mode", 1))
    env["VK_LAYER_DISPATCH_FILTER"] = str(layer.get("dispatch_filter", ""))
    env.pop("VK_LAYER_TEST_POST_GRAPH_COPY_COUNT", None)
    test_copy_count = layer.get("test_post_graph_copy_count")
    if test_copy_count is not None:
        if not _is_int(test_copy_count) or not 1 <= test_copy_count <= 65536:
            raise E2EError(f"case {name}: test_post_graph_copy_count must be an integer from 1 to 65536")
        env["VK_LAYER_TEST_POST_GRAPH_COPY_COUNT"] = str(test_copy_count)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    timeout = float(case.get("timeout_seconds", default_timeout))
    if timeout <= 0:
        raise E2EError("case timeout must be positive")
    started_ns = time.time_ns()
    timed_out = False
    exit_code: int | None = None
    cleanup_ok = True
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, shell=False, start_new_session=True)
        pid = process.pid
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(process)
            exit_code = process.wait(timeout=10)
        if os.name != "nt":
            deadline = time.monotonic() + 2.0
            while _process_group_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_group_alive(pid):
                cleanup_ok = False
                _kill_process_tree(process)
    ended_ns = time.time_ns()

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    combined = stdout_text + "\n" + stderr_text
    errors: list[str] = []
    expected_exit = int(case.get("expected_exit", 0))
    if timed_out:
        errors.append(f"case exceeded {timeout} seconds")
    if exit_code != expected_exit:
        errors.append(f"exit code: expected {expected_exit}, got {exit_code}")
    if not cleanup_ok:
        errors.append(f"fixture process group {pid} leaked child processes")
    for stream, text in (("stdout", stdout_text), ("stderr", stderr_text)):
        key = f"expected_{stream}_contains"
        diagnostics = case.get(key, [])
        if isinstance(diagnostics, str):
            diagnostics = [diagnostics]
        if not isinstance(diagnostics, list) or not all(isinstance(item, str) and item for item in diagnostics):
            raise E2EError(f"case {name}: {key} must be a nonempty string or an array of nonempty strings")
        for expected in diagnostics:
            if expected not in text:
                errors.append(f"{stream} does not contain required diagnostic: {expected!r}")
    for pattern in BAD_LOG_PATTERNS:
        if pattern.search(combined):
            errors.append(f"unexpected fatal log pattern: {pattern.pattern}")

    events: list[Any] = []
    for line in combined.splitlines():
        if line.startswith("E2E_EVENT "):
            try:
                events.append(json.loads(line[len("E2E_EVENT "):]))
            except json.JSONDecodeError:
                errors.append(f"malformed E2E_EVENT line: {line}")
    event_errors, event_assertions = validate_events(
        events, case.get("expected_events", []), case.get("expected_event_order", []))
    errors.extend(event_errors)

    validation: dict[str, Any] | None = None
    expectation = case.get("expected_capture")
    if expectation is not None:
        if expectation.get("absent"):
            if capture_root.exists() or capture_root.is_symlink():
                errors.append("capture root exists but was expected to be completely absent")
        else:
            validation = validate_capture(capture_root, expectation, int(layer.get("statistics_mode", 1)))
            errors.extend(validation["errors"])
    result = {
        "name": name,
        "result": "pass" if not errors else "fail",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "process_group_cleanup_ok": cleanup_ok,
        "elapsed_ms": round((ended_ns - started_ns) / 1_000_000, 3),
        "start_ns": started_ns,
        "end_ns": ended_ns,
        "command": command,
        "run_dir": str(run_dir),
        "capture_root": str(capture_root),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "events": events,
        "event_assertions": event_assertions,
        "validation": validation,
        "errors": errors,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    try:
        matrix = parse_matrix(args.matrix)
        selected = select_cases(matrix, args.case, args.tag)
        if args.list:
            for case in selected:
                print(case["name"])
            return 0
        if args.fixture is None:
            raise E2EError("--fixture is required when running cases")
        variables: dict[str, str] = {}
        for item in args.var:
            if "=" not in item:
                raise E2EError(f"invalid --var {item!r}; expected NAME=VALUE")
            key, value = item.split("=", 1)
            if not key:
                raise E2EError("--var name must not be empty")
            variables[key] = value
        work_root = _safe_work_root(args.work_root)
        results = [run_case(case, args.fixture, work_root, variables, args.timeout_seconds) for case in selected]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "matrix": str(args.matrix),
            "generated_ns": time.time_ns(),
            "ok": all(result["result"] == "pass" for result in results),
            "cases": results,
        }
        summary_path = args.summary_json or (work_root / "summary.json")
        report_path = args.report or (work_root / "report.txt")
        write_reports(summary, summary_path, report_path)
        print(json.dumps({"ok": summary["ok"], "summary": str(summary_path), "report": str(report_path)}))
        return 0 if summary["ok"] else 1
    except E2EError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
