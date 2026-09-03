# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import neural_statistics_e2e as e2e


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_capture(root: Path, mode: int = 1, executed_index: int = 0) -> dict[str, Path]:
    pipeline_dir = root / "pipeline_000000"
    session_dir = pipeline_dir / "session_000000"
    dispatch_dir = session_dir / "dispatch_000000"
    dispatch_dir.mkdir(parents=True)
    debug = pipeline_dir / "debug_database.bin"
    info = pipeline_dir / "neural_statistics_info.txt"
    spirv = pipeline_dir / "shader_module_0.spv"
    raw = dispatch_dir / f"statistics_mode{mode}.bin"
    debug.write_bytes(b"debug")
    info.write_text("statistics", encoding="utf-8")
    spirv.write_bytes(b"\x03\x02#\x07fixture-spirv")
    raw.write_bytes(b"stats")
    write_json(root / "capture.json", {
        "schema_version": 2,
        "status": "complete",
        "error": None,
        "capture": {
            "layer_name": "VK_LAYER_LGL_neural_statistics",
            "layer_version": "1.0.0",
            "commit_identity": "test",
            "layer_implementation_version": 1,
            "statistics_mode": mode,
            "dispatch_filter": "",
        },
        "devices": [{"id": 0, "name": "test-device", "vendor_id": 1, "device_id": 2,
                     "driver_version": 3, "api_version": 4}],
        "warnings": [],
        "pipelines": [{"id": 0, "path": "pipeline_000000/pipeline.json"}],
    })
    write_json(pipeline_dir / "pipeline.json", {
        "schema_version": 2,
        "id": 0,
        "device_id": 0,
        "friendly_name": "pipeline",
        "diagnostic_handle": 10,
        "flags": 0,
        "pipeline_layout_handle": 11,
        "resource_bindings": [{"descriptor_set": 0, "binding": 0, "array_element": 0}],
        "vendor_options": None,
        "identifier_only": False,
        "foreign_processing_engine": False,
        "statistics_enabled": True,
        "shader": {
            "module_id": 0,
            "spirv_available": True,
            "friendly_name": "shader",
            "entry_point": "main",
            "specialization_entries": [],
            "specialization_data_size": 0,
        },
        "artifacts": [
            {"path": "pipeline_000000/debug_database.bin", "type": "debug_database", "size": debug.stat().st_size},
            {"path": "pipeline_000000/neural_statistics_info.txt", "type": "statistics_info", "size": info.stat().st_size},
            {"path": "pipeline_000000/shader_module_0.spv", "type": "shader_module", "size": spirv.stat().st_size},
        ],
        "sessions": [{"id": 0, "path": "pipeline_000000/session_000000/session.json"}],
    })
    write_json(session_dir / "session.json", {
        "schema_version": 2,
        "id": 0,
        "pipeline_id": 0,
        "device_id": 0,
        "friendly_name": "session",
        "diagnostic_handle": 12,
        "flags": 0,
        "dispatches": [{"id": 0, "path": "pipeline_000000/session_000000/dispatch_000000/dispatch.json"}],
    })
    write_json(dispatch_dir / "dispatch.json", {
        "schema_version": 2,
        "id": 0,
        "session_id": 0,
        "executed_index": executed_index,
        "diagnostic_command_buffer_handle": 13,
        "artifacts": [{
            "path": f"pipeline_000000/session_000000/dispatch_000000/statistics_mode{mode}.bin",
            "type": "dispatch_statistics",
            "size": raw.stat().st_size,
        }],
    })
    return {
        "capture": root / "capture.json",
        "pipeline": pipeline_dir / "pipeline.json",
        "session": session_dir / "session.json",
        "dispatch": dispatch_dir / "dispatch.json",
        "spirv": spirv,
        "raw": raw,
    }


def valid_expected(index: int = 0):
    return {
        "capture_status": "complete",
        "artifact_classes": ["debug_database", "statistics_info", "shader_module", "dispatch_statistics"],
        "pipelines": [{"id": 0, "sessions": [{"id": 0, "dispatches": [{"id": 0, "executed_index": index}]}]}],
    }


class MatrixTests(unittest.TestCase):
    def test_parsing_selection_and_safe_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            write_json(path, {"schema_version": 1, "cases": [
                {"name": "one", "tags": ["a"]}, {"name": "two-case", "tags": ["b"]}
            ]})
            matrix = e2e.parse_matrix(path)
            self.assertEqual(["two-case"], [case["name"] for case in e2e.select_cases(matrix, ["two-case"], [])])
            self.assertEqual(["one"], [case["name"] for case in e2e.select_cases(matrix, [], ["a"])])
            with self.assertRaises(e2e.E2EError):
                e2e.select_cases(matrix, ["missing"], [])
            self.assertEqual("x/{RUN_DIR}/y", e2e.substitute("x/{RUN_DIR}/{VALUE}", {"VALUE": "y"}))
            with self.assertRaises(e2e.E2EError):
                e2e.substitute("{MISSING}", {})

    def test_multiple_logical_device_case_uses_fixture_and_requires_two_devices(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        case = next(item for item in matrix["cases"] if item["name"] == "multiple-logical-devices")
        arguments = case["fixture"]["args"]
        self.assertEqual("2", arguments[arguments.index("--device-count") + 1])
        self.assertEqual(2, case["expected_capture"]["device_count"])
        self.assertIn({"event": "second_logical_device_used", "count": 1}, case["expected_events"])

    def test_immediate_resubmission_regression_case_requires_four_successful_dispatches(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        case = next(item for item in matrix["cases"] if item["name"] == "immediate-primary-resubmission-no-retry")
        arguments = case["fixture"]["args"]
        self.assertIn("--immediate-resubmission", arguments)
        repeats = arguments.index("--dispatch-repeats")
        self.assertEqual("4", arguments[repeats + 1])
        self.assertNotIn("--resubmissions", arguments)
        self.assertEqual(0, case["expected_exit"])
        self.assertIn({"event": "immediate_resubmission_submit", "count": 4}, case["expected_events"])
        dispatches = case["expected_capture"]["pipelines"][0]["sessions"][0]["dispatches"]
        self.assertEqual(
            [{"id": index, "executed_index": index} for index in range(4)],
            dispatches,
        )

    def test_distinct_one_time_primaries_same_session_expect_two_captures(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        case = next(item for item in matrix["cases"] if item["name"] == "distinct-one-time-primaries-same-session")
        arguments = case["fixture"]["args"]
        submissions = arguments.index("--distinct-primary-submissions")
        self.assertEqual("2", arguments[submissions + 1])
        self.assertEqual(0, case["expected_exit"])
        dispatches = case["expected_capture"]["pipelines"][0]["sessions"][0]["dispatches"]
        self.assertEqual(
            [{"id": 0, "executed_index": 0}, {"id": 1, "executed_index": 1}],
            dispatches,
        )

    def test_same_session_data_graph_stage_semaphore_uses_two_queues_without_intermediate_host_wait(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        case = next(item for item in matrix["cases"] if item["name"] == "same-session-data-graph-stage-semaphore")
        arguments = case["fixture"]["args"]
        self.assertIn("--distinct-primary-stage-semaphore", arguments)
        self.assertEqual("core", arguments[arguments.index("--submit-route") + 1])
        self.assertEqual("2", arguments[arguments.index("--queue-count") + 1])
        self.assertEqual("2", arguments[arguments.index("--distinct-primary-submissions") + 1])
        self.assertEqual(255, case["expected_exit"])
        self.assertIn("vkQueueSubmit2 was not exposed", case["expected_stderr_contains"])

        khr_case = next(item for item in matrix["cases"] if item["name"] == "same-session-data-graph-stage-semaphore-khr")
        khr_arguments = khr_case["fixture"]["args"]
        self.assertEqual("khr", khr_arguments[khr_arguments.index("--submit-route") + 1])
        events = khr_case["expected_events"]
        self.assertIn("--second-dispatch-conditional-false", khr_arguments)
        self.assertEqual(256, khr_case["layer"]["test_post_graph_copy_count"])
        self.assertIn({"event": "conditional_false_dispatch_recorded", "count": 1, "where": {"submission": 1}}, events)
        self.assertIn({"event": "data_graph_stage_semaphore_signal", "count": 1, "where": {"submission": 0}}, events)
        self.assertIn({"event": "data_graph_stage_semaphore_wait", "count": 1, "where": {"submission": 1}}, events)
        self.assertEqual(
            [{"dispatch_id": 0, "content": "nonzero"}],
            khr_case["expected_capture"]["statistics_contents"],
        )
        self.assertEqual([[0, 1]], khr_case["expected_capture"]["statistics_differ"])
        self.assertEqual(0, khr_case["expected_exit"])

    def test_same_submit_duplicate_destination_cases_capture_each_occurrence(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        cases = {item["name"]: item for item in matrix["cases"]}

        primary = cases["same-primary-twice-in-one-submit"]
        self.assertIn("--primary-command-buffers-per-submit", primary["fixture"]["args"])
        self.assertEqual(0, primary["expected_exit"])
        primary_dispatches = primary["expected_capture"]["pipelines"][0]["sessions"][0]["dispatches"]
        self.assertEqual([0, 1], [item["executed_index"] for item in primary_dispatches])
        self.assertIn(
            {"event": "primary_command_buffer_batch", "count": 1, "where": {"result": 2}},
            primary["expected_events"],
        )
        self.assertIn(
            {"event": "primary_command_buffer_batch_marshaled", "count": 1, "where": {"result": 2}},
            primary["expected_events"],
        )

        secondary = cases["same-secondary-twice-in-primary"]
        self.assertIn("--secondary-executions", secondary["fixture"]["args"])
        self.assertEqual(0, secondary["expected_exit"])
        secondary_dispatches = secondary["expected_capture"]["pipelines"][0]["sessions"][0]["dispatches"]
        self.assertEqual([0, 1], [item["executed_index"] for item in secondary_dispatches])

    def test_deterministic_matrix_event_counts_are_exact_and_enforced(self):
        matrix_path = Path(__file__).parents[1] / "matrix" / "target.json"
        matrix = e2e.parse_matrix(matrix_path)
        cases = {item["name"]: item for item in matrix["cases"]}

        primary = cases["same-primary-twice-in-one-submit"]
        primary_events = [
            {"event": "primary_command_buffer_batch", "result": 2},
            {"event": "primary_command_buffer_batch_marshaled", "result": 2},
        ]
        errors, assertions = e2e.validate_events(primary_events, primary["expected_events"], [])
        self.assertEqual([], errors)
        self.assertTrue(all(assertion["observed"] == 1 and assertion["ok"] for assertion in assertions))
        errors, _ = e2e.validate_events(primary_events + [primary_events[0]], primary["expected_events"], [])
        self.assertTrue(any("got 2" in error for error in errors))

        immediate = cases["immediate-primary-resubmission-no-retry"]
        immediate_events = [{"event": "immediate_resubmission_submit"} for _ in range(4)]
        immediate_events.extend({"event": "application_fence_complete"} for _ in range(4))
        errors, assertions = e2e.validate_events(immediate_events, immediate["expected_events"], [])
        self.assertEqual([], errors)
        self.assertEqual([4, 4], [assertion["observed"] for assertion in assertions])
        errors, _ = e2e.validate_events(
            immediate_events + [{"event": "immediate_resubmission_submit"}],
            immediate["expected_events"],
            [],
        )
        self.assertTrue(any("got 5" in error for error in errors))

        multiple_queues = cases["multiple-queues"]
        queue_events = [
            {"event": "queue_selected", "submission": 0, "result": 0},
            {"event": "queue_selected", "submission": 1, "result": 1},
            {"event": "queue_selected", "submission": 2, "result": 0},
        ]
        errors, assertions = e2e.validate_events(queue_events, multiple_queues["expected_events"], [])
        self.assertEqual([], errors)
        self.assertEqual([2, 1], [assertion["observed"] for assertion in assertions])
        errors, _ = e2e.validate_events(
            queue_events + [{"event": "queue_selected", "submission": 3, "result": 1}],
            multiple_queues["expected_events"],
            [],
        )
        self.assertTrue(any("got 2" in error for error in errors))

    def test_absolute_and_traversing_case_names_are_rejected(self):
        for name in ("../escape", "/absolute", "C-drive", ".", "a/../../b", "UPPER"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "matrix.json"
                write_json(path, {"schema_version": 1, "cases": [{"name": name}]})
                with self.assertRaises(e2e.E2EError):
                    e2e.parse_matrix(path)

    def test_reserved_and_arbitrary_environment_overrides_are_rejected(self):
        for environment in ({"VK_LAYER_CAPTURE_FOLDER": "/tmp/escape"}, {"LD_LIBRARY_PATH": "/bad"}, {"OTHER": "x"}):
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "matrix.json"
                write_json(path, {"schema_version": 1, "cases": [{"name": "case", "environment": environment}]})
                with self.assertRaises(e2e.E2EError):
                    e2e.parse_matrix(path)


class ValidatorTests(unittest.TestCase):
    def test_exact_hierarchy_contract_and_spirv_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root, executed_index=3)
            reference = Path(directory) / "reference.spv"
            reference.write_bytes(paths["spirv"].read_bytes())
            expected = valid_expected(3)
            expected["spirv_file"] = str(reference)
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(4, result["documents"])
            self.assertTrue(result["spirv_comparison"]["captures"][0]["matches"])
            self.assertEqual(e2e.sha256(reference), result["spirv_comparison"]["reference_sha256"])

    def test_unavailable_static_artifacts_require_producer_explanations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)

            pipeline = read_json(paths["pipeline"])
            pipeline["artifacts"] = []
            pipeline["shader"]["spirv_available"] = False
            write_json(paths["pipeline"], pipeline)
            paths["spirv"].unlink()
            (root / "pipeline_000000/debug_database.bin").unlink()
            (root / "pipeline_000000/neural_statistics_info.txt").unlink()

            capture = read_json(paths["capture"])
            capture["warnings"] = [
                "debug database was unavailable for pipeline 0",
                "neural statistics info was unavailable for pipeline 0",
            ]
            write_json(paths["capture"], capture)
            expected = valid_expected()
            expected["artifact_classes"] = ["dispatch_statistics"]
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])

            pipeline["shader"]["spirv_available"] = True
            write_json(paths["pipeline"], pipeline)
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("shader_module" in error for error in result["errors"]))

            pipeline["shader"]["spirv_available"] = False
            write_json(paths["pipeline"], pipeline)
            capture["warnings"].remove("debug database was unavailable for pipeline 0")
            write_json(paths["capture"], capture)
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("debug_database" in error for error in result["errors"]))

    def test_static_omission_warning_cannot_contradict_a_materialized_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            capture = read_json(paths["capture"])
            capture["warnings"] = ["debug database was unavailable for pipeline 0"]
            write_json(paths["capture"], capture)
            result = e2e.validate_capture(root, valid_expected(), 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("expected exactly 0 debug_database" in error for error in result["errors"]))

    def test_error_capture_may_publish_dispatch_metadata_before_raw_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            capture = read_json(paths["capture"])
            capture["status"] = "error"
            capture["error"] = "mapped statistics collection failed"
            write_json(paths["capture"], capture)
            dispatch = read_json(paths["dispatch"])
            dispatch["artifacts"] = []
            write_json(paths["dispatch"], dispatch)
            paths["raw"].unlink()

            expected = valid_expected()
            expected["capture_status"] = "error"
            expected["artifact_classes"] = ["debug_database", "statistics_info", "shader_module"]
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])

            capture["status"] = "complete"
            capture["error"] = None
            write_json(paths["capture"], capture)
            expected["capture_status"] = "complete"
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("exactly one" in error for error in result["errors"]))

    def test_error_capture_accepts_every_static_artifact_publication_prefix(self):
        artifact_order = list(e2e.STATIC_ARTIFACT_ORDER)
        for prefix_length in range(len(artifact_order) + 1):
            with self.subTest(prefix_length=prefix_length), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                paths = write_capture(root)
                pipeline = read_json(paths["pipeline"])
                descriptors = pipeline["artifacts"]
                self.assertEqual(artifact_order, [item["type"] for item in descriptors])
                for descriptor in descriptors[prefix_length:]:
                    (root / descriptor["path"]).unlink()
                pipeline["artifacts"] = descriptors[:prefix_length]
                write_json(paths["pipeline"], pipeline)

                capture = read_json(paths["capture"])
                capture["status"] = "error"
                capture["error"] = "static artifact publication failed"
                write_json(paths["capture"], capture)

                expected = valid_expected()
                expected["capture_status"] = "error"
                expected["artifact_classes"] = ["dispatch_statistics", *artifact_order[:prefix_length]]
                result = e2e.validate_capture(root, expected, 1)
                self.assertTrue(result["ok"], result["errors"])

    def test_error_capture_rejects_non_prefix_static_artifact_sets_and_order(self):
        cases = {
            "missing-middle": (0, 2),
            "missing-first": (1, 2),
            "reordered": (1, 0, 2),
        }
        for name, retained_indices in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                paths = write_capture(root)
                pipeline = read_json(paths["pipeline"])
                descriptors = pipeline["artifacts"]
                retained = set(retained_indices)
                for index, descriptor in enumerate(descriptors):
                    if index not in retained:
                        (root / descriptor["path"]).unlink()
                pipeline["artifacts"] = [descriptors[index] for index in retained_indices]
                write_json(paths["pipeline"], pipeline)

                capture = read_json(paths["capture"])
                capture["status"] = "error"
                capture["error"] = "static artifact publication failed"
                write_json(paths["capture"], capture)

                expected = valid_expected()
                expected["capture_status"] = "error"
                expected["artifact_classes"] = [
                    "dispatch_statistics", *(descriptors[index]["type"] for index in retained_indices)
                ]
                result = e2e.validate_capture(root, expected, 1)
                self.assertFalse(result["ok"], result)
                self.assertTrue(any("published prefix" in error for error in result["errors"]), result["errors"])

    def test_error_capture_still_validates_every_present_static_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            pipeline = read_json(paths["pipeline"])
            descriptors = pipeline["artifacts"]
            for descriptor in descriptors[1:]:
                (root / descriptor["path"]).unlink()
            pipeline["artifacts"] = descriptors[:1]
            pipeline["artifacts"][0]["size"] += 1
            write_json(paths["pipeline"], pipeline)

            capture = read_json(paths["capture"])
            capture["status"] = "error"
            capture["error"] = "static artifact publication failed"
            write_json(paths["capture"], capture)

            expected = valid_expected()
            expected["capture_status"] = "error"
            expected["artifact_classes"] = ["debug_database", "dispatch_statistics"]
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"], result)
            self.assertTrue(any("size mismatch" in error for error in result["errors"]), result["errors"])

    def test_producer_shaped_internal_temporary_is_status_aware(self):
        temporary_name = (
            ".capture-internal-0000000000000001-0000000000000002-0000000000000003.tmp")
        for status in ("error", "incomplete", "complete"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                paths = write_capture(root)
                (paths["pipeline"].parent / temporary_name).write_bytes(b"partially written publication")
                capture = read_json(paths["capture"])
                capture["status"] = status
                capture["error"] = "publication failed" if status == "error" else None
                write_json(paths["capture"], capture)

                expected = valid_expected()
                expected["capture_status"] = status
                result = e2e.validate_capture(root, expected, 1)
                if status == "complete":
                    self.assertFalse(result["ok"], result)
                    self.assertTrue(any("complete capture" in error for error in result["errors"]),
                                    result["errors"])
                else:
                    self.assertTrue(result["ok"], result["errors"])

    def test_arbitrary_temporary_names_remain_invalid_for_error_captures(self):
        names = (
            "capture.json.tmp",
            ".capture-internal-not-a-producer-name.tmp",
            ".capture-internal-0000000000000000-0000000000000000-000000000000000G.tmp",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                paths = write_capture(root)
                (root / name).write_bytes(b"not a reserved producer temporary")
                capture = read_json(paths["capture"])
                capture["status"] = "error"
                capture["error"] = "publication failed"
                write_json(paths["capture"], capture)

                expected = valid_expected()
                expected["capture_status"] = "error"
                result = e2e.validate_capture(root, expected, 1)
                self.assertFalse(result["ok"], result)
                self.assertTrue(any(
                    "temporary file remains" in error or "internal publication file remains" in error
                    for error in result["errors"]), result["errors"])

    def test_only_text_statistics_info_may_be_zero_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            info = root / "pipeline_000000/neural_statistics_info.txt"
            info.write_bytes(b"")
            pipeline = read_json(paths["pipeline"])
            descriptor = next(item for item in pipeline["artifacts"] if item["type"] == "statistics_info")
            descriptor["size"] = 0
            write_json(paths["pipeline"], pipeline)
            result = e2e.validate_capture(root, valid_expected(), 1)
            self.assertTrue(result["ok"], result["errors"])

            binary_info = info.with_suffix(".bin")
            info.rename(binary_info)
            descriptor["path"] = "pipeline_000000/neural_statistics_info.bin"
            write_json(paths["pipeline"], pipeline)
            result = e2e.validate_capture(root, valid_expected(), 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("only a text statistics-info artifact may be empty" in error
                                for error in result["errors"]))

    def test_statistics_content_expectations_distinguish_zero_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            expected = valid_expected()
            expected["statistics_contents"] = [{"dispatch_id": 0, "content": "nonzero"}]
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])

            paths["raw"].write_bytes(b"\x00" * 5)
            expected["statistics_contents"] = [{"dispatch_id": 0, "content": "zero"}]
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])

            expected["statistics_contents"] = [{"dispatch_id": 0, "content": "nonzero"}]
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("expected nonzero, got zero" in error for error in result["errors"]))

    def test_statistics_difference_expectation_rejects_identical_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            dispatch_dir = root / "pipeline_000000/session_000000/dispatch_000001"
            dispatch_dir.mkdir()
            second_raw = dispatch_dir / "statistics_mode1.bin"
            second_raw.write_bytes(b"different")
            write_json(dispatch_dir / "dispatch.json", {
                "schema_version": 2,
                "id": 1,
                "session_id": 0,
                "executed_index": 1,
                "diagnostic_command_buffer_handle": 14,
                "artifacts": [{
                    "path": "pipeline_000000/session_000000/dispatch_000001/statistics_mode1.bin",
                    "type": "dispatch_statistics",
                    "size": second_raw.stat().st_size,
                }],
            })
            session = read_json(paths["session"])
            session["dispatches"].append({
                "id": 1,
                "path": "pipeline_000000/session_000000/dispatch_000001/dispatch.json",
            })
            write_json(paths["session"], session)
            expected = valid_expected()
            expected["pipelines"][0]["sessions"][0]["dispatches"].append(
                {"id": 1, "executed_index": 1}
            )
            expected["statistics_differ"] = [[0, 1]]
            result = e2e.validate_capture(root, expected, 1)
            self.assertTrue(result["ok"], result["errors"])

            second_raw.write_bytes(paths["raw"].read_bytes())
            dispatch = read_json(dispatch_dir / "dispatch.json")
            dispatch["artifacts"][0]["size"] = second_raw.stat().st_size
            write_json(dispatch_dir / "dispatch.json", dispatch)
            result = e2e.validate_capture(root, expected, 1)
            self.assertFalse(result["ok"])
            self.assertTrue(any("byte-identical" in error for error in result["errors"]))

    def _invalid(self, mutate, expected_fragment: str):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            paths = write_capture(root)
            mutate(root, paths)
            result = e2e.validate_capture(root, valid_expected(), 1)
            self.assertFalse(result["ok"], result)
            self.assertTrue(any(expected_fragment.lower() in error.lower() for error in result["errors"]), result["errors"])

    def test_traversal_absolute_backslash_and_drive_paths(self):
        values = ("../outside.json", "/absolute.json", "pipeline_000000\\pipeline.json", "C:/outside.json")
        for value in values:
            with self.subTest(value=value):
                self._invalid(lambda root, paths, value=value: self._set_ref(paths["capture"], "pipelines", value), "path")

    @staticmethod
    def _set_ref(path: Path, key: str, value: str):
        doc = read_json(path)
        doc[key][0]["path"] = value
        write_json(path, doc)

    def test_artifacts_must_be_immediate_children_of_their_owner(self):
        def nest_pipeline_artifact(root, paths):
            source = root / "pipeline_000000/debug_database.bin"
            nested = source.parent / "nested"
            nested.mkdir()
            moved = nested / source.name
            source.rename(moved)
            pipeline = read_json(paths["pipeline"])
            descriptor = next(item for item in pipeline["artifacts"] if item["type"] == "debug_database")
            descriptor["path"] = moved.relative_to(root).as_posix()
            write_json(paths["pipeline"], pipeline)

        def nest_dispatch_artifact(root, paths):
            source = paths["raw"]
            nested = source.parent / "nested"
            nested.mkdir()
            moved = nested / source.name
            source.rename(moved)
            dispatch = read_json(paths["dispatch"])
            descriptor = next(item for item in dispatch["artifacts"] if item["type"] == "dispatch_statistics")
            descriptor["path"] = moved.relative_to(root).as_posix()
            write_json(paths["dispatch"], dispatch)

        self._invalid(nest_pipeline_artifact, "immediate child")
        self._invalid(nest_dispatch_artifact, "immediate child")

    def test_symlink_is_rejected(self):
        def mutate(root, paths):
            target = paths["raw"]
            target.unlink()
            outside = root.parent / "outside.bin"
            outside.write_bytes(b"stats")
            try:
                os.symlink(outside, target)
            except OSError as exc:
                raise unittest.SkipTest(f"symlink creation unavailable: {exc}")
        self._invalid(mutate, "symlink")

    def test_duplicate_ids_and_references(self):
        def mutate(root, paths):
            doc = read_json(paths["capture"])
            doc["pipelines"].append(dict(doc["pipelines"][0]))
            write_json(paths["capture"], doc)
        self._invalid(mutate, "duplicate")

    def test_wrong_id_derived_path(self):
        self._invalid(lambda root, paths: self._set_ref(paths["capture"], "pipelines", "pipeline_000001/pipeline.json"),
                      "expected pipeline_000000")

    def test_duplicate_artifact_descriptor(self):
        def mutate(root, paths):
            doc = read_json(paths["pipeline"])
            doc["artifacts"].append(dict(doc["artifacts"][0]))
            write_json(paths["pipeline"], doc)
        self._invalid(mutate, "duplicate artifact")

    def test_unreferenced_public_file(self):
        self._invalid(lambda root, paths: (root / "orphan.bin").write_bytes(b"x"), "public file set mismatch")

    def test_unknown_artifact_type(self):
        def mutate(root, paths):
            doc = read_json(paths["dispatch"])
            doc["artifacts"][0]["type"] = "mystery"
            write_json(paths["dispatch"], doc)
        self._invalid(mutate, "unknown artifact")

    def test_multiple_raw_files_and_mismatched_mode(self):
        def duplicate(root, paths):
            extra = paths["raw"].with_name("extra.bin")
            extra.write_bytes(b"x")
            doc = read_json(paths["dispatch"])
            doc["artifacts"].append({
                "path": extra.relative_to(root).as_posix(), "type": "dispatch_statistics", "size": 1})
            write_json(paths["dispatch"], doc)
        self._invalid(duplicate, "exactly one")

        def mismatch(root, paths):
            wrong = paths["raw"].with_name("statistics_mode0.bin")
            paths["raw"].rename(wrong)
            doc = read_json(paths["dispatch"])
            doc["artifacts"][0]["path"] = wrong.relative_to(root).as_posix()
            write_json(paths["dispatch"], doc)
        self._invalid(mismatch, "does not match capture mode")

    def test_hash_and_constants_fields_anywhere(self):
        for field in ("sha256_hash", "graph_constants"):
            with self.subTest(field=field):
                def mutate(root, paths, field=field):
                    doc = read_json(paths["pipeline"])
                    doc[field] = [] if field == "graph_constants" else "abc"
                    write_json(paths["pipeline"], doc)
                self._invalid(mutate, "forbidden")

    def test_inconsistent_status_error(self):
        def mutate(root, paths):
            doc = read_json(paths["capture"])
            doc["status"] = "complete"
            doc["error"] = "should be null"
            write_json(paths["capture"], doc)
        self._invalid(mutate, "requires null error")

    def test_malformed_hierarchy_and_metadata(self):
        def mutate(root, paths):
            session = read_json(paths["session"])
            session["pipeline_id"] = 99
            session["friendly_name"] = 7
            write_json(paths["session"], session)
        self._invalid(mutate, "pipeline_id mismatch")

    def test_complete_lazy_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            root.mkdir()
            write_json(root / "capture.json", {
                "schema_version": 2, "status": "complete", "error": None,
                "capture": {"layer_name": "layer", "layer_version": "1", "commit_identity": "x",
                            "layer_implementation_version": 1, "statistics_mode": 1, "dispatch_filter": "7"},
                "devices": [{"id": 0, "name": "device", "vendor_id": None, "device_id": None,
                             "driver_version": None, "api_version": None}],
                "warnings": [], "pipelines": [],
            })
            result = e2e.validate_capture(root, {"capture_status": "complete", "pipelines": []}, 1)
            self.assertTrue(result["ok"], result["errors"])


class OrchestratorTests(unittest.TestCase):
    def _script(self, directory: Path) -> Path:
        script = directory / "fake_fixture.py"
        script.write_text(
            "import json, os, pathlib, subprocess, sys, time\n"
            "mode=sys.argv[1]\n"
            "if mode=='tree-sleep':\n"
            " marker=pathlib.Path(sys.argv[2])\n"
            " subprocess.Popen([sys.executable,'-c',\"import pathlib,time;time.sleep(1);pathlib.Path(r'%s').write_text('alive')\" % marker])\n"
            " time.sleep(10)\n"
            "if mode=='sleep': time.sleep(5)\n"
            "if mode=='fail': raise SystemExit(7)\n"
            "print('E2E_EVENT '+json.dumps({'event':'fixture-ok','timestamp_ns':time.time_ns()}), flush=True)\n",
            encoding="utf-8")
        return script

    def _case(self, script: Path, mode: str, expected_exit: int = 0):
        return {"name": mode, "fixture": {"args": [str(script), mode]}, "expected_exit": expected_exit,
                "layer": {"statistics_mode": 1, "dispatch_filter": ""},
                "expected_events": [{"event": "fixture-ok", "count": 1}]}

    def test_run_summary_events_and_root_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self._script(root)
            result = e2e.run_case(self._case(script, "pass"), Path(sys.executable), root / "runs", {}, 2)
            self.assertEqual("pass", result["result"])
            self.assertEqual((root / "runs").resolve(), Path(result["run_dir"]).resolve().parent)
            self.assertTrue(result["event_assertions"][0]["ok"])
            summary = {"ok": True, "matrix": "test", "cases": [result]}
            e2e.write_reports(summary, root / "summary.json", root / "report.txt")
            self.assertTrue(json.loads((root / "summary.json").read_text())["ok"])
            self.assertIn("behavioral_assertion", (root / "report.txt").read_text())
            for name in ("../escape", "/absolute"):
                with self.assertRaises(e2e.E2EError):
                    e2e._create_run_dir(root / "runs", name)

    def test_expected_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self._script(root)
            accepted = self._case(script, "fail", 7)
            accepted.pop("expected_events")
            result = e2e.run_case(accepted, Path(sys.executable), root / "accepted", {}, 2)
            self.assertEqual("pass", result["result"])

    def test_expected_stderr_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {
                "name": "diagnostic",
                "fixture": {"args": ["-c", "echo fixture-diagnostic >&2; exit 7"]},
                "expected_exit": 7,
                "expected_stderr_contains": "fixture-diagnostic",
                "layer": {"statistics_mode": 1, "dispatch_filter": ""},
            }
            result = e2e.run_case(case, Path("/bin/sh"), root / "accepted", {}, 2)
            self.assertEqual("pass", result["result"])
            case["expected_stderr_contains"] = "missing-diagnostic"
            result = e2e.run_case(case, Path("/bin/sh"), root / "rejected", {}, 2)
            self.assertEqual("fail", result["result"])

    def test_timeout_kills_child_process_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "child-survived.txt"
            script = self._script(root)
            case = self._case(script, "tree-sleep")
            case["fixture"]["args"].append(str(marker))
            case.pop("expected_events")
            started = time.monotonic()
            result = e2e.run_case(case, Path(sys.executable), root / "runs", {}, 0.2)
            self.assertTrue(result["timed_out"])
            self.assertLess(time.monotonic() - started, 5)
            time.sleep(1.3)
            self.assertFalse(marker.exists(), "timed-out child process survived process-tree cleanup")
            self.assertEqual("fail", result["result"])


if __name__ == "__main__":
    unittest.main()
