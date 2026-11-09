# Copyright 2026 Smart Farm Robotix VCC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detector profiles and the INT8 negligibility gate."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.config as config_module
from app.config import Settings, get_settings
from app.detectors.artifact_provenance import ArtifactTilingError
from app.detectors.dfine_ort import DFineORTDetector
from app.detectors.dfine_tensorrt import DFineTensorRTDetector
from app.detectors.profiles import (
    PROFILES,
    DetectorProfile,
    ProfileRefusal,
    assert_profile_matches,
    get_profile,
    negligibility_record_path,
)
from app.detectors.registry import get_detector
from app.main import _DETECTOR_CACHE, app, gatekeeper_auth
from app.storage import read_session_metadata
from app.tests.tiled_stubs import (
    ABSENT,
    TileScript,
    install_ort_stub,
    pad_tile,
    write_artifact,
    write_frame,
)

TILING = {"tile": 1280, "stride": 1024}
KINDS = ["onnx", "engine"]


def _build(kind, files, profile, record_dir=None):
    if kind == "onnx":
        return DFineORTDetector(
            model_path=files.path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=files.provenance_path,
            designation_manifest_path=files.manifest_path,
            profile=profile,
            negligibility_record_dir=record_dir,
        )
    return DFineTensorRTDetector(
        engine_path=files.path,
        execution_target="jetson",
        provenance_path=files.provenance_path,
        designation_manifest_path=files.manifest_path,
        profile=profile,
        negligibility_record_dir=record_dir,
    )


def _artifact(tmp_path, kind, variant, precision, tiling=TILING, imgsz=1280):
    extra = {}
    if variant is not ABSENT:
        extra["variant"] = variant
    if precision is not ABSENT:
        extra["precision"] = precision
    return write_artifact(tmp_path, kind=kind, imgsz=imgsz, tiling=tiling, extra=extra)


def test_the_five_profiles_are_exactly_the_declared_set():
    assert PROFILES == {
        "legacy-full-frame": DetectorProfile("legacy-full-frame", None, None, False, ("identity",), False),
        "edge-s-fp16": DetectorProfile("edge-s-fp16", "s", "fp16", True, ("identity",), False),
        "cloud-l-fp32": DetectorProfile("cloud-l-fp32", "l", "fp32", True, ("identity",), False),
        "cloud-l-fp32-tta": DetectorProfile("cloud-l-fp32-tta", "l", "fp32", True, ("identity", "hflip"), False),
        "edge-s-int8": DetectorProfile("edge-s-int8", "s", "int8", True, ("identity",), True),
    }
    assert not any("cascade" in name for name in PROFILES)


def test_the_profile_setting_is_distinct_from_the_scenario_topology_setting():
    settings = Settings(INFERENCE_SCENARIO="mixed")
    assert settings.detector_profile == "legacy-full-frame"
    assert settings.scenario == "mixed"
    assert Settings(INFERENCE_DETECTOR_PROFILE="edge-s-fp16").scenario == "cloud"
    with pytest.raises(ValidationError):
        Settings(INFERENCE_DETECTOR_PROFILE="cascade")


def test_an_unknown_profile_name_is_refused_by_name():
    with pytest.raises(ProfileRefusal, match="cascade"):
        get_profile("cascade")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "profile, variant, precision, tiling, field, required, recorded",
    [
        ("edge-s-fp16", "l", "fp16", TILING, "variant", "'s'", "'l'"),
        ("edge-s-fp16", "s", "fp32", TILING, "precision", "'fp16'", "'fp32'"),
        ("edge-s-fp16", "s", "fp16", ABSENT, "tiling", "tiling", "<missing>"),
        ("cloud-l-fp32-tta", "l", "fp16", TILING, "precision", "'fp32'", "'fp16'"),
        ("edge-s-int8", "s", "fp16", TILING, "precision", "'int8'", "'fp16'"),
        ("edge-s-fp16", ABSENT, "fp16", TILING, "variant", "'s'", "<missing>"),
        ("edge-s-fp16", "s", ABSENT, TILING, "precision", "'fp16'", "<missing>"),
        ("edge-s-fp16", 5, "fp16", TILING, "variant", "'s'", "5"),
        ("edge-s-fp16", "s", ["fp16"], TILING, "precision", "'fp16'", "['fp16']"),
        ("edge-s-fp16", "S", "fp16", TILING, "variant", "'s'", "'S'"),
    ],
)
def test_a_profile_refuses_an_artifact_that_does_not_match_it(
    kind, tmp_path, profile, variant, precision, tiling, field, required, recorded
):
    files = _artifact(tmp_path, kind, variant, precision, tiling)
    with pytest.raises(ProfileRefusal) as excinfo:
        _build(kind, files, profile)
    message = str(excinfo.value)
    assert profile in message
    assert field in message
    assert required in message
    assert recorded in message
    assert str(files.provenance_path) in message


def test_a_malformed_tiling_under_a_profile_refuses_through_the_shared_validator(tmp_path):
    files = _artifact(tmp_path, "onnx", "s", "fp16", tiling={"tile": 1280})
    with pytest.raises(ArtifactTilingError, match="tile"):
        _build("onnx", files, "edge-s-fp16")


def test_a_conforming_artifact_starts_under_each_declared_profile(tmp_path):
    cases = [
        ("edge-s-fp16", "s", "fp16", ("identity",)),
        ("cloud-l-fp32", "l", "fp32", ("identity",)),
        ("cloud-l-fp32-tta", "l", "fp32", ("identity", "hflip")),
    ]
    for profile, variant, precision, views in cases:
        directory = tmp_path / profile
        directory.mkdir()
        detector = _build("onnx", _artifact(directory, "onnx", variant, precision), profile)
        assert detector._profile_name == profile
        assert detector._tta_views == views
        assert detector._negligibility_record_sha256 is None


def test_the_default_profile_accepts_any_record(tmp_path):
    files = write_artifact(tmp_path, imgsz=640, class_names=None)
    detector = _build("onnx", files, "legacy-full-frame")
    assert detector._profile_name == "legacy-full-frame"
    assert detector._tta_views == ("identity",)


def test_stub_detectors_cannot_run_under_a_profile(monkeypatch):
    settings = Settings(INFERENCE_DETECTOR_PROFILE="edge-s-fp16")
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    for impl in ("lookup", "null"):
        with pytest.raises(ProfileRefusal, match=impl):
            get_detector(impl)


def test_the_registry_passes_the_configured_profile(tmp_path, monkeypatch):
    files = _artifact(tmp_path, "onnx", "s", "fp16")
    settings = Settings(
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_DETECTOR_PROFILE="edge-s-fp16",
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    assert get_detector("ort")._profile_name == "edge-s-fp16"
    mismatched = Settings(
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_DETECTOR_PROFILE="cloud-l-fp32-tta",
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: mismatched)
    with pytest.raises(ProfileRefusal, match="cloud-l-fp32-tta"):
        get_detector("ort")


def _int8(tmp_path, kind="onnx"):
    files = _artifact(tmp_path, kind, "s", "int8")
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    return files, record_dir, negligibility_record_path(record_dir, files.sha256)


@pytest.mark.parametrize("kind", KINDS)
def test_int8_refuses_when_no_record_exists(kind, tmp_path):
    files, record_dir, record = _int8(tmp_path, kind)
    with pytest.raises(ProfileRefusal) as excinfo:
        _build(kind, files, "edge-s-int8", record_dir)
    assert "edge-s-int8" in str(excinfo.value)
    assert str(record) in str(excinfo.value)


def test_int8_refuses_when_no_record_directory_is_configured(tmp_path):
    files, _, _ = _int8(tmp_path)
    with pytest.raises(ProfileRefusal, match="INFERENCE_NEGLIGIBILITY_RECORD_DIR"):
        _build("onnx", files, "edge-s-int8", None)


def test_int8_refuses_an_unreadable_record(tmp_path):
    files, record_dir, record = _int8(tmp_path)
    record.mkdir()
    with pytest.raises(ProfileRefusal, match="unreadable") as excinfo:
        _build("onnx", files, "edge-s-int8", record_dir)
    assert str(record) in str(excinfo.value)


@pytest.mark.parametrize("content", ["not json {", "[1, 2]", '"a string"', "null", "42"])
def test_int8_refuses_a_record_that_is_not_a_json_object(tmp_path, content):
    files, record_dir, record = _int8(tmp_path)
    record.write_text(content)
    with pytest.raises(ProfileRefusal) as excinfo:
        _build("onnx", files, "edge-s-int8", record_dir)
    assert str(record) in str(excinfo.value)


@pytest.mark.parametrize("body", [{"artifact_sha256": "0" * 64}, {}, {"artifact_sha256": None}, {"artifact_sha256": 7}])
def test_int8_refuses_a_record_naming_another_artifact_or_none(tmp_path, body):
    files, record_dir, record = _int8(tmp_path)
    record.write_text(json.dumps(body))
    with pytest.raises(ProfileRefusal) as excinfo:
        _build("onnx", files, "edge-s-int8", record_dir)
    message = str(excinfo.value)
    assert str(record) in message
    assert files.sha256 in message


OTHER_IDENTITIES = {
    "the source onnx sha256": {"source_onnx_sha256": "a" * 64},
    "the checkpoint sha256": {
        "pt_provenance": {"model_name": "dfine-s", "source": "s", "license": "Apache-2.0", "file_sha256": "b" * 64}
    },
    "an unrelated 64-hex string": {},
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("identity", sorted(OTHER_IDENTITIES))
def test_int8_refuses_a_record_naming_any_other_64_hex_identity(kind, tmp_path, identity):
    """The identity is the provenance's ``file_sha256``: a record whose
    ``artifact_sha256`` is some other 64-hex string of the same provenance
    (or none of it) does not clear the gate, even at the right path."""
    extra = {"variant": "s", "precision": "int8", **OTHER_IDENTITIES[identity]}
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling=TILING, extra=extra)
    other = {"the source onnx sha256": "a" * 64, "the checkpoint sha256": "b" * 64}.get(identity, "c" * 64)
    assert other != files.sha256
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    record = negligibility_record_path(record_dir, files.sha256)
    record.write_text(json.dumps({"artifact_sha256": other}))
    with pytest.raises(ProfileRefusal) as excinfo:
        _build(kind, files, "edge-s-int8", record_dir)
    message = str(excinfo.value)
    assert "edge-s-int8" in message and str(record) in message
    assert other in message and files.sha256 in message


@pytest.mark.parametrize("kind", KINDS)
def test_int8_does_not_accept_a_record_filed_under_another_identity(kind, tmp_path):
    """A record that names, and is filed under, a different 64-hex string leaves
    the loaded artifact's own record absent: the path is built from ``file_sha256``."""
    extra = {"variant": "s", "precision": "int8", "source_onnx_sha256": "a" * 64}
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling=TILING, extra=extra)
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    negligibility_record_path(record_dir, "a" * 64).write_text(json.dumps({"artifact_sha256": "a" * 64}))
    with pytest.raises(ProfileRefusal) as excinfo:
        _build(kind, files, "edge-s-int8", record_dir)
    assert str(negligibility_record_path(record_dir, files.sha256)) in str(excinfo.value)


@pytest.mark.parametrize("kind", KINDS)
def test_int8_identity_is_the_provenance_file_sha256(kind, tmp_path):
    extra = {"variant": "s", "precision": "int8", "source_onnx_sha256": "a" * 64}
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling=TILING, extra=extra)
    file_sha256 = json.loads(files.provenance_path.read_text())["file_sha256"]
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    record = negligibility_record_path(record_dir, file_sha256)
    record.write_text(json.dumps({"artifact_sha256": file_sha256}))
    detector = _build(kind, files, "edge-s-int8", record_dir)
    assert detector.artifact_sha256 == file_sha256
    assert detector._negligibility_record_sha256 == hashlib.sha256(record.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", KINDS)
def test_int8_starts_on_a_matching_record_and_records_its_sha256(kind, tmp_path):
    files, record_dir, record = _int8(tmp_path, kind)
    record.write_text(json.dumps({"artifact_sha256": files.sha256, "mode": "int8", "measured_delta": 0.0}))
    detector = _build(kind, files, "edge-s-int8", record_dir)
    assert detector._profile_name == "edge-s-int8"
    assert detector._negligibility_record_sha256 == hashlib.sha256(record.read_bytes()).hexdigest()


@pytest.fixture()
def client_env(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()

    def make(files, profile, record_dir=None, script=None, imgsz=1280):
        settings = Settings(
            INFERENCE_USING_GATEKEEPER="false",
            INFERENCE_ORCH_PUSH_ENABLED="false",
            INFERENCE_STORAGE_DIR=str(storage),
            INFERENCE_DATA_ROOT=str(storage),
            INFERENCE_DETECTOR_IMPL="ort",
            INFERENCE_ONNX_MODEL_PATH=str(files.path),
            INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
            INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
            INFERENCE_DETECTOR_PROFILE=profile,
            INFERENCE_NEGLIGIBILITY_RECORD_DIR=str(record_dir or tmp_path / "records"),
            INFERENCE_TARGET_CLASS_SET=["aphid_infestation_region"],
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings)
        monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[gatekeeper_auth] = lambda: None
        factory = install_ort_stub(monkeypatch, script, imgsz)
        return storage, TestClient(app, raise_server_exceptions=False), factory

    yield make
    app.dependency_overrides.clear()
    _DETECTOR_CACHE.pop("ort", None)


def _session(client, path):
    return client.post(
        "/api/v1/sessions",
        data={"latitude": "10.0", "longitude": "20.0"},
        files=[("files", (path.name, BytesIO(path.read_bytes()), "image/png"))],
    )


def test_default_profile_serves_the_full_frame_path_and_the_metadata_records_it(client_env, tmp_path):
    files = write_artifact(tmp_path, imgsz=640, class_names=None)
    frame = write_frame(tmp_path / "frame.png", 200, 100)
    storage, client, factory = client_env(files, "legacy-full-frame", script=TileScript([pad_tile([], [], [])]), imgsz=640)

    resp = _session(client, frame)

    assert resp.status_code == 200, resp.text
    (call,) = factory.calls
    assert call["images_shape"] == (1, 3, 640, 640)
    assert call["orig_target_sizes"] == [[200, 100]]
    metadata = read_session_metadata(storage, resp.json()["session_id"])
    assert metadata["detector_profile"] == "legacy-full-frame"
    assert metadata["tiling"] is None
    assert metadata["fusion_rule_id"] is None and metadata["fusion_params"] is None
    assert metadata["tta_views"] == ["identity"]
    assert metadata["negligibility_record_sha256"] is None


def test_the_int8_profile_records_the_negligibility_record_in_the_session_metadata(client_env, tmp_path):
    files, record_dir, record = _int8(tmp_path)
    record.write_text(json.dumps({"artifact_sha256": files.sha256}))
    frame = write_frame(tmp_path / "frame.png", 1300, 1300)
    storage, client, _ = client_env(
        files, "edge-s-int8", record_dir, script=TileScript([pad_tile([], [], [])] * 4)
    )

    resp = _session(client, frame)

    assert resp.status_code == 200, resp.text
    metadata = read_session_metadata(storage, resp.json()["session_id"])
    assert metadata["detector_profile"] == "edge-s-int8"
    assert metadata["negligibility_record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert metadata["tta_views"] == ["identity"]
