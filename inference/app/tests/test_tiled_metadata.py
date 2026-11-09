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

"""
Tiled-path provenance in the session metadata file, wire contract unchanged.

The contract source and its mirror are byte-identical, and the wire ``provenance`` block carries none of the
tiled-path fields.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings, get_settings
from app.contract import CONTRACT_VERSION, DetectorProvenance, SessionResults
from app.main import _DETECTOR_CACHE, _build_submission_from_metadata, app, gatekeeper_auth
from app.storage import read_session_metadata
from app.tests.tiled_stubs import (
    REGION_CLASS,
    install_ort_stub,
    load_fixture,
    script_from_raw,
    write_artifact,
    write_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PY = "inference/app/contract.py"
CONTRACT_MIRRORS = (
    "orchestrator/app/contract.py",
)

PINNED_ORIGINS = [
    [0, 0], [1024, 0], [2048, 0], [2776, 0],
    [0, 1024], [1024, 1024], [2048, 1024], [2776, 1024],
    [0, 1760], [1024, 1760], [2048, 1760], [2776, 1760],
]
TILED_KEYS = (
    "detector_profile", "tiling", "fusion_rule_id", "fusion_params", "tta_views",
    "region_threshold", "negligibility_record_sha256",
)

RAW_L = load_fixture("raw_tiles_l.json")
GOLDEN = "frame-04.jpg"


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory):
    return write_frame(tmp_path_factory.mktemp("meta-frame") / "frame.jpg", 4056, 3040)


@pytest.fixture()
def tiled_session(tmp_path, monkeypatch, frame_path):
    files = write_artifact(
        tmp_path, imgsz=1280, tiling={"tile": 1280, "stride": 1024}, extra={"variant": "s", "precision": "fp16"}
    )
    storage = tmp_path / "storage"
    storage.mkdir()
    settings = Settings(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(storage),
        INFERENCE_DATA_ROOT=str(storage),
        INFERENCE_DETECTOR_IMPL="ort",
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_DETECTOR_PROFILE="edge-s-fp16",
        INFERENCE_REGION_THRESHOLD=0.35,
        INFERENCE_TARGET_CLASS_SET=[REGION_CLASS],
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = lambda: None
    install_ort_stub(monkeypatch, script_from_raw(RAW_L, GOLDEN), 1280)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/v1/sessions",
        data={"latitude": "10.0", "longitude": "20.0", "time_start": "2026-01-01T10:00:00Z", "time_end": "2026-01-01T10:30:00Z"},
        files=[("files", ("frame.jpg", BytesIO(frame_path.read_bytes()), "image/jpeg"))],
    )
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    yield client, session_id, read_session_metadata(storage, session_id)
    app.dependency_overrides.clear()
    _DETECTOR_CACHE.pop("ort", None)


def test_the_metadata_file_names_the_fusion_rule_and_its_parameters(tiled_session):
    _, _, metadata = tiled_session
    assert metadata["fusion_rule_id"] == "union-merge-v1"
    assert metadata["fusion_params"] == {
        "iou": 0.30, "contain": 0.60, "edge_tol": 4, "edge_extent": 0.50, "same_tile_iou": 0.60,
    }
    assert metadata["tiling"]["tile"] == 1280
    assert metadata["tiling"]["stride"] == 1024
    assert metadata["tiling"]["origins"] == PINNED_ORIGINS
    assert metadata["tiling"]["grids"] == {"4056x3040": PINNED_ORIGINS}
    assert metadata["detector_profile"] == "edge-s-fp16"
    assert metadata["tta_views"] == ["identity"]
    assert metadata["region_threshold"] == 0.35
    assert metadata["negligibility_record_sha256"] is None


def test_the_session_results_still_validate_against_the_current_contract(tiled_session):
    client, session_id, metadata = tiled_session
    assert CONTRACT_VERSION == "1.5.0"
    assert metadata["contract_version"] == "1.5.0"
    results = client.get(f"/api/v1/sessions/{session_id}/results")
    assert results.status_code == 200
    SessionResults(**results.json())
    submission = _build_submission_from_metadata(metadata, session_id)
    assert submission.contract_version == "1.5.0"


def test_the_wire_provenance_block_carries_none_of_the_tiled_path_fields(tiled_session):
    _, _, metadata = tiled_session
    wire = metadata["provenance"]
    assert not set(TILED_KEYS) & set(wire)
    DetectorProvenance(**wire)
    assert set(TILED_KEYS) <= set(metadata)


def test_contract_mirror_is_byte_identical():
    """The contract source and its mirror are byte-identical. A missing file fails; nothing is skipped."""
    source = (REPO_ROOT / CONTRACT_PY).read_bytes()
    drifted = [
        mirror
        for mirror in CONTRACT_MIRRORS
        if (REPO_ROOT / mirror).read_bytes() != source
    ]
    assert drifted == [], f"mirrors drifted from {CONTRACT_PY}: {drifted}"
