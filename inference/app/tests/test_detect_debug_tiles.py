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
``/internal/detect-debug?tiles=1`` with the raw per-tile outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings, get_settings
from app.main import _DETECTOR_CACHE, app, gatekeeper_auth
from app.tests.tiled_stubs import (
    N_QUERIES,
    REGION_CLASS,
    TileScript,
    install_ort_stub,
    load_fixture,
    script_from_raw,
    write_artifact,
    write_frame,
)

FRAME_W, FRAME_H = 4056, 3040
RAW = {"l": load_fixture("raw_tiles_l.json"), "s": load_fixture("raw_tiles_s.json")}
EXPECTED = {"l": load_fixture("expected_regions_l.json"), "s": load_fixture("expected_regions_s.json")}
RAW_L = RAW["l"]
SOURCE = "frame-04.jpg"
OVERHANGING = [("l", "frame-13.jpg"), ("s", "frame-04.jpg")]

PINNED_ORIGINS = [
    (0, 0), (1024, 0), (2048, 0), (2776, 0),
    (0, 1024), (1024, 1024), (2048, 1024), (2776, 1024),
    (0, 1760), (1024, 1760), (2048, 1760), (2776, 1760),
]


@pytest.fixture(scope="module")
def frame_bytes(tmp_path_factory):
    return write_frame(tmp_path_factory.mktemp("debug-frame") / "frame.jpg", FRAME_W, FRAME_H).read_bytes()


def _client(tmp_path, monkeypatch, artifact_kwargs, script=None, region_threshold=0.40):
    files = write_artifact(tmp_path, **artifact_kwargs)
    settings = Settings(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(tmp_path / "storage"),
        INFERENCE_DATA_ROOT=str(tmp_path / "storage"),
        INFERENCE_DETECTOR_IMPL="ort",
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_REGION_THRESHOLD=region_threshold,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = lambda: None
    if script is not None:
        install_ort_stub(monkeypatch, script, artifact_kwargs["imgsz"])
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean():
    yield
    app.dependency_overrides.clear()
    _DETECTOR_CACHE.pop("ort", None)


def _post(client, frame_bytes, tiles=None):
    params = {} if tiles is None else {"tiles": tiles}
    return client.post(
        "/internal/detect-debug",
        params=params,
        files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
    )


def _full_output_script(model: str, source_file: str):
    """The runtime's complete top-300 output for every tile of a golden frame:
    the fixture's retained rows (score >= 0.10) followed by synthetic
    low-score rows whose boxes overhang the tile on every side (negative and
    beyond 1280) and whose values are all distinct, so a dropped, reordered
    or clipped row cannot compare equal. Returns the script and, per tile,
    the ``(labels, boxes, scores)`` lists the response must carry."""
    tiles = sorted((t for t in RAW[model]["tiles"] if t["source_file"] == source_file), key=lambda t: t["tile_index"])
    outputs, expected = [], []
    for tile in tiles:
        n = len(tile["scores"])
        synthetic = np.arange(N_QUERIES - n, dtype=np.float32)
        labels = np.zeros(N_QUERIES, dtype=np.int64)
        boxes = np.zeros((N_QUERIES, 4), dtype=np.float32)
        scores = np.zeros(N_QUERIES, dtype=np.float32)
        labels[:n] = tile["labels"]
        if n:
            boxes[:n] = tile["boxes"]
        scores[:n] = tile["scores"]
        boxes[n:] = np.stack(
            [-1.5 - 0.25 * synthetic, 0.5 * synthetic - 3.0, 1281.25 + 0.25 * synthetic, 1290.5 - 0.5 * synthetic], axis=1
        )
        scores[n:] = 0.09 - 0.0002 * synthetic
        outputs.append((labels, boxes, scores))
        expected.append((labels.tolist(), boxes.tolist(), scores.tolist(), n))
    return TileScript(outputs), expected


@pytest.mark.parametrize("model, source_file", OVERHANGING)
def test_tiles_1_returns_the_raw_per_tile_outputs_format(tmp_path, monkeypatch, frame_bytes, model, source_file):
    script, expected_tiles = _full_output_script(model, source_file)
    client = _client(
        tmp_path,
        monkeypatch,
        dict(imgsz=1280, tiling={"tile": 1280, "stride": 1024}),
        script=script,
    )
    resp = _post(client, frame_bytes, tiles=1)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert [t["tile"] for t in body["tiles"]] == [
        f"frame_t{k:02d}_x{x0}_y{y0}" for k, (x0, y0) in enumerate(PINNED_ORIGINS)
    ]
    source_tiles = sorted(
        (t for t in RAW[model]["tiles"] if t["source_file"] == source_file), key=lambda t: t["tile_index"]
    )
    overhangs = 0
    for entry, source, (labels, boxes, scores, n) in zip(body["tiles"], source_tiles, expected_tiles):
        assert set(entry) == {"tile", "labels", "boxes", "scores"}
        assert len(entry["labels"]) == len(entry["boxes"]) == len(entry["scores"]) == N_QUERIES
        assert entry["labels"] == labels
        assert entry["boxes"] == boxes
        assert entry["scores"] == scores
        assert entry["boxes"][:n] == source["boxes"] and entry["scores"][:n] == source["scores"]
        assert entry["boxes"][n][0] < 0 and entry["boxes"][n][2] > 1280
        assert min(entry["boxes"][-1]) < 0 and max(entry["boxes"][-1]) > 1280
        overhangs += sum(1 for box in source["boxes"] if min(box) < 0 or max(box) > 1280)
    assert overhangs > 0

    expected = EXPECTED[model]["regions"]["0.40"][source_file]
    assert len(body["regions"]) == len(expected)
    for got, want in zip(body["regions"], expected):
        assert got["box"] == want["box"]
        assert got["score"] == want["score"]
        assert got["n_members"] == want["n_members"]

    assert body["detector_impl_id"] == "dfine-ort-detector"
    assert body["detector_profile"] == "legacy-full-frame"
    assert body["fusion_rule_id"] == "union-merge-v1"
    assert body["fusion_params"] == {
        "iou": 0.30, "contain": 0.60, "edge_tol": 4, "edge_extent": 0.50, "same_tile_iou": 0.60,
    }
    assert body["tiling"]["tile"] == 1280 and body["tiling"]["stride"] == 1024
    assert [tuple(o) for o in body["tiling"]["origins"]] == PINNED_ORIGINS
    assert body["tta_views"] == ["identity"]
    assert body["region_threshold"] == 0.40
    assert body["negligibility_record_sha256"] is None
    assert len(body["detections"]) == len(expected)
    assert all(d["label"] == REGION_CLASS for d in body["detections"])


def test_a_scripted_negative_box_and_its_region_stay_raw_in_the_debug_response(tmp_path, monkeypatch, frame_bytes):
    """Tile boxes AND regions in the debug output are raw: only the canonical
    detections are clamped to [0, 1]."""
    tiles = [
        (np.zeros(N_QUERIES, dtype=np.int64), np.zeros((N_QUERIES, 4), dtype=np.float32), np.zeros(N_QUERIES, dtype=np.float32))
        for _ in range(12)
    ]
    tiles[0][1][0] = [-3.5, 10.0, 100.0, 200.0]
    tiles[0][2][0] = 0.9
    client = _client(tmp_path, monkeypatch, dict(imgsz=1280, tiling={"tile": 1280, "stride": 1024}), script=TileScript(tiles))
    body = _post(client, frame_bytes, tiles=1).json()
    assert body["tiles"][0]["boxes"][0] == [-3.5, 10.0, 100.0, 200.0]
    assert [r["box"] for r in body["regions"]] == [[-3.5, 10.0, 100.0, 200.0]]
    assert [(d["x_min"], d["y_min"]) for d in body["detections"]] == [(0.0, 10.0 / FRAME_H)]


def test_without_tiles_the_response_keeps_the_plain_keys(tmp_path, monkeypatch, frame_bytes):
    client = _client(
        tmp_path,
        monkeypatch,
        dict(imgsz=1280, tiling={"tile": 1280, "stride": 1024}),
        script=script_from_raw(RAW_L, SOURCE),
    )
    for params in (None, 0):
        resp = _post(client, frame_bytes, tiles=params)
        assert resp.status_code == 200, resp.text
        assert set(resp.json()) == {"detector_impl_id", "designation_status", "execution_target", "detections"}
        _DETECTOR_CACHE.pop("ort", None)
        install_ort_stub(monkeypatch, script_from_raw(RAW_L, SOURCE), 1280)


def test_tiles_1_on_a_full_frame_detector_is_refused_not_silently_ignored(tmp_path, monkeypatch, frame_bytes):
    client = _client(tmp_path, monkeypatch, dict(imgsz=640, class_names=None), script=script_from_raw(RAW_L, SOURCE))
    resp = _post(client, frame_bytes, tiles=1)
    assert resp.status_code == 409
    assert "tiled" in resp.json()["detail"]


def test_tiles_must_be_0_or_1(tmp_path, monkeypatch, frame_bytes):
    client = _client(tmp_path, monkeypatch, dict(imgsz=1280, tiling={"tile": 1280, "stride": 1024}))
    assert _post(client, frame_bytes, tiles=2).status_code == 422
