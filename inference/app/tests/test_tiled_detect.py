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
The tiled path inside both detectors and the session path.

Stub runtimes return scripted per-tile tensors routed from the committed
golden raw fixture; nothing here loads a model.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings, get_settings
from app.contract import CONTRACT_VERSION, SessionResults
from app.detectors.base import ImageRef
from app.detectors import tiled_path
from app.detectors.dfine_common import DFineCanonicalizationError
from app.detectors.dfine_ort import DFineORTDetector
from app.detectors.dfine_tensorrt import DFineTensorRTDetector
from app.detectors.tiling import load_frame_rgb, preprocess_tile, tile_frame
from app.main import _DETECTOR_CACHE, _build_submission_from_metadata, app, gatekeeper_auth
from app.storage import read_session_metadata, read_session_results
from app.tests.tiled_stubs import (
    REGION_CLASS,
    TileScript,
    install_ort_stub,
    install_trt_runtime,
    load_fixture,
    pad_tile,
    script_from_raw,
    write_artifact,
    write_frame,
)

FRAME_W, FRAME_H = 4056, 3040
TILING = {"tile": 1280, "stride": 1024}

BEST = "frame-04.jpg"
OTHER = "frame-06.jpg"
OVERHANGING = "frame-13.jpg"

RAW = {"l": load_fixture("raw_tiles_l.json"), "s": load_fixture("raw_tiles_s.json")}
EXPECTED = {"l": load_fixture("expected_regions_l.json"), "s": load_fixture("expected_regions_s.json")}
RAW_L, EXPECTED_L = RAW["l"], EXPECTED["l"]
SOURCES = sorted(EXPECTED_L["regions"]["0.40"])


def _expected_regions(source_file: str, threshold: str = "0.40", model: str = "l"):
    return EXPECTED[model]["regions"][threshold][source_file]


def _canonical(value: float, extent: int) -> float:
    """The canonical mapping, written out independently: divide by the frame
    extent, clamp to [0, 1] -- the only transformation after fusion."""
    return min(max(value / extent, 0.0), 1.0)


def _assert_regions_match(detections, expected):
    """Exact (``==``) equality with the reference's regions: no tolerance, no rounding."""
    assert len(detections) == len(expected)
    for det, exp in zip(detections, expected):
        x0, y0, x1, y1 = exp["box"]
        assert det.label == REGION_CLASS
        assert det.x_min == _canonical(x0, FRAME_W)
        assert det.y_min == _canonical(y0, FRAME_H)
        assert det.x_max == _canonical(x1, FRAME_W)
        assert det.y_max == _canonical(y1, FRAME_H)
        assert det.score == exp["score"]


def _ort_detector(files, **kwargs):
    return DFineORTDetector(
        model_path=files.path,
        execution_target="cloud",
        provider="CPUExecutionProvider",
        provenance_path=files.provenance_path,
        designation_manifest_path=files.manifest_path,
        **kwargs,
    )


def _trt_detector(files, **kwargs):
    return DFineTensorRTDetector(
        engine_path=files.path,
        execution_target="jetson",
        provenance_path=files.provenance_path,
        designation_manifest_path=files.manifest_path,
        **kwargs,
    )


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory):
    return write_frame(tmp_path_factory.mktemp("frames") / "frame.jpg", FRAME_W, FRAME_H)


@pytest.fixture(scope="module")
def other_frame_path(tmp_path_factory):
    return write_frame(tmp_path_factory.mktemp("frames-other") / "other.jpg", FRAME_W, FRAME_H, seed=11)


def test_ort_yields_frame_normalized_region_detections(tmp_path, frame_path, monkeypatch):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    factory = install_ort_stub(monkeypatch, script_from_raw(RAW_L, BEST), 1280)
    detector = _ort_detector(files)

    result = detector.detect(ImageRef(image_id="img-1", filename=frame_path.name, path=frame_path))

    assert result.image_id == "img-1"
    _assert_regions_match(result.detections, _expected_regions(BEST))
    assert max(d.score for d in result.detections) == 0.9400203227996826

    assert [c["images_shape"] for c in factory.calls] == [(4, 3, 1280, 1280)] * 3
    assert all(c["orig_target_sizes"] == [[1280, 1280]] * 4 for c in factory.calls)

    tiles = tile_frame(load_frame_rgb(frame_path), 1280, 1024)
    expected_hashes = [hashlib.sha256(preprocess_tile(crop).tobytes()).hexdigest() for _, _, _, crop in tiles]
    seen = [h for c in factory.calls for h in c["images_sha256"]]
    assert seen == expected_hashes


def test_the_tile_batch_setting_sets_the_ort_chunk_size(tmp_path, frame_path, monkeypatch):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    factory = install_ort_stub(monkeypatch, script_from_raw(RAW_L, OTHER), 1280)
    detector = _ort_detector(files, tile_batch=5)
    detector.detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    assert [c["images_shape"][0] for c in factory.calls] == [5, 5, 2]


def test_the_region_threshold_is_applied_inside_the_fusion_stage(tmp_path, frame_path, monkeypatch):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    install_ort_stub(monkeypatch, script_from_raw(RAW_L, BEST), 1280)
    detector = _ort_detector(files, region_threshold=0.60)
    result = detector.detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    _assert_regions_match(result.detections, _expected_regions(BEST, "0.60"))


@pytest.mark.parametrize("model", ["l", "s"])
@pytest.mark.parametrize("source_file", SOURCES)
def test_every_golden_frame_yields_exactly_the_reference_regions(
    tmp_path, frame_path, monkeypatch, model, source_file
):
    """13 frames x 2 models, ``==`` throughout -- the overhanging frame included:
    the reference fuses the runtime's raw boxes, so the detector must too."""
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    install_ort_stub(monkeypatch, script_from_raw(RAW[model], source_file), 1280)
    result = _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    _assert_regions_match(result.detections, _expected_regions(source_file, model=model))


def test_the_golden_set_is_thirteen_frames_and_includes_the_overhanging_one():
    assert len(SOURCES) == 13
    assert OVERHANGING in SOURCES
    assert sorted(EXPECTED["s"]["regions"]["0.40"]) == SOURCES


@pytest.mark.parametrize(
    "model, source_file",
    [("l", OVERHANGING), ("l", "frame-09.jpg"), ("s", "frame-04.jpg")],
)
def test_the_golden_overhanging_boxes_reach_fusion_exactly_as_the_runtime_returned_them(
    tmp_path, frame_path, monkeypatch, model, source_file
):
    """The golden tiles whose boxes overhang the tile by a fraction of a pixel:
    the fusion stage's input holds those boxes unclipped and unrounded."""
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    install_ort_stub(monkeypatch, script_from_raw(RAW[model], source_file), 1280)
    captured = _capture_collect(monkeypatch)
    _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))

    (tile_outputs,) = captured
    golden = sorted((t for t in RAW[model]["tiles"] if t["source_file"] == source_file), key=lambda t: t["tile_index"])
    overhanging = 0
    for (index, x0, y0, boxes, scores), tile in zip(tile_outputs, golden):
        assert (index, x0, y0) == (tile["tile_index"], tile["offset_x"], tile["offset_y"])
        n = len(tile["boxes"])
        assert boxes[:n] == tile["boxes"] and scores[:n] == tile["scores"]
        overhanging += sum(1 for box, score in zip(tile["boxes"], tile["scores"]) if score >= 0.40 and (min(box) < 0 or max(box) > 1280))
    assert overhanging > 0


def _one_tile_script(tile_index: int, labels, boxes, scores) -> TileScript:
    return TileScript([pad_tile(labels, boxes, scores) if k == tile_index else pad_tile([], [], []) for k in range(12)])


def _capture_collect(monkeypatch):
    seen = []
    real = tiled_path.collect

    def spy(tile_outputs):
        seen.append(tile_outputs)
        return real(tile_outputs)

    monkeypatch.setattr(tiled_path, "collect", spy)
    return seen


def test_a_raw_box_with_a_negative_coordinate_reaches_fusion_unchanged_and_is_clamped_once_at_the_end(
    tmp_path, frame_path, monkeypatch
):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    script = TileScript(
        [
            pad_tile([0], [[-3.5, 10.0, 100.0, 200.0]], [0.9]) if k == 0
            else pad_tile([0], [[1000.0, 100.0, 1290.0, 200.0]], [0.8]) if k == 11
            else pad_tile([], [], [])
            for k in range(12)
        ]
    )
    install_ort_stub(monkeypatch, script, 1280)
    captured = _capture_collect(monkeypatch)

    detector = _ort_detector(files)
    result = detector.detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))

    (tile_outputs,) = captured
    by_tile = {index: (x0, y0, boxes, scores) for index, x0, y0, boxes, scores in tile_outputs}
    assert by_tile[0][2][0] == [-3.5, 10.0, 100.0, 200.0]
    assert by_tile[11][2][0] == [1000.0, 100.0, 1290.0, 200.0]

    regions = {(d.x_min, d.y_min, d.x_max, d.y_max): d for d in result.detections}
    assert set(regions) == {
        (0.0, 10.0 / FRAME_H, 100.0 / FRAME_W, 200.0 / FRAME_H),
        (3776.0 / FRAME_W, (1760.0 + 100.0) / FRAME_H, 1.0, (1760.0 + 200.0) / FRAME_H),
    }


def test_the_edge_touch_rule_reads_the_raw_coordinate_of_an_overhanging_box(tmp_path, frame_path, monkeypatch):
    """A box ending 5 px past its tile's right edge does not touch that edge
    within the 4 px tolerance; clipped to the tile it would, and would join
    its neighbour across the boundary."""
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    script = TileScript(
        [
            pad_tile([0], [[1270.0, 100.0, 1285.0, 300.0]], [0.9]) if k == 0
            else pad_tile([0], [[0.0, 100.0, 10.0, 300.0]], [0.9]) if k == 1
            else pad_tile([], [], [])
            for k in range(12)
        ]
    )
    install_ort_stub(monkeypatch, script, 1280)
    result = _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    assert len(result.detections) == 2

    script = TileScript(
        [
            pad_tile([0], [[1270.0, 100.0, 1283.0, 300.0]], [0.9]) if k == 0
            else pad_tile([0], [[0.0, 100.0, 10.0, 300.0]], [0.9]) if k == 1
            else pad_tile([], [], [])
            for k in range(12)
        ]
    )
    install_ort_stub(monkeypatch, script, 1280)
    joined = _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    assert len(joined.detections) == 1
    assert joined.detections[0].x_min == 1024.0 / FRAME_W and joined.detections[0].x_max == 1283.0 / FRAME_W


@pytest.mark.parametrize("label", [1, -1])
def test_a_raw_label_outside_the_class_manifest_still_refuses(tmp_path, frame_path, monkeypatch, label):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    install_ort_stub(monkeypatch, _one_tile_script(0, [label], [[10.0, 10.0, 50.0, 50.0]], [0.9]), 1280)
    with pytest.raises(DFineCanonicalizationError, match=f"label index {label} out of range"):
        _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))


@pytest.mark.parametrize("max_batch, runs", [(1, [1] * 12), (12, [12])])
def test_tensorrt_chunks_to_the_engines_max_batch_and_finds_the_same_regions(
    tmp_path, frame_path, max_batch, runs
):
    files = write_artifact(tmp_path, kind="engine", imgsz=1280, tiling=TILING)
    detector = _trt_detector(files, tile_batch=12)
    context = install_trt_runtime(
        detector, script_from_raw(RAW_L, BEST), 1280, batch_profile=(1, max_batch, max_batch)
    )

    result = detector.detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))

    assert context.runs == runs
    assert context.orig_target_sizes == [[1280, 1280]] * 12
    _assert_regions_match(result.detections, _expected_regions(BEST))

    tiles = tile_frame(load_frame_rgb(frame_path), 1280, 1024)
    assert context.images_sha256 == [
        hashlib.sha256(preprocess_tile(crop).tobytes()).hexdigest() for _, _, _, crop in tiles
    ]


def test_tensorrt_never_exceeds_the_smaller_of_the_setting_and_the_engine(tmp_path, frame_path):
    files = write_artifact(tmp_path, kind="engine", imgsz=1280, tiling=TILING)
    detector = _trt_detector(files)
    context = install_trt_runtime(detector, script_from_raw(RAW_L, OTHER), 1280, batch_profile=(1, 12, 12))
    detector.detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    assert context.runs == [4, 4, 4]


@pytest.mark.parametrize("method", ["detect", "detect_tiled_debug"])
def test_tensorrt_tiled_path_keeps_the_thread_affinity_guard(tmp_path, frame_path, method):
    import threading

    from app.detectors.dfine_tensorrt import TensorRTThreadAffinityError

    files = write_artifact(tmp_path, kind="engine", imgsz=1280, tiling=TILING)
    detector = _trt_detector(files)
    script = script_from_raw(RAW_L, OTHER)
    context = install_trt_runtime(detector, script, 1280)
    errors = []

    def other_thread():
        try:
            getattr(detector, method)(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
        except TensorRTThreadAffinityError as exc:
            errors.append(exc)

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert len(errors) == 1
    assert context.runs == [] and script.cursor == 0


def test_a_tiled_artifact_never_takes_the_full_frame_resize_path(tmp_path, frame_path, monkeypatch):
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
    factory = install_ort_stub(monkeypatch, script_from_raw(RAW_L, OTHER), 1280)
    _ort_detector(files).detect(ImageRef(image_id="i", filename=frame_path.name, path=frame_path))
    assert all(c["orig_target_sizes"] != [[FRAME_W, FRAME_H]] for c in factory.calls)


@pytest.fixture()
def session_env(tmp_path, monkeypatch):
    """A tiled ORT artifact behind the registry, a client over a temp
    storage dir, and the cache cleared so the registry builds it fresh."""
    files = write_artifact(tmp_path, imgsz=1280, tiling=TILING)
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
        INFERENCE_TARGET_CLASS_SET=[REGION_CLASS],
        INFERENCE_CONFIDENCE_FLOOR=0.3,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = lambda: None
    yield storage, settings, TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()
    _DETECTOR_CACHE.pop("ort", None)


def _post_session(client, frames):
    return client.post(
        "/api/v1/sessions",
        data={"latitude": "10.0", "longitude": "20.0", "time_start": "2026-01-01T10:00:00Z", "time_end": "2026-01-01T10:30:00Z"},
        files=[("files", (name, BytesIO(path.read_bytes()), "image/jpeg")) for name, path in frames],
    )


def test_a_two_image_session_scores_by_max_region_and_picks_the_best_image(
    session_env, frame_path, other_frame_path, monkeypatch
):
    storage, settings, client = session_env
    install_ort_stub(monkeypatch, script_from_raw(RAW_L, BEST, OTHER), 1280)

    resp = _post_session(client, [("best.jpg", frame_path), ("other.jpg", other_frame_path)])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    metadata = read_session_metadata(storage, body["session_id"])
    ids = {record["filename"]: record["image_id"] for record in metadata["images"]}
    scores = {r["image_id"]: r["score"] for r in read_session_results(storage, body["session_id"])}

    assert scores[ids["best.jpg"]] == 0.9400203227996826
    assert scores[ids["other.jpg"]] == 0.8869433403015137
    assert body["best_image_id"] == ids["best.jpg"]
    assert body["infestation_score"] == 0.9400203227996826

    assert len(metadata["best_image_detections"]) == len(_expected_regions(BEST))

    assert metadata["contract_version"] == CONTRACT_VERSION == "1.5.0"
    submission = _build_submission_from_metadata(metadata, body["session_id"])
    assert submission.contract_version == "1.5.0"
    results = client.get(f"/api/v1/sessions/{body['session_id']}/results")
    assert results.status_code == 200
    SessionResults(**results.json())


@pytest.mark.parametrize("small_first", [False, True])
def test_a_frame_smaller_than_one_tile_refuses_the_whole_session(
    session_env, frame_path, tmp_path, monkeypatch, small_first
):
    storage, settings, client = session_env
    small = write_frame(tmp_path / "small.jpg", 1280, 720)
    script = script_from_raw(RAW_L, BEST)
    factory = install_ort_stub(monkeypatch, script, 1280)

    frames = [("big.jpg", frame_path), ("small.jpg", small)]
    resp = _post_session(client, frames[::-1] if small_first else frames)

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "small.jpg" in detail and "1280x720" in detail and "1280" in detail

    assert list(storage.glob("sessions/*/results.json")) == []
    assert list(storage.glob("sessions/*/metadata.json")) == []

    assert all(c["images_shape"][1:] == (3, 1280, 1280) for c in factory.calls)
    assert all(c["orig_target_sizes"] == [[1280, 1280]] * c["images_shape"][0] for c in factory.calls)
    assert (script.cursor == 0) if small_first else (script.cursor == 12)
