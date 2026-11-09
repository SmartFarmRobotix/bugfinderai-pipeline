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

"""Flip test-time augmentation in the tiled path."""

from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings, get_settings
from app.detectors.base import ImageRef
from app.detectors.dfine_ort import DFineORTDetector
from app.detectors.artifact_provenance import TilingSpec
from app.detectors.tiled_path import run_tiled
from app.detectors.tiling import load_frame_rgb, preprocess_tile, tile_frame
from app.main import _DETECTOR_CACHE, app, gatekeeper_auth
from app.storage import read_session_metadata
from app.tests.tiled_stubs import (
    REGION_CLASS,
    TileScript,
    install_ort_stub,
    load_fixture,
    pad_tile,
    script_from_raw,
    write_artifact,
    write_frame,
)

TILE = 1280
TILING = {"tile": TILE, "stride": 1024}
FRAME_W, FRAME_H = 1300, 1300
TILES = 4

RAW_L = load_fixture("raw_tiles_l.json")
EXPECTED_L = load_fixture("expected_regions_l.json")
GOLDEN = "frame-04.jpg"


def _mirror(box):
    """A box in a horizontally mirrored tile: x' = tile - x, x0/x1 swapped."""
    return [TILE - box[2], box[1], TILE - box[0], box[3]]


def _empty():
    return pad_tile([], [], [])


def _one(box, score):
    return pad_tile([0], [box], [score])


def _detector(tmp_path, profile, variant, precision, **kwargs):
    files = write_artifact(
        tmp_path, imgsz=TILE, tiling=TILING, extra={"variant": variant, "precision": precision}
    )
    return DFineORTDetector(
        model_path=files.path,
        execution_target="cloud",
        provider="CPUExecutionProvider",
        provenance_path=files.provenance_path,
        designation_manifest_path=files.manifest_path,
        profile=profile,
        **kwargs,
    ), files


@pytest.fixture()
def frame(tmp_path):
    return write_frame(tmp_path / "frame.png", FRAME_W, FRAME_H)


def _ref(frame):
    return ImageRef(image_id="i", filename=frame.name, path=frame)


def test_the_same_object_in_both_views_is_one_region_and_the_flip_view_is_the_mirrored_tile(
    tmp_path, frame, monkeypatch
):
    detector, _ = _detector(tmp_path, "cloud-l-fp32-tta", "l", "fp32", tile_batch=TILES)
    box = [100.0, 100.0, 300.0, 200.0]
    script = TileScript([_one(box, 0.90), _empty(), _empty(), _empty()] + [_one(_mirror(box), 0.80), _empty(), _empty(), _empty()])
    factory = install_ort_stub(monkeypatch, script, TILE)

    result = detector.detect(_ref(frame))

    assert len(result.detections) == 1
    (det,) = result.detections
    assert det.label == REGION_CLASS
    assert det.score == pytest.approx(0.90)
    assert (det.x_min, det.y_min, det.x_max, det.y_max) == pytest.approx(
        (100 / FRAME_W, 100 / FRAME_H, 300 / FRAME_W, 200 / FRAME_H), abs=1e-9
    )

    identity_call, flip_call = factory.calls
    tiles = tile_frame(load_frame_rgb(frame), TILE, 1024)
    tensors = [preprocess_tile(crop) for _, _, _, crop in tiles]
    assert identity_call["images_sha256"] == [hashlib.sha256(t.tobytes()).hexdigest() for t in tensors]
    assert flip_call["images_sha256"] == [
        hashlib.sha256(np.ascontiguousarray(t[:, :, ::-1]).tobytes()).hexdigest() for t in tensors
    ]


def test_slightly_different_views_fuse_into_the_union_box_with_the_max_score(tmp_path, frame, monkeypatch):
    detector, _ = _detector(tmp_path, "cloud-l-fp32-tta", "l", "fp32", tile_batch=TILES)
    identity_box = [100.0, 100.0, 300.0, 200.0]
    flip_seen = _mirror([102.0, 102.0, 295.0, 201.0])
    script = TileScript(
        [_one(identity_box, 0.70), _empty(), _empty(), _empty()]
        + [_one(flip_seen, 0.95), _empty(), _empty(), _empty()]
    )
    install_ort_stub(monkeypatch, script, TILE)
    (det,) = detector.detect(_ref(frame)).detections
    assert det.score == pytest.approx(0.95)
    assert (det.x_min, det.y_min, det.x_max, det.y_max) == pytest.approx(
        (100 / FRAME_W, 100 / FRAME_H, 300 / FRAME_W, 201 / FRAME_H), abs=1e-9
    )


def test_an_object_only_the_flip_view_sees_is_kept(tmp_path, frame, monkeypatch):
    detector, _ = _detector(tmp_path, "cloud-l-fp32-tta", "l", "fp32", tile_batch=TILES)
    box = [400.0, 400.0, 600.0, 500.0]
    script = TileScript([_empty()] * 4 + [_one(_mirror(box), 0.66), _empty(), _empty(), _empty()])
    install_ort_stub(monkeypatch, script, TILE)
    (det,) = detector.detect(_ref(frame)).detections
    assert det.score == pytest.approx(0.66)
    assert det.x_min == pytest.approx(400 / FRAME_W, abs=1e-9)


def test_view_detections_below_the_threshold_never_bridge(tmp_path, frame, monkeypatch):
    """Threshold first: a sub-threshold detection in the other view cannot
    join two views' detections, exactly as in the frozen cross-tile rule."""
    detector, _ = _detector(tmp_path, "cloud-l-fp32-tta", "l", "fp32", tile_batch=TILES)
    a, b = [100.0, 100.0, 300.0, 200.0], [100.0, 150.0, 300.0, 250.0]
    bridge = [100.0, 125.0, 300.0, 225.0]
    script = TileScript(
        [pad_tile([0, 0], [a, b], [0.9, 0.8]), _empty(), _empty(), _empty()]
        + [pad_tile([0], [_mirror(bridge)], [0.2]), _empty(), _empty(), _empty()]
    )
    install_ort_stub(monkeypatch, script, TILE)
    assert len(detector.detect(_ref(frame)).detections) == 2


def test_identity_only_views_are_byte_identical_to_the_plain_tiled_path_over_the_golden_set(
    tmp_path, monkeypatch
):
    source = tmp_path / "frame.jpg"
    write_frame(source, 4056, 3040)
    ref = ImageRef(image_id="i", filename=source.name, path=source)

    outputs = []
    for index, profile in enumerate(("legacy-full-frame", "edge-s-fp16")):
        directory = tmp_path / f"d{index}"
        directory.mkdir()
        detector, _ = _detector(directory, profile, "s", "fp16")
        install_ort_stub(monkeypatch, script_from_raw(RAW_L, GOLDEN), TILE)
        outputs.append(detector.detect(ref).detections)
    assert outputs[0] == outputs[1]

    plain = run_tiled(
        ref,
        tiling=TilingSpec(TILE, 1024),
        imgsz=TILE,
        class_names=[REGION_CLASS],
        run_batch=_stub_runner(monkeypatch),
        batch=4,
        threshold=0.40,
    )
    assert plain.detections == outputs[0]
    assert len(plain.detections) == len(EXPECTED_L["regions"]["0.40"][GOLDEN])


def _stub_runner(monkeypatch):
    """run_batch over a fresh scripted stub session (identity view only)."""
    from app.detectors.dfine_common import RawTensor

    script = script_from_raw(RAW_L, GOLDEN)

    def run(images, sizes):
        labels, boxes, scores = script.take(images.shape[0])
        return [
            RawTensor("labels", labels.dtype, labels),
            RawTensor("boxes", boxes.dtype, boxes),
            RawTensor("scores", scores.dtype, scores),
            RawTensor("orig_target_sizes", sizes.dtype, sizes),
        ]

    return run


def test_unknown_or_repeated_views_are_refused():
    for views in (("hflip",), ("identity", "vflip"), ("identity", "identity"), ()):
        with pytest.raises(ValueError, match="views"):
            run_tiled(
                ImageRef("i", "f.png", None),
                tiling=TilingSpec(TILE, 1024),
                imgsz=TILE,
                class_names=[REGION_CLASS],
                run_batch=None,
                batch=1,
                threshold=0.4,
                views=views,
            )


def _session_metadata(tmp_path, monkeypatch, profile, script):
    """One tiled session over a 2x2-grid frame under ``profile``; the session's metadata."""
    files = write_artifact(tmp_path, imgsz=TILE, tiling=TILING, extra={"variant": "l", "precision": "fp32"})
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
        INFERENCE_DETECTOR_PROFILE=profile,
        INFERENCE_TARGET_CLASS_SET=[REGION_CLASS],
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = lambda: None
    try:
        install_ort_stub(monkeypatch, script, TILE)
        frame = write_frame(tmp_path / "frame.png", FRAME_W, FRAME_H)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/sessions",
            data={"latitude": "10.0", "longitude": "20.0"},
            files=[("files", ("frame.png", BytesIO(frame.read_bytes()), "image/png"))],
        )
        assert resp.status_code == 200, resp.text
        return read_session_metadata(storage, resp.json()["session_id"])
    finally:
        app.dependency_overrides.clear()
        _DETECTOR_CACHE.pop("ort", None)


def test_the_session_metadata_records_the_tta_views(tmp_path, monkeypatch):
    box = [100.0, 100.0, 300.0, 200.0]
    script = TileScript([_one(box, 0.9), _empty(), _empty(), _empty()] + [_one(_mirror(box), 0.8), _empty(), _empty(), _empty()])
    metadata = _session_metadata(tmp_path, monkeypatch, "cloud-l-fp32-tta", script)
    assert metadata["tta_views"] == ["identity", "hflip"]
    assert metadata["detector_profile"] == "cloud-l-fp32-tta"
    assert metadata["best_image_detections"] and len(metadata["best_image_detections"]) == 1


def test_a_plain_cloud_profile_session_records_the_identity_view_only(tmp_path, monkeypatch):
    box = [100.0, 100.0, 300.0, 200.0]
    script = TileScript([_one(box, 0.9), _empty(), _empty(), _empty()])
    metadata = _session_metadata(tmp_path, monkeypatch, "cloud-l-fp32", script)
    assert metadata["tta_views"] == ["identity"]
    assert metadata["detector_profile"] == "cloud-l-fp32"
    assert metadata["best_image_detections"] and len(metadata["best_image_detections"]) == 1
