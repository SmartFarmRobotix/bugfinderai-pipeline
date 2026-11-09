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
Image size and tiling grid from provenance: both D-FINE detector classes read ``imgsz`` and ``tiling`` from
the loaded provenance through one shared validator; no constructor carries
an image-size default.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.detectors.artifact_provenance import (
    ArtifactImageSizeError,
    ArtifactTilingError,
    TilingSpec,
)
from app.detectors.base import ImageRef
from app.detectors.dfine_ort import DFineORTDetector
from app.detectors.dfine_tensorrt import DFineTensorRTDetector
from app.tests.tiled_stubs import (
    ABSENT,
    REGION_CLASS,
    TileScript,
    install_ort_stub,
    install_trt_runtime,
    pad_tile,
    write_artifact,
    write_frame,
)

KINDS = ["onnx", "engine"]


def _build(kind, files):
    if kind == "onnx":
        return DFineORTDetector(
            model_path=files.path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=files.provenance_path,
            designation_manifest_path=files.manifest_path,
        )
    return DFineTensorRTDetector(
        engine_path=files.path,
        execution_target="jetson",
        provenance_path=files.provenance_path,
        designation_manifest_path=files.manifest_path,
    )


def _empty_script(tiles: int) -> TileScript:
    return TileScript([pad_tile([], [], []) for _ in range(tiles)])


@pytest.mark.parametrize("cls", [DFineORTDetector, DFineTensorRTDetector])
def test_no_constructor_carries_an_image_size(cls):
    assert "imgsz" not in inspect.signature(cls.__init__).parameters


@pytest.mark.parametrize("kind", KINDS)
def test_a_valid_tiling_and_imgsz_are_read_from_the_provenance(kind, tmp_path):
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling={"tile": 1280, "stride": 1024})
    detector = _build(kind, files)
    assert detector._imgsz == 1280
    assert detector._tiling == TilingSpec(tile=1280, stride=1024)


@pytest.mark.parametrize("kind", KINDS)
def test_imgsz_640_without_a_tiling_key_runs_the_resize_path_at_640(kind, tmp_path, monkeypatch):
    files = write_artifact(tmp_path, kind=kind, imgsz=640, class_names=None)
    detector = _build(kind, files)
    assert detector._tiling is None
    image = write_frame(tmp_path / "frame.png", 64, 48)
    ref = ImageRef(image_id="i", filename="frame.png", path=image)
    script = _empty_script(1)
    if kind == "onnx":
        factory = install_ort_stub(monkeypatch, script, 640)
        detector.detect(ref)
        (call,) = factory.calls
        assert call["images_shape"] == (1, 3, 640, 640)
        assert call["orig_target_sizes"] == [[64, 48]]
    else:
        context = install_trt_runtime(detector, script, 640)
        detector.detect(ref)
        assert context.runs == [1]
        assert context.orig_target_sizes == [[64, 48]]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("imgsz", [0, ABSENT, "640", -1, 640.0, True, None])
def test_a_missing_or_unusable_image_size_is_refused_naming_field_and_path(kind, imgsz, tmp_path):
    files = write_artifact(tmp_path, kind=kind, imgsz=imgsz)
    with pytest.raises(ArtifactImageSizeError) as excinfo:
        _build(kind, files)
    message = str(excinfo.value)
    assert "imgsz" in message
    assert str(files.provenance_path) in message
    if imgsz is not ABSENT:
        assert repr(imgsz) in message


@pytest.mark.parametrize("kind", KINDS)
def test_a_tiled_artifact_sends_tile_tensors_at_the_recorded_size(kind, tmp_path, monkeypatch):
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling={"tile": 1280, "stride": 1024})
    detector = _build(kind, files)
    image = write_frame(tmp_path / "frame.png", 1400, 1300)
    ref = ImageRef(image_id="i", filename="frame.png", path=image)
    script = _empty_script(4)
    if kind == "onnx":
        factory = install_ort_stub(monkeypatch, script, 1280)
        detector.detect(ref)
        (call,) = factory.calls
        assert call["images_shape"] == (4, 3, 1280, 1280)
        assert call["orig_target_sizes"] == [[1280, 1280]] * 4
    else:
        context = install_trt_runtime(detector, script, 1280, batch_profile=(1, 4, 4))
        detector.detect(ref)
        assert context.runs == [4]
        assert ("images", (4, 3, 1280, 1280)) in context.shapes_set
        assert context.orig_target_sizes == [[1280, 1280]] * 4


MALFORMED_TILING = [
    None,
    {},
    "1280",
    {"tile": 0, "stride": 1},
    {"tile": 1280},
    {"tile": 1280, "stride": 2000},
    {"tile": 1280, "stride": 0},
    {"tile": 1280, "stride": -1},
    {"tile": -1280, "stride": 1024},
    {"stride": 1024},
    {"tile": 1280.0, "stride": 1024},
    {"tile": "1280", "stride": "1024"},
    {"tile": True, "stride": True},
    {"tile": 1280, "stride": True},
    {"tile": 1280, "stride": 1024.0},
    {"tile": 1280, "stride": "1024"},
    [1280, 1024],
    0,
    False,
    [],
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", MALFORMED_TILING, ids=repr)
def test_a_malformed_tiling_value_is_refused_quoting_it_and_never_read_as_full_frame(
    kind, value, tmp_path, monkeypatch
):
    files = write_artifact(tmp_path, kind=kind, imgsz=1280, tiling=value)
    script = _empty_script(12)
    factory = install_ort_stub(monkeypatch, script, 1280)
    with pytest.raises(ArtifactTilingError) as excinfo:
        _build(kind, files)
    message = str(excinfo.value)
    assert repr(value) in message
    assert str(files.provenance_path) in message
    assert "tiling" in message
    assert factory.sessions == []
    assert script.cursor == 0


@pytest.mark.parametrize("kind", KINDS)
def test_the_tile_is_the_model_input_so_a_different_imgsz_is_refused(kind, tmp_path):
    files = write_artifact(tmp_path, kind=kind, imgsz=640, tiling={"tile": 1280, "stride": 1024})
    with pytest.raises(ArtifactTilingError) as excinfo:
        _build(kind, files)
    message = str(excinfo.value)
    assert repr({"tile": 1280, "stride": 1024}) in message
    assert "'imgsz' 640" in message
    assert str(files.provenance_path) in message


@pytest.mark.parametrize("kind", KINDS)
def test_the_imgsz_refusal_comes_before_the_tiling_check(kind, tmp_path):
    files = write_artifact(tmp_path, kind=kind, imgsz=0, tiling=None)
    with pytest.raises(ArtifactImageSizeError):
        _build(kind, files)


def test_the_registry_factories_pass_nothing_extra(tmp_path, monkeypatch):
    import app.config as config_module
    from app.config import Settings
    from app.detectors.registry import get_detector

    files = write_artifact(tmp_path, kind="onnx", imgsz=1280, tiling={"tile": 1280, "stride": 1024})
    settings = Settings(
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    detector = get_detector("ort")
    assert (detector._imgsz, detector._tiling) == (1280, TilingSpec(1280, 1024))

    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    engine = write_artifact(engine_dir, kind="engine", imgsz=640)
    settings = Settings(
        INFERENCE_ENGINE_PATH=str(engine.path),
        INFERENCE_ENGINE_PROVENANCE_PATH=str(engine.provenance_path),
        INFERENCE_EXECUTION_TARGET="jetson",
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    detector = get_detector("tensorrt")
    assert (detector._imgsz, detector._tiling) == (640, None)
