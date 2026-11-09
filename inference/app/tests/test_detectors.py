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

"""Unit tests for the pluggable detector interface."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.detectors.base import ImageRef
from app.detectors.lookup import LookupDetectionValidationError, LookupDetector
from app.detectors.null import NullDetector
from app.detectors.registry import get_detector


def _stock_class_map_hash() -> str:
    """Independent re-computation of the stock COCO list's canonical class-
    map hash, for fixture provenance records -- deliberately not calling into
    ``app.detectors.dfine_common``'s own hash function."""
    import hashlib

    from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES

    return hashlib.sha256("\n".join(DFINE_STOCK_COCO_CLASS_NAMES).encode("utf-8")).hexdigest()


class FakeAvailableProviders:
    """Named fake for ``onnxruntime.get_available_providers``.

    Models the documented contract of that call (ONNX Runtime Python API
    reference, https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.get_available_providers,
    and the execution-providers guide,
    https://onnxruntime.ai/): it takes no
    arguments and returns the plain ``list[str]`` of execution providers
    this ONNX Runtime *build* supports on this host — a host/build-time
    property queried independently of any session, not something
    ``InferenceSession(..., providers=[...])`` construction can widen or
    narrow. There is no documented error path for this call, so this fake
    has none — it only ever returns the configured list.
    """

    def __init__(self, providers):
        self._providers = list(providers)

    def __call__(self):
        return list(self._providers)


@pytest.fixture()
def sample_results_path(tmp_path: Path) -> Path:
    data = {
        "known.png": {
            "image_width": 100,
            "image_height": 100,
            "count": 1,
            "boxes": [
                {
                    "x_min": 0.1,
                    "y_min": 0.1,
                    "x_max": 0.2,
                    "y_max": 0.2,
                    "score": 0.77,
                    "label": "brassica_seedling",
                }
            ],
        },
        "clean.png": {"image_width": 100, "image_height": 100, "count": 0, "boxes": []},
    }
    path = tmp_path / "detection_results.json"
    path.write_text(json.dumps(data))
    return path


class TestLookupDetector:
    def test_known_image_returns_its_own_entry(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path)
        ref = ImageRef(image_id="img-abc", filename="known.png", path=Path("/tmp/known.png"))
        result = detector.detect(ref)
        assert result.image_id == "img-abc"
        assert len(result.detections) == 1
        assert result.detections[0].score == 0.77
        assert result.detections[0].label == "brassica_seedling"

    def test_known_image_matched_by_stem(self, sample_results_path):
        """Filenames without a matching extension still resolve by stem."""
        detector = LookupDetector(results_path=sample_results_path)
        ref = ImageRef(image_id="img-x", filename="known.jpg", path=Path("/tmp/known.jpg"))
        result = detector.detect(ref)
        assert len(result.detections) == 1

    def test_same_image_twice_same_result(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path)
        ref = ImageRef(image_id="img-abc", filename="known.png", path=Path("/tmp/known.png"))
        first = detector.detect(ref)
        second = detector.detect(ref)
        assert first == second

    def test_different_images_get_their_own_results(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path)
        known = detector.detect(ImageRef("id1", "known.png", Path("/tmp/known.png")))
        clean = detector.detect(ImageRef("id2", "clean.png", Path("/tmp/clean.png")))
        assert len(known.detections) == 1
        assert len(clean.detections) == 0

    def test_unknown_image_deterministic_empty_fallback(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path)
        ref = ImageRef(image_id="img-unknown", filename="never_prepared.png", path=Path("/tmp/x.png"))
        first = detector.detect(ref)
        second = detector.detect(ref)
        assert first.detections == []
        assert first == second

    def test_missing_results_file_yields_empty_detections(self, tmp_path):
        detector = LookupDetector(results_path=tmp_path / "does_not_exist.json")
        result = detector.detect(ImageRef("id1", "anything.png", Path("/tmp/x.png")))
        assert result.detections == []

    def test_has_impl_id_and_version(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path)
        assert detector.impl_id == "lookup-detector"
        assert detector.version


class TestLookupDetectorFieldValidation:
    """The stub validates score/label rather than silently
    defaulting a malformed value — its sparse/empty contract on a
    missing/unknown image is otherwise unchanged."""

    def test_score_above_one_raises(self, tmp_path):
        import json

        path = tmp_path / "detection_results.json"
        path.write_text(json.dumps({
            "bad.png": {"boxes": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "score": 1.5, "label": "aphid"}]}
        }))
        detector = LookupDetector(results_path=path)
        with pytest.raises(LookupDetectionValidationError):
            detector.detect(ImageRef("id1", "bad.png", Path("/tmp/bad.png")))

    def test_score_below_zero_raises(self, tmp_path):
        import json

        path = tmp_path / "detection_results.json"
        path.write_text(json.dumps({
            "bad.png": {"boxes": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "score": -0.1, "label": "aphid"}]}
        }))
        detector = LookupDetector(results_path=path)
        with pytest.raises(LookupDetectionValidationError):
            detector.detect(ImageRef("id1", "bad.png", Path("/tmp/bad.png")))

    def test_label_not_in_class_manifest_raises(self, tmp_path):
        import json

        path = tmp_path / "detection_results.json"
        path.write_text(json.dumps({
            "bad.png": {"boxes": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "score": 0.5, "label": "dragon"}]}
        }))
        detector = LookupDetector(results_path=path, class_names=["aphid", "leaf"])
        with pytest.raises(LookupDetectionValidationError):
            detector.detect(ImageRef("id1", "bad.png", Path("/tmp/bad.png")))

    def test_label_validation_skipped_without_configured_class_manifest(self, tmp_path):
        """No class manifest configured -> label validation doesn't apply;
        only score bounds are checked, matching this detector's existing
        free-label behaviour."""
        import json

        path = tmp_path / "detection_results.json"
        path.write_text(json.dumps({
            "ok.png": {"boxes": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "score": 0.5, "label": "anything"}]}
        }))
        detector = LookupDetector(results_path=path)
        result = detector.detect(ImageRef("id1", "ok.png", Path("/tmp/ok.png")))
        assert result.detections[0].label == "anything"

    def test_valid_label_in_manifest_passes(self, sample_results_path):
        detector = LookupDetector(results_path=sample_results_path, class_names=["brassica_seedling"])
        result = detector.detect(ImageRef("id1", "known.png", Path("/tmp/known.png")))
        assert len(result.detections) == 1


class TestNullDetector:
    def test_always_empty(self):
        detector = NullDetector()
        result = detector.detect(ImageRef("id1", "any.png", Path("/tmp/x.png")))
        assert result.detections == []

    def test_has_impl_id_and_version(self):
        detector = NullDetector()
        assert detector.impl_id == "null-detector"
        assert detector.version


class TestRegistrySwap:
    """Unit test swaps implementations without changing the contract."""

    def test_swap_lookup_to_null_same_interface(self, sample_results_path, monkeypatch):
        ref = ImageRef("id1", "known.png", Path("/tmp/known.png"))

        lookup = LookupDetector(results_path=sample_results_path)
        lookup_result = lookup.detect(ref)

        null = get_detector("null")
        null_result = null.detect(ref)

        assert lookup_result.image_id == null_result.image_id == "id1"
        assert isinstance(lookup_result.detections, list)
        assert isinstance(null_result.detections, list)
        assert lookup_result.detections != null_result.detections

    def test_unknown_impl_raises(self):
        with pytest.raises(ValueError):
            get_detector("does-not-exist")

    def test_registry_returns_lookup(self):
        assert isinstance(get_detector("lookup"), LookupDetector)

    def test_registry_returns_null(self):
        assert isinstance(get_detector("null"), NullDetector)

    def test_registry_returns_tensorrt_detector_config_pointed(self, tmp_path, monkeypatch):
        """The registry
        constructs the Jetson D-FINE detector from
        settings.engine_path/engine_provenance_path — config-pointed, no
        code change to swap engines."""
        import hashlib
        import json

        from app.config import Settings
        from app.detectors.dfine_tensorrt import DFineTensorRTDetector

        engine_bytes = b"fixture-engine"
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(engine_bytes)
        engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
        provenance_path = tmp_path / "model.engine.provenance.json"
        provenance_path.write_text(json.dumps({
            "execution_target": "cloud",
            "file_sha256": engine_sha256,
            "file_path": "engines/model.engine",
            "imgsz": 640,
            "class_map_hash": _stock_class_map_hash(),
            "pt_provenance": {
                "model_name": "dfine-l",
                "source": "dfine-coco-stock",
                "license": "Apache-2.0",
                "file_sha256": "ptsha",
            },
        }))
        (tmp_path / "designation-manifest.json").write_text(json.dumps({
            "targets": {"cloud": {"artifacts": {engine_sha256: {"designation_status": "integration-only"}}}}
        }))

        fixture_settings = Settings(
            INFERENCE_ENGINE_PATH=str(engine_path),
            INFERENCE_ENGINE_PROVENANCE_PATH=str(provenance_path),
        )
        import app.config as config_module
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = get_detector("tensorrt")
        assert isinstance(detector, DFineTensorRTDetector)
        assert detector.engine_path == engine_path
        assert detector.execution_target == "cloud"

    def test_registry_returns_ort_detector_config_pointed(self, tmp_path, monkeypatch):
        """Registry constructs the shared ONNX Runtime detector
        from settings.onnx_model_path/onnx_provenance_path/
        onnx_execution_provider — config-pointed, same swap-by-file-
        replacement mechanism as the tensorrt factory above."""
        import hashlib
        import json

        from app.config import Settings
        from app.detectors.dfine_ort import DFineORTDetector

        onnx_bytes = b"fixture-onnx"
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(onnx_bytes)
        onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
        provenance_path = tmp_path / "model.onnx.provenance.json"
        provenance_path.write_text(json.dumps({
            "execution_target": "cloud",
            "file_sha256": onnx_sha256,
            "file_path": "onnx/model.onnx",
            "imgsz": 640,
            "class_map_hash": _stock_class_map_hash(),
            "pt_provenance": {
                "model_name": "dfine-l",
                "source": "dfine-coco-stock",
                "license": "Apache-2.0",
                "file_sha256": "ptsha",
            },
        }))
        (tmp_path / "designation-manifest.json").write_text(json.dumps({
            "targets": {"cloud": {"artifacts": {onnx_sha256: {"designation_status": "integration-only"}}}}
        }))

        fixture_settings = Settings(
            INFERENCE_ONNX_MODEL_PATH=str(model_path),
            INFERENCE_ONNX_PROVENANCE_PATH=str(provenance_path),
            INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        )
        import app.config as config_module
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = get_detector("ort")
        assert isinstance(detector, DFineORTDetector)
        assert detector.model_path == model_path
        assert detector.execution_target == "cloud"
        assert detector._provider == "CPUExecutionProvider"

    def test_registry_wires_ort_cuda_provider_fail_closed(self, tmp_path, monkeypatch):
        """A cloud-configured ORT detector
        whose CUDAExecutionProvider is unavailable at construction raises
        via the registry, never silently constructs a CPU-backed session."""
        import onnxruntime as ort

        from app.config import Settings
        from app.detectors.dfine_ort import ORTProviderUnavailableError

        monkeypatch.setattr(ort, "get_available_providers", FakeAvailableProviders(["CPUExecutionProvider"]))

        fixture_settings = Settings(
            INFERENCE_ONNX_MODEL_PATH=str(tmp_path / "does-not-exist.onnx"),
            INFERENCE_EXECUTION_TARGET="cloud",
            INFERENCE_ONNX_EXECUTION_PROVIDER="CUDAExecutionProvider",
        )
        import app.config as config_module
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(ORTProviderUnavailableError):
            get_detector("ort")

    def test_registry_refuses_ort_detector_for_jetson_target(self, tmp_path, monkeypatch):
        """The registry must not permit INFERENCE_DETECTOR_IMPL=ort with
        INFERENCE_EXECUTION_TARGET=jetson: Jetson runs only the TensorRT engine. Configured with an available CUDA provider so the refusal
        proves the runtime-matrix check, not a masked provider failure."""
        import onnxruntime as ort

        from app.config import Settings
        from app.detectors.dfine_ort import ORTUnsupportedExecutionTargetError

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            FakeAvailableProviders(["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )

        fixture_settings = Settings(
            INFERENCE_ONNX_MODEL_PATH=str(tmp_path / "does-not-exist.onnx"),
            INFERENCE_EXECUTION_TARGET="jetson",
            INFERENCE_ONNX_EXECUTION_PROVIDER="CUDAExecutionProvider",
        )
        import app.config as config_module
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(ORTUnsupportedExecutionTargetError, match="jetson"):
            get_detector("ort")

    def test_registry_wires_execution_target_and_refuses_mismatch(self, tmp_path, monkeypatch):
        """The registry passes settings.execution_target
        (INFERENCE_EXECUTION_TARGET) into the DFineTensorRTDetector, which
        refuses an engine built for a different target."""
        import hashlib
        import json

        from app.config import Settings
        from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES
        from app.detectors.dfine_tensorrt import ArtifactTargetMismatchError

        engine_bytes = b"fixture-engine-jetson"
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(engine_bytes)
        provenance_path = tmp_path / "model.engine.provenance.json"
        provenance_path.write_text(json.dumps({
            "execution_target": "jetson",
            "file_sha256": hashlib.sha256(engine_bytes).hexdigest(),
            "file_path": "engines/model.engine",
            "imgsz": 640,
            "class_map_hash": hashlib.sha256(
                "\n".join(DFINE_STOCK_COCO_CLASS_NAMES).encode("utf-8")
            ).hexdigest(),
            "pt_provenance": {
                "model_name": "dfine-s",
                "source": "dfine-coco-stock",
                "license": "Apache-2.0",
                "file_sha256": "ptsha",
            },
        }))

        fixture_settings = Settings(
            INFERENCE_ENGINE_PATH=str(engine_path),
            INFERENCE_ENGINE_PROVENANCE_PATH=str(provenance_path),
        )
        import app.config as config_module
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(ArtifactTargetMismatchError):
            get_detector("tensorrt")
