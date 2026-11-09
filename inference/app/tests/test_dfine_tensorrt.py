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
Unit tests for the Jetson D-FINE TensorRT executor.

Runs entirely without a GPU or the tensorrt/pycuda packages:
construction/provenance/designation mechanics use fixture files; the by-name tensor binding proof
(``_run_engine``) uses a fake trt/cuda/engine/context harness — defined
below — that faithfully mimics the real pycuda/tensorrt device-memory API
shape (mem_alloc/set_tensor_address/memcpy_*), deliberately declaring the
engine's five I/O tensors in a SCRAMBLED order, so a positional-binding
bug would fail this test even though it might pass on a naturally-ordered
engine. Only ``detect``'s real runtime call remains untestable here —
proving that path raises a clear, distinguishable error is what this
environment can test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

from app.detectors.base import ImageRef
from app.detectors.designation import DesignationManifestError
from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES
from app.detectors.dfine_tensorrt import (
    ArtifactProvenanceError,
    ArtifactTargetMismatchError,
    ClassMapResolutionError,
    DFineEngineBindingError,
    DFineEngineExecutionError,
    DFineTensorRTDetector,
    TensorRTRuntimeUnavailable,
    provenance_path_for,
)


def _fixture_class_map_hash(class_names) -> str:
    """Independent re-computation of the canonical class-map hash, for
    building fixtures -- deliberately NOT calling into
    ``app.detectors.dfine_common``'s own hash function: a fixture built
    from the SUT's own hash function couldn't catch a bug in that
    function, since both sides would agree by construction."""
    return hashlib.sha256("\n".join(class_names).encode("utf-8")).hexdigest()


class _FakeTensorIOMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class _FakeTrtModule:
    TensorIOMode = _FakeTensorIOMode

    @staticmethod
    def nptype(dtype):
        return dtype


class _FakeDeviceMem:
    """Stands in for a pycuda.driver.DeviceAllocation: production code
    only ever uses the object it got back from mem_alloc (never re-derives
    it from the address later), except for the address int handed to
    set_tensor_address — so this class registers itself by id for that
    one indirection."""

    _registry: Dict[int, "_FakeDeviceMem"] = {}

    def __init__(self, nbytes: int):
        self.buffer = bytearray(nbytes)
        _FakeDeviceMem._registry[id(self)] = self

    def __int__(self) -> int:
        return id(self)


class _FakeStream:
    handle = 0xF00D

    def synchronize(self) -> None:
        pass


class _FakeCudaModule:
    def mem_alloc(self, nbytes: int) -> _FakeDeviceMem:
        return _FakeDeviceMem(nbytes)

    def memcpy_htod_async(self, dst: _FakeDeviceMem, src: np.ndarray, stream) -> None:
        arr = np.ascontiguousarray(src)
        dst.buffer[: arr.nbytes] = arr.tobytes()

    def memcpy_dtoh_async(self, dst: np.ndarray, src: _FakeDeviceMem, stream) -> None:
        flat = np.frombuffer(bytes(src.buffer[: dst.nbytes]), dtype=dst.dtype)
        dst[...] = flat.reshape(dst.shape)

    def Stream(self) -> _FakeStream:
        return _FakeStream()


class _FakeEngine:
    """Declares its I/O tensors in a DELIBERATELY scrambled order — two
    inputs and three outputs interleaved, not grouped or ordered
    naturally — so the production code under test must be looking each
    one up by name, never by ordinal binding index."""

    def __init__(self, tensors: List[Tuple[str, str, Any, tuple]]):
        self._tensors = tensors

    @property
    def num_io_tensors(self) -> int:
        return len(self._tensors)

    def get_tensor_name(self, i: int) -> str:
        return self._tensors[i][0]

    def get_tensor_mode(self, name: str) -> str:
        return next(mode for n, mode, _, _ in self._tensors if n == name)

    def get_tensor_dtype(self, name: str):
        return next(dtype for n, _, dtype, _ in self._tensors if n == name)

    def get_shape(self, name: str) -> tuple:
        return next(shape for n, _, _, shape in self._tensors if n == name)


class _FakeContext:
    """Dynamic-shape-faithful fake: like a real execution context for an
    engine built from our dynamic-batch ONNX export (an ONNX export with a dynamic batch axis), every OUTPUT shape reports a ``-1`` batch dim until
    ``set_input_shape`` has pinned every input's runtime shape — so the
    production code only sees concrete output shapes if it sets input
    shapes first. ``execute_async_v3`` returns the real API's success
    boolean."""

    def __init__(self, engine: _FakeEngine, outputs: Dict[str, np.ndarray], execute_result: bool = True):
        self._engine = engine
        self._outputs = outputs
        self._execute_result = execute_result
        self._mem_by_name: Dict[str, _FakeDeviceMem] = {}
        self.input_names_set: List[str] = []
        self.input_shapes_set: Dict[str, tuple] = {}

    def _num_engine_inputs(self) -> int:
        return sum(
            1 for name, mode, _, _ in self._engine._tensors if mode == _FakeTensorIOMode.INPUT
        )

    def set_input_shape(self, name: str, shape) -> None:
        assert self._engine.get_tensor_mode(name) == _FakeTensorIOMode.INPUT, name
        self.input_shapes_set[name] = tuple(shape)

    def get_tensor_shape(self, name: str) -> tuple:
        shape = self._engine.get_shape(name)
        if (
            self._engine.get_tensor_mode(name) != _FakeTensorIOMode.INPUT
            and len(self.input_shapes_set) < self._num_engine_inputs()
        ):
            return (-1,) + tuple(shape[1:])
        return shape

    def set_tensor_address(self, name: str, addr: int) -> None:
        self._mem_by_name[name] = _FakeDeviceMem._registry[addr]
        if self._engine.get_tensor_mode(name) == _FakeTensorIOMode.INPUT:
            self.input_names_set.append(name)

    def execute_async_v3(self, stream_handle) -> bool:
        if not self._execute_result:
            return False
        for name, array in self._outputs.items():
            mem = self._mem_by_name[name]
            arr = np.ascontiguousarray(array)
            mem.buffer[: arr.nbytes] = arr.tobytes()
        return True


def _scrambled_engine_and_context(
    labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray
) -> Tuple[_FakeEngine, _FakeContext]:
    n = labels.shape[1]
    tensors = [
        ("orig_target_sizes", _FakeTensorIOMode.INPUT, np.int64, (1, 2)),
        ("boxes", _FakeTensorIOMode.OUTPUT, np.float32, (1, n, 4)),
        ("images", _FakeTensorIOMode.INPUT, np.float32, (1, 3, 640, 640)),
        ("labels", _FakeTensorIOMode.OUTPUT, np.int64, (1, n)),
        ("scores", _FakeTensorIOMode.OUTPUT, np.float32, (1, n)),
    ]
    engine = _FakeEngine(tensors)
    context = _FakeContext(engine, outputs={"labels": labels, "boxes": boxes, "scores": scores})
    return engine, context


def _write_engine_fixture(
    tmp_path: Path,
    engine_bytes: bytes,
    execution_target: str = "jetson",
    pt_sha: str = "ptsha000",
    engine_sha: Optional[str] = None,
    designation_status: str = "integration-only",
    class_names: Optional[List[str]] = None,
    class_map_hash: Optional[str] = None,
    include_class_map_hash: bool = True,
) -> Tuple[Path, Path, Path]:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(engine_bytes)

    declared_sha256 = engine_sha if engine_sha is not None else hashlib.sha256(engine_bytes).hexdigest()
    provenance = {
        "artifact_kind": "engine",
        "execution_target": execution_target,
        "source_onnx_sha256": "onnxsha000",
        "source_onnx_provenance_file": "provenance/dfine-s-jetson.onnx.json",
        "file_sha256": declared_sha256,
        "file_path": f"engines/{engine_path.name}",
        "imgsz": 640,
        "pt_provenance": {
            "model_name": "dfine-s",
            "source": "dfine-coco-stock",
            "license": "Apache-2.0",
            "file_sha256": pt_sha,
        },
    }
    if include_class_map_hash:
        provenance["class_map_hash"] = (
            class_map_hash
            if class_map_hash is not None
            else _fixture_class_map_hash(class_names if class_names is not None else DFINE_STOCK_COCO_CLASS_NAMES)
        )
    if class_names is not None:
        provenance["class_names"] = class_names
    provenance_path = provenance_path_for(engine_path)
    provenance_path.write_text(json.dumps(provenance))

    manifest = {
        "schema_version": "1.0.0",
        "targets": {execution_target: {"artifacts": {declared_sha256: {"designation_status": designation_status}}}},
    }
    manifest_path = provenance_path.parent / "designation-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    return engine_path, provenance_path, manifest_path


class TestDFineTensorRTDetectorConstruction:
    def test_missing_engine_file_raises(self, tmp_path):
        with pytest.raises(ArtifactProvenanceError):
            DFineTensorRTDetector(engine_path=tmp_path / "does_not_exist.engine", execution_target="jetson")

    def test_missing_provenance_manifest_raises(self, tmp_path):
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"not-a-real-engine")
        with pytest.raises(ArtifactProvenanceError):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_constructs_with_fixture_engine_and_provenance(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"fixture-engine-bytes-v1")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert detector.impl_id == "dfine-tensorrt-detector"
        assert detector.version
        assert detector.designation_status == "integration-only"

    def test_tampered_engine_bytes_raise(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"original-engine-bytes")
        engine_path.write_bytes(b"tampered-bytes-the-manifest-does-not-describe")

        with pytest.raises(ArtifactProvenanceError, match="sha256 mismatch"):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_execution_target_mismatch_raises(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"cloud-built-bytes", execution_target="cloud")
        with pytest.raises(ArtifactTargetMismatchError):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_designation_status_designated_resolved_from_manifest(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(
            tmp_path, b"designated-engine-bytes", designation_status="designated"
        )
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert detector.designation_status == "designated"

    def test_artifact_absent_from_designation_manifest_fails_loudly(self, tmp_path):
        engine_path, _, manifest_path = _write_engine_fixture(tmp_path, b"some-engine-bytes")
        manifest_path.write_text(json.dumps({
            "targets": {"jetson": {"artifacts": {"0" * 64: {"designation_status": "integration-only"}}}}
        }))
        with pytest.raises(DesignationManifestError):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_version_derived_from_engine_sha256(self, tmp_path):
        engine_bytes = b"version-check-engine-bytes"
        engine_path, _, _ = _write_engine_fixture(tmp_path, engine_bytes)
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert detector.version == hashlib.sha256(engine_bytes).hexdigest()[:12]

    def test_contract_provenance_fields_populate_engine_subfields(self, tmp_path):
        engine_bytes = b"provenance-fields-bytes"
        engine_path, _, _ = _write_engine_fixture(tmp_path, engine_bytes)
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        fields = detector.contract_provenance_fields()
        assert fields["model_version"] == "dfine-s"
        assert fields["model_license"] == "Apache-2.0"
        assert fields["conversion_chain"]["engine_sha256"] == hashlib.sha256(engine_bytes).hexdigest()
        assert fields["conversion_chain"]["execution_target"] == "jetson"


class TestByNameBindingThroughToCanonicalizer:
    def test_sets_both_inputs_and_reads_three_outputs_by_name(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"binding-test-engine")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        labels = np.array([[0, 1, 0]], dtype=np.int64)
        boxes = np.array([[[10, 20, 190, 80], [-20, -10, 50, 40], [150, 90, 250, 150]]], dtype=np.float32)
        scores = np.array([[0.91, 0.12, 0.55]], dtype=np.float32)
        engine, context = _scrambled_engine_and_context(labels, boxes, scores)

        images = np.zeros((1, 3, 640, 640), dtype=np.float32)
        orig_target_sizes = np.array([[200, 100]], dtype=np.int64)

        raw_tensors = detector._run_engine(
            _FakeTrtModule, _FakeCudaModule(), engine, context, images, orig_target_sizes
        )

        assert set(context.input_names_set) == {"images", "orig_target_sizes"}

        raw_names = {t.name for t in raw_tensors}
        assert raw_names == {"labels", "boxes", "scores", "orig_target_sizes"}

        from app.detectors.dfine_common import canonicalize_dfine

        result = canonicalize_dfine(raw_tensors, class_names=["aphid", "leaf"], image_id="img-trt")
        assert len(result.detections) == 3
        det0 = result.detections[0]
        assert det0.label == "aphid"
        assert det0.x_min == pytest.approx(10 / 200)
        assert det0.y_max == pytest.approx(80 / 100)
        det2 = result.detections[2]
        assert det2.x_max == pytest.approx(1.0)
        assert det2.y_max == pytest.approx(1.0)

    def test_engine_missing_a_required_output_raises_binding_error(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"missing-output-engine")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        tensors = [
            ("images", _FakeTensorIOMode.INPUT, np.float32, (1, 3, 640, 640)),
            ("orig_target_sizes", _FakeTensorIOMode.INPUT, np.int64, (1, 2)),
            ("labels", _FakeTensorIOMode.OUTPUT, np.int64, (1, 1)),
            ("scores", _FakeTensorIOMode.OUTPUT, np.float32, (1, 1)),
        ]
        engine = _FakeEngine(tensors)
        context = _FakeContext(
            engine, outputs={"labels": np.array([[0]]), "scores": np.array([[0.5]], dtype=np.float32)}
        )

        with pytest.raises(DFineEngineBindingError, match="boxes"):
            detector._run_engine(
                _FakeTrtModule,
                _FakeCudaModule(),
                engine,
                context,
                np.zeros((1, 3, 640, 640), dtype=np.float32),
                np.array([[10, 10]], dtype=np.int64),
            )

    def test_engine_with_unexpected_input_name_raises_binding_error(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"unexpected-input-engine")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        tensors = [
            ("pixel_values", _FakeTensorIOMode.INPUT, np.float32, (1, 3, 640, 640)),
            ("orig_target_sizes", _FakeTensorIOMode.INPUT, np.int64, (1, 2)),
            ("labels", _FakeTensorIOMode.OUTPUT, np.int64, (1, 1)),
            ("boxes", _FakeTensorIOMode.OUTPUT, np.float32, (1, 1, 4)),
            ("scores", _FakeTensorIOMode.OUTPUT, np.float32, (1, 1)),
        ]
        engine = _FakeEngine(tensors)
        context = _FakeContext(engine, outputs={})

        with pytest.raises(DFineEngineBindingError, match="pixel_values"):
            detector._run_engine(
                _FakeTrtModule,
                _FakeCudaModule(),
                engine,
                context,
                np.zeros((1, 3, 640, 640), dtype=np.float32),
                np.array([[10, 10]], dtype=np.int64),
            )


class TestDynamicShapeAndExecutionFailure:
    """Dynamic-engine mechanics: input shapes must be pinned before output
    shapes are queried (the fake reports -1 output dims until then, like a
    real dynamic-batch engine), and a failed enqueue must raise, never
    fall through to reading uninitialized buffers."""

    def _detector(self, tmp_path, engine_bytes: bytes) -> DFineTensorRTDetector:
        engine_path, _, _ = _write_engine_fixture(tmp_path, engine_bytes)
        return DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_input_shapes_are_set_before_output_shapes_resolve(self, tmp_path):
        detector = self._detector(tmp_path, b"dynamic-shape-engine")
        labels = np.array([[1]], dtype=np.int64)
        boxes = np.array([[[0, 0, 5, 5]]], dtype=np.float32)
        scores = np.array([[0.7]], dtype=np.float32)
        engine, context = _scrambled_engine_and_context(labels, boxes, scores)

        raw_tensors = detector._run_engine(
            _FakeTrtModule,
            _FakeCudaModule(),
            engine,
            context,
            np.zeros((1, 3, 640, 640), dtype=np.float32),
            np.array([[10, 10]], dtype=np.int64),
        )

        assert context.input_shapes_set == {
            "images": (1, 3, 640, 640),
            "orig_target_sizes": (1, 2),
        }
        assert {t.name for t in raw_tensors} == {"labels", "boxes", "scores", "orig_target_sizes"}

    def test_unresolved_output_shape_raises_execution_error(self, tmp_path):
        detector = self._detector(tmp_path, b"unresolved-shape-engine")
        labels = np.array([[1]], dtype=np.int64)
        boxes = np.array([[[0, 0, 5, 5]]], dtype=np.float32)
        scores = np.array([[0.7]], dtype=np.float32)
        engine, context = _scrambled_engine_and_context(labels, boxes, scores)
        context.set_input_shape = lambda name, shape: None

        with pytest.raises(DFineEngineExecutionError, match="unresolved"):
            detector._run_engine(
                _FakeTrtModule,
                _FakeCudaModule(),
                engine,
                context,
                np.zeros((1, 3, 640, 640), dtype=np.float32),
                np.array([[10, 10]], dtype=np.int64),
            )

    def test_failed_enqueue_raises_execution_error(self, tmp_path):
        detector = self._detector(tmp_path, b"failed-enqueue-engine")
        labels = np.array([[1]], dtype=np.int64)
        boxes = np.array([[[0, 0, 5, 5]]], dtype=np.float32)
        scores = np.array([[0.7]], dtype=np.float32)
        engine, _ = _scrambled_engine_and_context(labels, boxes, scores)
        failing_context = _FakeContext(
            engine,
            outputs={"labels": labels, "boxes": boxes, "scores": scores},
            execute_result=False,
        )

        with pytest.raises(DFineEngineExecutionError, match="execute_async_v3"):
            detector._run_engine(
                _FakeTrtModule,
                _FakeCudaModule(),
                engine,
                failing_context,
                np.zeros((1, 3, 640, 640), dtype=np.float32),
                np.array([[10, 10]], dtype=np.int64),
            )


class TestRuntimeUnavailable:
    """Live engine execution is untestable on this host (no GPU, no
    tensorrt/pycuda installed) — proving detect fails with a clear,
    distinguishable error is what this environment can test."""

    def test_detect_raises_tensorrt_runtime_unavailable(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"fixture-bytes")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        img_path = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(img_path)
        ref = ImageRef(image_id="id1", filename="img.png", path=img_path)

        with pytest.raises(TensorRTRuntimeUnavailable):
            detector.detect(ref)


class TestProbe:
    """The health probe: one engine execution on a zero batch of one,
    on the runtime's owning thread only."""

    def _detector(self, tmp_path) -> DFineTensorRTDetector:
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"probe-engine")
        return DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_probe_from_another_thread_is_refused_before_any_device_call(self, tmp_path):
        import threading

        from app.detectors.dfine_tensorrt import TensorRTThreadAffinityError

        detector = self._detector(tmp_path)
        detector._runtime = ("trt", "cuda", "engine", "context")
        detector._runtime_owner_ident = threading.get_ident() + 12345
        device_calls: List[Any] = []
        detector._run_engine = lambda *a, **k: device_calls.append(a)

        with pytest.raises(TensorRTThreadAffinityError):
            detector.probe()
        assert device_calls == []

    def test_probe_on_the_owning_thread_executes_a_zero_batch_of_one(self, tmp_path):
        import threading

        detector = self._detector(tmp_path)
        detector._runtime = ("trt", "cuda", "engine", "context")
        detector._runtime_owner_ident = threading.get_ident()
        calls: List[Any] = []
        detector._run_engine = lambda trt, cuda, engine, context, images, sizes: calls.append((images, sizes))

        detector.probe()

        ((images, sizes),) = calls
        assert images.shape == (1, 3, detector._imgsz, detector._imgsz)
        assert images.dtype == np.float32
        assert not images.any()
        assert sizes.shape == (1, 2)
        assert sizes.dtype == np.int64

    def test_probe_without_the_runtime_raises_runtime_unavailable(self, tmp_path):
        detector = self._detector(tmp_path)
        with pytest.raises(TensorRTRuntimeUnavailable):
            detector.probe()


class TestClassMapResolution:
    """Class-map resolution: one construction-time rule, both directions fail-closed, for the Jetson
    TensorRT executor specifically. detect itself
    needs a real GPU runtime (TestRuntimeUnavailable, above), so the
    aphid-labeling test exercises the resolved ``_class_names`` through
    the same ``_run_engine`` + ``canonicalize_dfine`` composition
    ``detect`` itself uses."""

    def test_aphid_record_labels_index_0_aphid(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"aphid-engine-bytes", class_names=["aphid"])
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert detector._class_names == ["aphid"]

        labels = np.array([[0]], dtype=np.int64)
        boxes = np.array([[[10, 20, 190, 80]]], dtype=np.float32)
        scores = np.array([[0.91]], dtype=np.float32)
        engine, context = _scrambled_engine_and_context(labels, boxes, scores)
        raw_tensors = detector._run_engine(
            _FakeTrtModule,
            _FakeCudaModule(),
            engine,
            context,
            np.zeros((1, 3, 640, 640), dtype=np.float32),
            np.array([[200, 100]], dtype=np.int64),
        )

        from app.detectors.dfine_common import canonicalize_dfine

        result = canonicalize_dfine(raw_tensors, class_names=detector._class_names, image_id="img-trt")
        assert result.detections[0].label == "aphid"

    def test_record_without_class_names_serves_stock_unchanged(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"stock-engine-bytes")
        detector = DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

        assert detector._class_names == DFINE_STOCK_COCO_CLASS_NAMES

    def test_class_names_hash_disagreement_refuses_naming_both_hashes(self, tmp_path):
        wrong_hash = "f" * 64
        engine_path, _, _ = _write_engine_fixture(
            tmp_path, b"mismatched-aphid-engine", class_names=["aphid"], class_map_hash=wrong_hash
        )

        with pytest.raises(ClassMapResolutionError) as excinfo:
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert wrong_hash in str(excinfo.value)
        assert _fixture_class_map_hash(["aphid"]) in str(excinfo.value)

    def test_stock_fallback_hash_drift_refuses_naming_both_hashes(self, tmp_path):
        drifted_hash = "e" * 64
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"drifted-stock-engine", class_map_hash=drifted_hash)

        with pytest.raises(ClassMapResolutionError) as excinfo:
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
        assert drifted_hash in str(excinfo.value)
        assert _fixture_class_map_hash(DFINE_STOCK_COCO_CLASS_NAMES) in str(excinfo.value)

    def test_record_with_no_class_map_hash_refuses_by_name(self, tmp_path):
        engine_path, _, _ = _write_engine_fixture(tmp_path, b"no-hash-engine", include_class_map_hash=False)

        with pytest.raises(ClassMapResolutionError, match="class_map_hash"):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    @pytest.mark.parametrize(
        "invalid_class_names",
        [
            pytest.param([], id="empty_list"),
            pytest.param([""], id="empty_string_member"),
            pytest.param(["aphid\ncar"], id="newline_bearing_member"),
            pytest.param("aphid", id="bare_string"),
            pytest.param({"aphid": 0}, id="dict"),
        ],
    )
    def test_present_but_invalid_class_names_refuses_never_falls_back(self, tmp_path, invalid_class_names):
        """Absent and invalid are distinguished by key presence and
        list-typing, never truthiness-collapsed: a PRESENT-but-malformed ``class_names`` must refuse construction
        outright, never silently serve the stock list as if the key were
        absent."""
        engine_path, _, _ = _write_engine_fixture(
            tmp_path, b"invalid-class-names-engine", class_names=invalid_class_names
        )

        with pytest.raises(ClassMapResolutionError):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")

    def test_present_null_class_names_refuses_never_falls_back(self, tmp_path):
        """A present `class_names: null` is an invalid shape, not an absent field --
        key presence decides, never value truthiness. The fixture
        helper's own `class_names=None` means "omit the key" (the absent-field convention of the other tests), so this test patches the written
        record directly to get a key that is PRESENT with a null value."""
        engine_path, provenance_path, _ = _write_engine_fixture(tmp_path, b"null-class-names-engine")
        record = json.loads(provenance_path.read_text())
        record["class_names"] = None
        provenance_path.write_text(json.dumps(record))

        with pytest.raises(ClassMapResolutionError):
            DFineTensorRTDetector(engine_path=engine_path, execution_target="jetson")
