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
Unit tests for the ONNX Runtime D-FINE executor. No real ONNX model or GPU is used — the fail-closed provider
check runs before the model file is ever touched, and construction-level
provenance/designation mechanics use fixture files, exactly like the
fixture-only Jetson executor tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

from app.detectors.base import ImageRef
from app.detectors.designation import DesignationManifestError
from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES
from app.detectors.dfine_ort import (
    ArtifactProvenanceError,
    ArtifactTargetMismatchError,
    ClassMapResolutionError,
    DFineORTDetector,
    ORTProviderUnavailableError,
    ORTUnsupportedExecutionTargetError,
)


def _fixture_class_map_hash(class_names) -> str:
    """Independent re-computation of the canonical class-map hash, for
    building fixtures -- deliberately NOT calling into
    ``app.detectors.dfine_common``'s own hash function: a fixture built
    from the SUT's own hash function couldn't catch a bug in that
    function, since both sides would agree by construction."""
    return hashlib.sha256("\n".join(class_names).encode("utf-8")).hexdigest()


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


def _write_onnx_fixture(
    tmp_path: Path,
    onnx_bytes: bytes,
    execution_target: str = "cloud",
    designation_status: str = "integration-only",
    class_names: Optional[List[str]] = None,
    class_map_hash: Optional[str] = None,
    include_class_map_hash: bool = True,
) -> Tuple[Path, Path, Path]:
    model_path = tmp_path / "dfine-l.onnx"
    model_path.write_bytes(onnx_bytes)
    sha256 = hashlib.sha256(onnx_bytes).hexdigest()

    provenance = {
        "artifact_kind": "onnx",
        "execution_target": execution_target,
        "file_sha256": sha256,
        "file_path": f"onnx/{model_path.name}",
        "imgsz": 640,
        "pt_provenance": {
            "model_name": "dfine-l",
            "source": "dfine-coco-stock",
            "license": "Apache-2.0",
            "file_sha256": "ptsha000",
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
    provenance_path = tmp_path / f"{model_path.name}.provenance.json"
    provenance_path.write_text(json.dumps(provenance))

    manifest = {
        "schema_version": "1.0.0",
        "targets": {execution_target: {"artifacts": {sha256: {"designation_status": designation_status}}}},
    }
    manifest_path = tmp_path / "designation-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    return model_path, provenance_path, manifest_path


def _write_image_fixture(tmp_path: Path, width: int = 20, height: int = 10) -> Path:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(image_path)
    return image_path


class _FakeOutputMeta:
    """Named fake for one ``onnxruntime.NodeArg`` as returned by
    ``InferenceSession.get_outputs`` — only the ``.name`` attribute
    ``_ensure_session``/``detect`` actually reads."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeInferenceSession:
    """Named fake for ``onnxruntime.InferenceSession``.

    Models the documented contract of ``get_providers`` (ONNX Runtime
    Python API reference,
    https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession.get_providers):
    it returns the session's active execution providers in priority order,
    the effective one first. Per the execution-providers guide
    (https://onnxruntime.ai/docs/execution-providers/), a provider that
    fails to initialize (e.g. a missing CUDA shared library) is dropped
    from that list with a logged warning — construction itself does not
    raise, and the session silently ends up running on whatever provider
    IS left. This fake reproduces exactly that: the ``reported_providers``
    it returns from ``get_providers`` is configured independently of the
    ``providers`` argument the constructor received, for example CUDA advertised as available while a CUDA library such as
    libcurand is missing.
    """

    def __init__(
        self,
        model_path: str,
        providers: Sequence[str],
        *,
        reported_providers: Sequence[str],
        output_metas: Optional[Sequence[str]] = None,
        outputs_by_name: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.model_path = model_path
        self.requested_providers = list(providers)
        self._reported_providers = list(reported_providers)
        self._output_metas = [_FakeOutputMeta(name) for name in (output_metas or [])]
        self._outputs_by_name = dict(outputs_by_name or {})
        self.run_calls: List[Any] = []

    def get_providers(self) -> List[str]:
        return list(self._reported_providers)

    def get_outputs(self) -> List[_FakeOutputMeta]:
        return list(self._output_metas)

    def run(self, output_names, feed):
        self.run_calls.append((list(output_names), dict(feed)))
        return [self._outputs_by_name[name] for name in output_names]


class _FakeInferenceSessionFactory:
    """Callable stand-in for the ``onnxruntime.InferenceSession`` class
    itself: records the ``(model_path, providers)`` args ``_ensure_session``
    actually passed, and returns a ``FakeInferenceSession`` reporting this
    scenario's configured provider list regardless of what was requested."""

    def __init__(
        self,
        reported_providers: Sequence[str],
        output_metas: Optional[Sequence[str]] = None,
        outputs_by_name: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.reported_providers = list(reported_providers)
        self.output_metas = output_metas
        self.outputs_by_name = outputs_by_name
        self.calls: List[Any] = []
        self.sessions: List[FakeInferenceSession] = []

    def __call__(self, model_path, providers):
        self.calls.append((model_path, list(providers)))
        session = FakeInferenceSession(
            model_path,
            providers,
            reported_providers=self.reported_providers,
            output_metas=self.output_metas,
            outputs_by_name=self.outputs_by_name,
        )
        self.sessions.append(session)
        return session


class TestFailClosedProviderSelection:
    """A cloud/CUDA request whose provider is
    unavailable at construction raises, never silently builds a CPU
    session."""

    def test_cuda_unavailable_raises_before_touching_model_file(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        with pytest.raises(ORTProviderUnavailableError, match="CUDAExecutionProvider"):
            DFineORTDetector(
                model_path=tmp_path / "does-not-exist.onnx",
                execution_target="cloud",
                provider="CUDAExecutionProvider",
            )

    def test_cpu_provider_always_available_on_dev_host(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"fixture-onnx-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        assert detector.impl_id == "dfine-ort-detector"
        assert detector._provider == "CPUExecutionProvider"

    def test_cuda_available_constructs_successfully(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"fixture-cuda-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CUDAExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        assert detector._provider == "CUDAExecutionProvider"


class TestRuntimeMatrixEnforcement:
    """The runtime matrix reserves ONNX Runtime for dev/CI and cloud
    only — Jetson always runs the TensorRT engine. Construction refuses an
    execution_target="jetson" request unconditionally, before the provider
    check even runs (proven by configuring an available provider — the
    refusal must not be an artifact of an unrelated provider failure)."""

    def test_jetson_execution_target_refused_before_provider_check(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            FakeAvailableProviders(["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )

        with pytest.raises(ORTUnsupportedExecutionTargetError, match="jetson"):
            DFineORTDetector(
                model_path=tmp_path / "does-not-exist.onnx",
                execution_target="jetson",
                provider="CUDAExecutionProvider",
            )

    def test_cloud_execution_target_still_constructs(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", FakeAvailableProviders(["CPUExecutionProvider"])
        )
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"fixture-runtime-matrix-cloud", execution_target="cloud"
        )

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        assert detector.execution_target == "cloud"


class TestProvenanceAndDesignation:
    def test_tampered_onnx_bytes_raise(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"original-bytes")
        model_path.write_bytes(b"tampered-bytes-not-described-by-manifest")

        with pytest.raises(ArtifactProvenanceError, match="sha256 mismatch"):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )

    def test_execution_target_mismatch_raises(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"jetson-onnx-bytes", execution_target="jetson"
        )

        with pytest.raises(ArtifactTargetMismatchError):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )

    def test_designation_status_resolved_from_manifest(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"designated-bytes", designation_status="designated"
        )

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        assert detector.designation_status == "designated"

    def test_artifact_absent_from_designation_manifest_fails_loudly(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"some-bytes")
        manifest_path.write_text(json.dumps({
            "targets": {"cloud": {"artifacts": {"0" * 64: {"designation_status": "integration-only"}}}}
        }))

        with pytest.raises(DesignationManifestError):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )

    def test_contract_provenance_fields_have_no_engine_subfields(self, tmp_path, monkeypatch):
        """ONNX-terminal chain: engine_artifact/engine_sha256 are None."""
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"onnx-terminal-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        fields = detector.contract_provenance_fields()

        assert fields["conversion_chain"]["engine_artifact"] is None
        assert fields["conversion_chain"]["engine_sha256"] is None
        assert fields["conversion_chain"]["onnx_sha256"] == detector._verified_sha256

    def test_contract_provenance_fields_valid_against_contract_schema(self, tmp_path, monkeypatch):
        """An ONNX-terminal record (no engine fields) validates
        against the real DetectorProvenance/ConversionChainProvenance
        schema, not just this module's own dict shape."""
        import onnxruntime as ort

        from app.contract import ConversionChainProvenance, DetectorProvenance

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"schema-check-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        fields = detector.contract_provenance_fields()

        prov = DetectorProvenance(
            detector_impl_id=detector.impl_id,
            detector_version=detector.version,
            designation_status=detector.designation_status,
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
            **fields,
        )
        assert isinstance(prov.conversion_chain, ConversionChainProvenance)
        assert prov.conversion_chain.engine_artifact is None
        assert prov.conversion_chain.engine_sha256 is None

    def test_contract_provenance_fields_names_the_configured_provider(self, tmp_path, monkeypatch):
        """``contract_provenance_fields``
        names the provider this detector was constructed with (CPU here) —
        the field DetectorProvenance.execution_provider (contract_version
        1.4.0) records, and the value stays distinguishable from a
        CUDA-served session."""
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"cpu-provider-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        fields = detector.contract_provenance_fields()

        assert fields["execution_provider"] == "CPUExecutionProvider"

    def test_version_derived_from_verified_sha256(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        onnx_bytes = b"version-check-bytes"
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, onnx_bytes)

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        assert detector.version == hashlib.sha256(onnx_bytes).hexdigest()[:12]


class TestConstructedSessionVerification:
    """``_ensure_session`` must
    verify the SESSION IT ACTUALLY CONSTRUCTED, not just the pre-
    construction ``get_available_providers`` advertisement checked by
    ``_ensure_provider_available`` above — a session can silently fall back to CPU even when CUDA is
    advertised as available."""

    CLASS_NAMES = ["aphid", "leaf"]

    def _raw_outputs(self):
        return {
            "labels": np.array([[0]], dtype=np.int64),
            "boxes": np.array([[[2, 2, 10, 8]]], dtype=np.float32),
            "scores": np.array([[0.77]], dtype=np.float32),
        }

    def test_session_falls_back_to_cpu_refuses_before_any_inference(self, tmp_path, monkeypatch):
        """Construction
        succeeds (CUDA is advertised available) but the constructed
        session reports only CPUExecutionProvider — the detector must
        refuse, naming both providers, and never call session.run."""
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        factory = _FakeInferenceSessionFactory(reported_providers=["CPUExecutionProvider"])
        monkeypatch.setattr(ort, "InferenceSession", factory)

        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"cuda-configured-bytes")
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CUDAExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )

        with pytest.raises(ORTProviderUnavailableError, match="CUDAExecutionProvider") as excinfo:
            detector._ensure_session()
        assert "CPUExecutionProvider" in str(excinfo.value)

        assert factory.sessions[0].run_calls == []
        assert detector._session is None

    def test_requested_provider_first_proceeds_and_detects(self, tmp_path, monkeypatch):
        """The constructed
        session reports the requested provider first — construction
        proceeds and detect returns canonical detections end to end."""
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        factory = _FakeInferenceSessionFactory(
            reported_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            output_metas=["labels", "boxes", "scores"],
            outputs_by_name=self._raw_outputs(),
        )
        monkeypatch.setattr(ort, "InferenceSession", factory)

        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"cuda-healthy-bytes", class_names=self.CLASS_NAMES
        )
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CUDAExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        image_path = _write_image_fixture(tmp_path, width=20, height=10)

        result = detector.detect(ImageRef(image_id="img-1", filename="sample.png", path=image_path))

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.label == "aphid"
        assert det.score == pytest.approx(float(np.float32(0.77)))
        assert det.x_min == pytest.approx(2 / 20)
        assert det.y_min == pytest.approx(2 / 10)
        assert det.x_max == pytest.approx(10 / 20)
        assert det.y_max == pytest.approx(8 / 10)
        assert len(factory.calls) == 1

    def test_cpu_configured_detector_reporting_cpu_first_passes_same_rule(self, tmp_path, monkeypatch):
        """For dev/CI: a CPU-configured detector is not a
        special case — when the session reports CPUExecutionProvider
        first, the same first-position rule passes it."""
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        factory = _FakeInferenceSessionFactory(
            reported_providers=["CPUExecutionProvider"],
            output_metas=["labels", "boxes", "scores"],
            outputs_by_name=self._raw_outputs(),
        )
        monkeypatch.setattr(ort, "InferenceSession", factory)

        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"cpu-configured-bytes")
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )

        session = detector._ensure_session()

        assert session is factory.sessions[0]
        assert detector._ensure_session() is session
        assert len(factory.calls) == 1

    def test_probe_runs_the_session_on_a_zero_input_of_the_imgsz(self, tmp_path, monkeypatch):
        """The health probe: one ``session.run`` on the two named inputs the
        detector consumes, zeros at the artifact's imgsz (640 in the
        fixture provenance)."""
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        factory = _FakeInferenceSessionFactory(
            reported_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            output_metas=["labels", "boxes", "scores"],
            outputs_by_name=self._raw_outputs(),
        )
        monkeypatch.setattr(ort, "InferenceSession", factory)
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"cuda-probe-bytes")
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CUDAExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )

        detector.probe()

        (call,) = factory.sessions[0].run_calls
        _, feed = call
        assert feed["images"].shape == (1, 3, 640, 640)
        assert feed["images"].dtype == np.float32
        assert not feed["images"].any()
        assert feed["orig_target_sizes"].shape == (1, 2)
        assert feed["orig_target_sizes"].dtype == np.int64
        assert (feed["orig_target_sizes"] > 0).all()

    def test_probe_refuses_a_session_on_another_provider(self, tmp_path, monkeypatch):
        """The probe goes through the session check ``detect`` uses: a
        session that runs on CPU while CUDA was requested fails it, and no
        inference is run."""
        import onnxruntime as ort

        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        factory = _FakeInferenceSessionFactory(reported_providers=["CPUExecutionProvider"])
        monkeypatch.setattr(ort, "InferenceSession", factory)
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"cuda-lost-bytes")
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CUDAExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )

        with pytest.raises(ORTProviderUnavailableError, match="CUDAExecutionProvider"):
            detector.probe()
        assert factory.sessions[0].run_calls == []


class TestClassMapResolution:
    """Class-map resolution: one construction-time rule, both directions fail-closed, for the ONNX
    Runtime executor specifically."""

    def _raw_outputs(self):
        return {
            "labels": np.array([[0]], dtype=np.int64),
            "boxes": np.array([[[2, 2, 10, 8]]], dtype=np.float32),
            "scores": np.array([[0.77]], dtype=np.float32),
        }

    def test_aphid_record_labels_index_0_aphid(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        factory = _FakeInferenceSessionFactory(
            reported_providers=["CPUExecutionProvider"],
            output_metas=["labels", "boxes", "scores"],
            outputs_by_name=self._raw_outputs(),
        )
        monkeypatch.setattr(ort, "InferenceSession", factory)

        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"aphid-record-bytes", class_names=["aphid"]
        )
        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )
        image_path = _write_image_fixture(tmp_path, width=20, height=10)

        result = detector.detect(ImageRef(image_id="img-1", filename="sample.png", path=image_path))

        assert result.detections[0].label == "aphid"

    def test_record_without_class_names_serves_stock_unchanged(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(tmp_path, b"stock-record-bytes")

        detector = DFineORTDetector(
            model_path=model_path,
            execution_target="cloud",
            provider="CPUExecutionProvider",
            provenance_path=provenance_path,
            designation_manifest_path=manifest_path,
        )

        assert detector._class_names == DFINE_STOCK_COCO_CLASS_NAMES

    def test_class_names_hash_disagreement_refuses_naming_both_hashes(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        wrong_hash = "f" * 64
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"mismatched-aphid-bytes", class_names=["aphid"], class_map_hash=wrong_hash
        )

        with pytest.raises(ClassMapResolutionError) as excinfo:
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )
        assert wrong_hash in str(excinfo.value)
        assert _fixture_class_map_hash(["aphid"]) in str(excinfo.value)

    def test_stock_fallback_hash_drift_refuses_naming_both_hashes(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        drifted_hash = "e" * 64
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"drifted-stock-bytes", class_map_hash=drifted_hash
        )

        with pytest.raises(ClassMapResolutionError) as excinfo:
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )
        assert drifted_hash in str(excinfo.value)
        assert _fixture_class_map_hash(DFINE_STOCK_COCO_CLASS_NAMES) in str(excinfo.value)

    def test_record_with_no_class_map_hash_refuses_by_name(self, tmp_path, monkeypatch):
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"no-hash-bytes", include_class_map_hash=False
        )

        with pytest.raises(ClassMapResolutionError, match="class_map_hash"):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )

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
    def test_present_but_invalid_class_names_refuses_never_falls_back(
        self, tmp_path, monkeypatch, invalid_class_names
    ):
        """Absent and invalid are distinguished by key presence and
        list-typing, never truthiness-collapsed: a PRESENT-but-malformed ``class_names`` must refuse construction
        outright, never silently serve the stock list as if the key were
        absent."""
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"invalid-class-names-bytes", class_names=invalid_class_names
        )

        with pytest.raises(ClassMapResolutionError):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )

    def test_present_null_class_names_refuses_never_falls_back(self, tmp_path, monkeypatch):
        """A present `class_names: null` is an invalid shape, not an absent field --
        key presence decides, never value truthiness. The fixture
        helper's own `class_names=None` means "omit the key" (the absent-field convention of the other tests), so this test patches the written
        record directly to get a key that is PRESENT with a null value."""
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"null-class-names-bytes"
        )
        record = json.loads(provenance_path.read_text())
        record["class_names"] = None
        provenance_path.write_text(json.dumps(record))

        with pytest.raises(ClassMapResolutionError):
            DFineORTDetector(
                model_path=model_path,
                execution_target="cloud",
                provider="CPUExecutionProvider",
                provenance_path=provenance_path,
                designation_manifest_path=manifest_path,
            )
