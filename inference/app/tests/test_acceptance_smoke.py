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
Live invocations of the per-target acceptance-smoke gate: each test below drives
``run_acceptance_smoke`` through fixture ``Settings`` and real on-disk
ONNX/provenance/designation-manifest fixture files, exactly like
``test_detectors.py``'s ``TestRegistrySwap`` config-pointed-construction
tests — the artifact loading, sha256 verification, and designation
resolution that runs underneath are the real production code paths, only
the input files and env-derived ``Settings`` are fixtures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings
from app.detectors.acceptance_smoke import AcceptanceSmokeRefusal, run_acceptance_smoke
from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES
from app.detectors.dfine_ort import DFineORTDetector
from app.detectors.dfine_tensorrt import DFineTensorRTDetector
from app.main import app as fastapi_app

_STOCK_CLASS_MAP_HASH = hashlib.sha256(
    "\n".join(DFINE_STOCK_COCO_CLASS_NAMES).encode("utf-8")
).hexdigest()


def _write_onnx_fixture(
    tmp_path: Path,
    onnx_bytes: bytes,
    designation_status: str,
    declared_sha256: str = "",
    execution_target: str = "cloud",
    score_policy_id: str = "",
) -> Tuple[Path, Path, Path]:
    """Writes a fixture ONNX artifact plus its provenance manifest and a
    designation manifest recording ``designation_status`` for it. Passing
    an explicit ``declared_sha256`` that does NOT match ``onnx_bytes``'s
    real hash reproduces a hash-mismatched artifact. An empty
    (default) ``score_policy_id`` omits the field entirely, matching a
    pre-score_policy_id-bundle provenance record."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(onnx_bytes)
    real_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    provenance_sha256 = declared_sha256 or real_sha256

    provenance = {
        "artifact_kind": "onnx",
        "execution_target": execution_target,
        "file_sha256": provenance_sha256,
        "file_path": f"onnx/{model_path.name}",
        "imgsz": 640,
        "class_map_hash": _STOCK_CLASS_MAP_HASH,
        "pt_provenance": {
            "model_name": "dfine-l",
            "source": "dfine-coco-stock",
            "license": "Apache-2.0",
            "file_sha256": "ptsha000",
        },
    }
    if score_policy_id:
        provenance["score_policy_id"] = score_policy_id
    provenance_path = tmp_path / "model.onnx.provenance.json"
    provenance_path.write_text(json.dumps(provenance))

    manifest_path = tmp_path / "designation-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "targets": {execution_target: {"artifacts": {real_sha256: {"designation_status": designation_status}}}},
    }))

    return model_path, provenance_path, manifest_path


def _fixture_settings(model_path: Path, provenance_path: Path, execution_target: str = "cloud", **overrides) -> Settings:
    return Settings(
        INFERENCE_DETECTOR_IMPL="ort",
        INFERENCE_ONNX_MODEL_PATH=str(model_path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_EXECUTION_TARGET=execution_target,
        **overrides,
    )


def _write_engine_fixture(
    tmp_path: Path,
    engine_bytes: bytes,
    designation_status: str,
    declared_sha256: str = "",
) -> Tuple[Path, Path, Path]:
    """Jetson/TensorRT counterpart of ``_write_onnx_fixture``. ``DFineTensorRTDetector.__init__`` never opens the
    engine through the tensorrt runtime (lazily imported only inside
    ``detect``), so a fixture with arbitrary bytes exercises the same
    real hash-verification/designation-resolution code path as the ONNX
    fixtures above, with no GPU or tensorrt package required."""
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(engine_bytes)
    real_sha256 = hashlib.sha256(engine_bytes).hexdigest()
    provenance_sha256 = declared_sha256 or real_sha256

    provenance_path = tmp_path / "model.engine.provenance.json"
    provenance_path.write_text(json.dumps({
        "artifact_kind": "engine",
        "execution_target": "jetson",
        "file_sha256": provenance_sha256,
        "file_path": f"engines/{engine_path.name}",
        "imgsz": 640,
        "class_map_hash": _STOCK_CLASS_MAP_HASH,
        "pt_provenance": {
            "model_name": "dfine-s",
            "source": "dfine-coco-stock",
            "license": "Apache-2.0",
            "file_sha256": "ptsha000",
        },
    }))

    manifest_path = tmp_path / "designation-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "targets": {"jetson": {"artifacts": {real_sha256: {"designation_status": designation_status}}}},
    }))

    return engine_path, provenance_path, manifest_path


def _fixture_jetson_settings(engine_path: Path, provenance_path: Path, **overrides) -> Settings:
    return Settings(
        INFERENCE_DETECTOR_IMPL="tensorrt",
        INFERENCE_ENGINE_PATH=str(engine_path),
        INFERENCE_ENGINE_PROVENANCE_PATH=str(provenance_path),
        INFERENCE_EXECUTION_TARGET="jetson",
        **overrides,
    )


class TestAcceptanceSmokeRefusals:
    """Three live refusal invocations."""

    def test_case_a_lookup_stub_refuses(self, tmp_path, monkeypatch):
        fixture_settings = Settings(INFERENCE_DETECTOR_IMPL="lookup")
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="stub"):
            run_acceptance_smoke()

    def test_case_b_integration_only_artifact_refuses(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"integration-only-onnx-bytes", designation_status="integration-only"
        )
        fixture_settings = _fixture_settings(model_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="integration-only"):
            run_acceptance_smoke()

    def test_case_c_hash_mismatched_artifact_refuses(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path,
            b"tampered-onnx-bytes",
            designation_status="designated",
            declared_sha256="0" * 64,
        )
        fixture_settings = _fixture_settings(model_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="sha256 mismatch"):
            run_acceptance_smoke()

    def test_artifact_absent_from_designation_manifest_refuses(self, tmp_path, monkeypatch):
        """Not one of the three cases above, but the same refusal
        family — an artifact with valid provenance but no designation
        manifest entry at all must refuse exactly like an integration-only
        one, never silently default to a pass."""
        onnx_bytes = b"unlisted-onnx-bytes"
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(onnx_bytes)
        real_sha256 = hashlib.sha256(onnx_bytes).hexdigest()

        provenance_path = tmp_path / "model.onnx.provenance.json"
        provenance_path.write_text(json.dumps({
            "artifact_kind": "onnx",
            "execution_target": "cloud",
            "file_sha256": real_sha256,
            "file_path": f"onnx/{model_path.name}",
            "imgsz": 640,
            "class_map_hash": _STOCK_CLASS_MAP_HASH,
            "pt_provenance": {
                "model_name": "dfine-l",
                "source": "dfine-coco-stock",
                "license": "Apache-2.0",
                "file_sha256": "ptsha000",
            },
        }))
        manifest_path = tmp_path / "designation-manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "targets": {"cloud": {"artifacts": {"f" * 64: {"designation_status": "designated"}}}},
        }))

        fixture_settings = _fixture_settings(model_path, provenance_path, execution_target="cloud")
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="absent"):
            run_acceptance_smoke()


class TestAcceptanceSmokePositive:
    """The one positive invocation: a
    designated-matching artifact proceeds."""

    def test_designated_matching_artifact_proceeds(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"designated-onnx-bytes", designation_status="designated"
        )
        fixture_settings = _fixture_settings(model_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = run_acceptance_smoke()

        assert isinstance(detector, DFineORTDetector)
        assert detector.designation_status == "designated"
        assert detector.execution_target == "cloud"


class TestAcceptanceSmokeJetsonRefusals:
    """Jetson/TensorRT counterparts of ``TestAcceptanceSmokeRefusals``."""

    def test_jetson_integration_only_artifact_refuses(self, tmp_path, monkeypatch):
        engine_path, provenance_path, manifest_path = _write_engine_fixture(
            tmp_path, b"jetson-integration-only-engine-bytes", designation_status="integration-only"
        )
        fixture_settings = _fixture_jetson_settings(engine_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="integration-only"):
            run_acceptance_smoke()

    def test_jetson_hash_mismatched_artifact_refuses(self, tmp_path, monkeypatch):
        engine_path, provenance_path, manifest_path = _write_engine_fixture(
            tmp_path,
            b"jetson-tampered-engine-bytes",
            designation_status="designated",
            declared_sha256="0" * 64,
        )
        fixture_settings = _fixture_jetson_settings(engine_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="sha256 mismatch"):
            run_acceptance_smoke()


class TestAcceptanceSmokeJetsonPositive:
    """Jetson/TensorRT counterpart of ``TestAcceptanceSmokePositive``: the designated-matching
    Jetson/TensorRT path proceeds."""

    def test_jetson_designated_matching_artifact_proceeds(self, tmp_path, monkeypatch):
        engine_path, provenance_path, manifest_path = _write_engine_fixture(
            tmp_path, b"jetson-designated-engine-bytes", designation_status="designated"
        )
        fixture_settings = _fixture_jetson_settings(engine_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = run_acceptance_smoke()

        assert isinstance(detector, DFineTensorRTDetector)
        assert detector.designation_status == "designated"
        assert detector.execution_target == "jetson"


class TestAcceptanceSmokeCiDevMode:
    """
    The same entrypoint (``run_acceptance_smoke``), driven
    by the SAME fixtures that refuse under the default ``"acceptance"``
    mode above, proceeds under explicit ``INFERENCE_ACCEPTANCE_SMOKE_MODE=
    "ci-dev"`` — the deterministic stub is allowed only in CI/dev smoke tests: no
    live Jetson/GPU call, mode-gated by explicit config, never inferred.
    """

    def test_ci_dev_mode_proceeds_for_stub(self, monkeypatch):
        fixture_settings = Settings(INFERENCE_DETECTOR_IMPL="lookup", INFERENCE_ACCEPTANCE_SMOKE_MODE="ci-dev")
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = run_acceptance_smoke()

        assert detector is not None

    def test_ci_dev_mode_proceeds_for_non_designated_artifact(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"integration-only-onnx-bytes-ci-dev", designation_status="integration-only"
        )
        fixture_settings = _fixture_settings(
            model_path, provenance_path, INFERENCE_ACCEPTANCE_SMOKE_MODE="ci-dev"
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        detector = run_acceptance_smoke()

        assert isinstance(detector, DFineORTDetector)
        assert detector.designation_status == "integration-only"

    def test_ci_dev_mode_still_refuses_hash_mismatched_artifact(self, tmp_path, monkeypatch):
        """Mode only gates the acceptance-policy checks (stub exclusion,
        designation-status=="designated"); a genuine construction failure
        (corrupt/tampered artifact) is not an acceptance-policy question
        and still refuses in either mode."""
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path,
            b"tampered-onnx-bytes-ci-dev",
            designation_status="designated",
            declared_sha256="0" * 64,
        )
        fixture_settings = _fixture_settings(
            model_path, provenance_path, INFERENCE_ACCEPTANCE_SMOKE_MODE="ci-dev"
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        with pytest.raises(AcceptanceSmokeRefusal, match="sha256 mismatch"):
            run_acceptance_smoke()


class TestAcceptanceSmokeEndpoint:
    """
    Live invocations of ``POST /internal/acceptance-smoke`` through a real
    ``fastapi.testclient.TestClient`` against the actual ``app.main.app``: these exercise the real HTTP route -> real
    ``run_acceptance_smoke_endpoint`` -> real ``run_acceptance_smoke`` ->
    real ``registry.get_detector`` dispatch chain, not the helper called
    directly — proving the production entrypoint enforces the gate, not
    merely that the underlying helper works in isolation.
    """

    def test_endpoint_refuses_stub_with_409(self, monkeypatch):
        fixture_settings = Settings(INFERENCE_DETECTOR_IMPL="lookup")
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        client = TestClient(fastapi_app, raise_server_exceptions=False)
        response = client.post("/internal/acceptance-smoke")

        assert response.status_code == 409
        assert "stub" in response.json()["detail"]

    def test_endpoint_refuses_hash_mismatched_artifact_with_409(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path,
            b"tampered-onnx-bytes-endpoint",
            designation_status="designated",
            declared_sha256="1" * 64,
        )
        fixture_settings = _fixture_settings(model_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        client = TestClient(fastapi_app, raise_server_exceptions=False)
        response = client.post("/internal/acceptance-smoke")

        assert response.status_code == 409
        assert "sha256 mismatch" in response.json()["detail"]

    def test_endpoint_proceeds_for_designated_matching_artifact(self, tmp_path, monkeypatch):
        model_path, provenance_path, manifest_path = _write_onnx_fixture(
            tmp_path, b"designated-onnx-bytes-endpoint", designation_status="designated"
        )
        fixture_settings = _fixture_settings(model_path, provenance_path)
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        client = TestClient(fastapi_app, raise_server_exceptions=False)
        response = client.post("/internal/acceptance-smoke")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "designated"
        assert body["mode"] == "acceptance"
        assert body["execution_target"] == "cloud"
        assert body["designation_status"] == "designated"

    def test_endpoint_proceeds_for_stub_in_ci_dev_mode(self, monkeypatch):
        """Same route, same handler, explicit ``mode=ci-dev`` config —
        proceeds instead of refusing (409) for the very fixture that
        ``test_endpoint_refuses_stub_with_409`` above refuses under the
        default ``acceptance`` mode."""
        fixture_settings = Settings(INFERENCE_DETECTOR_IMPL="lookup", INFERENCE_ACCEPTANCE_SMOKE_MODE="ci-dev")
        monkeypatch.setattr(config_module, "get_settings", lambda: fixture_settings)

        client = TestClient(fastapi_app, raise_server_exceptions=False)
        response = client.post("/internal/acceptance-smoke")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "proceeded"
        assert body["mode"] == "ci-dev"

