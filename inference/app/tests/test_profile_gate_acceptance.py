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
The INT8 negligibility gate.

The service is started under ``edge-s-int8`` with a provenance that matches
the profile (variant s, precision int8, a tiling grid) and NO negligibility
record for the artifact: startup refuses, naming the profile and the missing record.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.config as config_module
import app.main as main_module
from app.config import Settings
from app.main import _DETECTOR_CACHE, app
from app.tests.tiled_stubs import write_artifact


def test_service_refuses_to_start_under_the_int8_profile_without_a_negligibility_record(tmp_path, monkeypatch):
    files = write_artifact(
        tmp_path,
        imgsz=1280,
        tiling={"tile": 1280, "stride": 1024},
        extra={"variant": "s", "precision": "int8"},
    )
    record_dir = tmp_path / "negligibility-records"
    record_dir.mkdir()
    settings = Settings(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(tmp_path / "storage"),
        INFERENCE_DATA_ROOT=str(tmp_path / "storage"),
        INFERENCE_DETECTOR_IMPL="ort",
        INFERENCE_ONNX_MODEL_PATH=str(files.path),
        INFERENCE_ONNX_PROVENANCE_PATH=str(files.provenance_path),
        INFERENCE_ONNX_EXECUTION_PROVIDER="CPUExecutionProvider",
        INFERENCE_DETECTOR_PROFILE="edge-s-int8",
        INFERENCE_NEGLIGIBILITY_RECORD_DIR=str(record_dir),
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.delitem(_DETECTOR_CACHE, "ort", raising=False)

    with pytest.raises(Exception) as excinfo:
        with TestClient(app) as client:
            assert client.get("/healthz").status_code == 200

    message = str(excinfo.value)
    assert "edge-s-int8" in message
    assert "negligibility" in message
    assert str(record_dir / f"{files.sha256}.negligibility.json") in message
    _DETECTOR_CACHE.pop("ort", None)
