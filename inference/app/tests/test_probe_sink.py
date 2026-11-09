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

"""The probe-sink endpoint
(``POST /api/v1/probe/sink``) -- authenticated exactly like the feeder's
session endpoints, size-capped, storage-free.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("INFERENCE_USING_GATEKEEPER", "false")
os.environ.setdefault("INFERENCE_ORCH_PUSH_ENABLED", "false")
os.environ.setdefault("INFERENCE_STORAGE_DIR", "/tmp/inf_test_storage_probe_sink")

from app.config import Settings, get_settings  # noqa: E402
from app.main import app, gatekeeper_auth, _register_service  # noqa: E402


def _make_settings(storage_dir: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(storage_dir),
        INFERENCE_DATA_ROOT=str(storage_dir),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _hash_tree(root: Path) -> str:
    """Recursive content hash of every file under root (path + bytes), order-independent."""
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


class TestProbeSinkAuthenticatedUpload:
    def test_authenticated_upload_returns_exact_byte_count_and_stores_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            settings = _make_settings(storage_dir, INFERENCE_EXTERNAL_API_KEY="probe-key")
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)

            payload = bytes(262144)
            before_hash = _hash_tree(storage_dir)
            resp = client.post(
                "/api/v1/probe/sink",
                headers={"X-API-Key": "probe-key"},
                content=payload,
            )
            after_hash = _hash_tree(storage_dir)

        assert resp.status_code == 200
        assert resp.json() == {"received_bytes": 262144}
        assert after_hash == before_hash, "sink must not create/overwrite anything under storage_dir"


class TestProbeSinkAuth:
    def test_missing_key_refused_with_session_endpoint_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), INFERENCE_EXTERNAL_API_KEY="probe-key")
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/probe/sink", content=b"x" * 10)
        assert resp.status_code == 403

    def test_wrong_key_refused_with_session_endpoint_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), INFERENCE_EXTERNAL_API_KEY="probe-key")
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/probe/sink", headers={"X-API-Key": "wrong-key"}, content=b"x" * 10
            )
        assert resp.status_code == 403


class TestProbeSinkCap:
    def test_over_cap_payload_refused_loudly_with_cap_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(
                Path(tmp), INFERENCE_EXTERNAL_API_KEY="probe-key", INFERENCE_PROBE_SINK_MAX_BYTES="100"
            )
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/probe/sink",
                headers={"X-API-Key": "probe-key"},
                content=b"x" * 200,
            )
        assert resp.status_code == 413
        assert "INFERENCE_PROBE_SINK_MAX_BYTES" in resp.text
        assert "100" in resp.text


class TestProbeSinkGatekeeperRegistration:
    def test_registration_endpoint_list_contains_sink_route(self):
        recorded_endpoints: list[str] = []

        def _fake_post_json(url, payload, headers, timeout):
            if "endpoint" in payload:
                recorded_endpoints.append(payload["endpoint"])
                return 200, "{}"
            return 200, '{"access": "fake-token"}'

        settings = _make_settings(Path("/tmp/inf_test_storage_probe_sink_gk"), INFERENCE_USING_GATEKEEPER="true")
        with patch("app.main._post_json", side_effect=_fake_post_json):
            _register_service(settings)

        assert "api/v1/probe/sink" in recorded_endpoints
