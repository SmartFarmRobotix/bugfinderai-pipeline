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
The orchestrator self-registers with the Gatekeeper (`gk_registration.py`), or the inference
service's push target (`/api/proxy/orchestrator/...`) has nothing to route to. Mirrors PDM's
best-effort, never-blocks-startup registration pattern.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gatekeeper
import gk_registration
import main
from gk_registration import register_apis_to_gatekeeper


class _Settings:
    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    service_name = "orchestrator"
    port = 8020


def _login_response():
    response = MagicMock(status_code=200)
    response.json.return_value = {"access": "test-token", "refresh": "test-refresh"}
    return response


class TestRegisterApisToGatekeeper:
    def test_registers_every_api_route_with_correct_base_url_and_service_name(self, monkeypatch):
        app = FastAPI()

        @app.post("/api/v1/sessions/")
        def _intake():
            return {}

        @app.get("/health")
        def _health():
            return {}

        calls = []

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_response()
            if url.endswith("/api/register_service/"):
                calls.append(json)
                return MagicMock(status_code=201)
            if url.endswith("/api/logout/"):
                return MagicMock(status_code=200)
            raise AssertionError(f"unexpected URL: {url}")

        monkeypatch.setattr(gk_registration.requests, "post", _fake_post)

        register_apis_to_gatekeeper(_Settings(), app)

        assert len(calls) == 1
        assert calls[0]["base_url"] == "http://orchestrator:8020/"
        assert calls[0]["service_name"] == "orchestrator"
        assert calls[0]["endpoint"] == "api/v1/sessions/"
        assert "POST" in calls[0]["methods"]

    def test_gatekeeper_unreachable_at_login_does_not_raise(self, monkeypatch):
        import requests as requests_module

        def _fake_post(url, headers=None, json=None, timeout=None):
            raise requests_module.ConnectionError("boom")

        monkeypatch.setattr(gk_registration.requests, "post", _fake_post)

        app = FastAPI()

        register_apis_to_gatekeeper(_Settings(), app)

    def test_one_route_failing_to_register_does_not_stop_the_others(self, monkeypatch):
        app = FastAPI()

        @app.post("/api/v1/sessions/")
        def _intake():
            return {}

        @app.get("/api/v1/sessions/{session_id}")
        def _get_session(session_id: str):
            return {}

        calls = []

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_response()
            if url.endswith("/api/register_service/"):
                calls.append(json["endpoint"])
                if json["endpoint"] == "api/v1/sessions/":
                    return MagicMock(status_code=502)
                return MagicMock(status_code=201)
            return MagicMock(status_code=200)

        monkeypatch.setattr(gk_registration.requests, "post", _fake_post)

        register_apis_to_gatekeeper(_Settings(), app)

        assert set(calls) == {"api/v1/sessions/", "api/v1/sessions/{session_id}"}


class TestLifespanRegistersOnStartupForCloudNotEdge:
    def test_cloud_profile_calls_registration_on_startup(self, monkeypatch):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "cloud")
        monkeypatch.setenv("ORCH_GATEKEEPER_USERNAME", "orchestrator")
        monkeypatch.setenv("ORCH_GATEKEEPER_PASSWORD", "orchestrator-secret")
        import config as config_module

        config_module.get_settings.cache_clear()

        called = {}

        def _fake_register(settings, app):
            called["invoked"] = True

        monkeypatch.setattr(main, "register_apis_to_gatekeeper", _fake_register)

        with TestClient(main.app):
            pass

        assert called.get("invoked") is True
        config_module.get_settings.cache_clear()

    def test_edge_profile_never_calls_registration(self, monkeypatch):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", "a-real-unique-edge-secret")
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)
        import config as config_module

        config_module.get_settings.cache_clear()

        called = {}

        def _fake_register(settings, app):
            called["invoked"] = True

        monkeypatch.setattr(main, "register_apis_to_gatekeeper", _fake_register)

        with TestClient(main.app):
            pass

        assert "invoked" not in called
        config_module.get_settings.cache_clear()
