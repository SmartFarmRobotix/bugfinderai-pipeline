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
The integration test asserts PDM received exactly the
risk-input fields.

Exercises the real `assess_via_pdm`/`get_gatekeeper_token` call path with
only the HTTP boundary mocked (`requests.post`) — the URL construction,
GK-proxy path, auth header, and JSON body assembly all run for real.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gatekeeper
import pdm_client
from contract import GrowthStage, RiskInput
from pdm_client import PdmConnectionError, PdmHTTPError, assess_via_pdm


class _Settings:
    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    pdm_service_name = "pdm"
    pdm_risk_assessment_path = "/api/v1/risk-assessment/"
    pdm_call_timeout_seconds = 15


def _risk_input() -> RiskInput:
    return RiskInput(
        session_id="sess-1",
        infestation_score=0.6,
        crop="brassica-oleracea",
        growth_stage=GrowthStage(representation="enum", unknown=False, stage_id="BBCH-19"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
        position_accuracy_m=5.0,
    )


_RISK_INPUT_FIELDS = {
    "session_id",
    "infestation_score",
    "crop",
    "growth_stage",
    "time_start",
    "time_end",
    "latitude",
    "longitude",
    "position_accuracy_m",
}


class TestPdmReceivesExactlyTheRiskInputFields:
    def test_request_body_is_exactly_the_risk_input(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "session_id": "sess-1",
                "risk_assessment": {"level": "Medium", "details": "ok", "provenance": {}},
                "errors": [],
            }
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        result = assess_via_pdm(_Settings(), _risk_input())

        assert result.level == "Medium"
        sent_body = captured["json"]
        assert set(sent_body.keys()) == _RISK_INPUT_FIELDS
        assert sent_body["session_id"] == "sess-1"
        assert sent_body["infestation_score"] == 0.6
        assert "best_image_id" not in sent_body
        assert "provenance" not in sent_body

    def test_request_reaches_pdm_through_the_gatekeeper_proxy_path(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            captured["url"] = url
            captured["headers"] = headers
            response = MagicMock(status_code=200)
            response.json.return_value = {"session_id": "sess-1", "risk_assessment": None, "errors": []}
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        assess_via_pdm(_Settings(), _risk_input())

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/pdm/api/v1/risk-assessment/"
        assert captured["headers"]["Authorization"] == "Bearer test-token"


class TestPdmAssessmentResultRawLevel:
    """`raw_level` is derived straight from
    `provenance["weather_rule_level_raw"]` — PDM's separate, pre-score
    field — never from `level` (PDM's own already-score-combined value)."""

    def test_raw_level_reads_from_provenance_not_level(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "session_id": "sess-1",
                "risk_assessment": {
                    "level": "High",
                    "details": "ok",
                    "provenance": {"weather_rule_level_raw": "moderate"},
                },
                "errors": [],
            }
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        result = assess_via_pdm(_Settings(), _risk_input())

        assert result.level == "High"
        assert result.raw_level == "moderate"
        assert result.raw_level != result.level

    def test_raw_level_absent_when_provenance_lacks_it(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "session_id": "sess-1",
                "risk_assessment": {"level": "High", "details": "ok", "provenance": {}},
                "errors": [],
            }
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        result = assess_via_pdm(_Settings(), _risk_input())

        assert result.raw_level is None


class TestPdmCallErrorHandling:
    def test_connection_failure_raises_pdm_connection_error(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        import requests as requests_module

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            raise requests_module.ConnectionError("boom")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        with pytest.raises(PdmConnectionError):
            assess_via_pdm(_Settings(), _risk_input())

    def test_http_error_raises_pdm_http_error(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            return MagicMock(status_code=502, text="upstream error")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        with pytest.raises(PdmHTTPError):
            assess_via_pdm(_Settings(), _risk_input())
