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
Report triggering: exercises the real `push_report`/
`get_gatekeeper_token` call path with only the HTTP boundary mocked,
mirroring test_pdm_client.py's pattern.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gatekeeper
import reporting_client
from reporting_client import ReportingConnectionError, ReportingHTTPError, push_report


class _Settings:
    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    rs_service_name = "reporting"
    rs_pest_session_report_path = "/api/v1/openagri-report/pest-session-report/"
    rs_call_timeout_seconds = 15


class TestPushReportReachesReportingThroughGatekeeperProxy:
    def test_request_reaches_rs_through_the_gatekeeper_proxy_path(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            response = MagicMock(status_code=200)
            response.json.return_value = {"uuid": "report-uuid-1"}
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(reporting_client.requests, "post", _fake_post)

        result = push_report(_Settings(), {"infestation_score": 0.5})

        assert result.report_id == "report-uuid-1"
        assert captured["url"] == (
            "http://gatekeeper:8000/api/proxy/reporting/api/v1/openagri-report/pest-session-report/"
        )
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"] == {"infestation_score": 0.5}


class TestPushReportErrorHandling:
    def test_connection_failure_raises_reporting_connection_error(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        import requests as requests_module

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            raise requests_module.ConnectionError("boom")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(reporting_client.requests, "post", _fake_post)

        with pytest.raises(ReportingConnectionError):
            push_report(_Settings(), {})

    def test_http_error_raises_reporting_http_error(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            return MagicMock(status_code=502, text="upstream error")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(reporting_client.requests, "post", _fake_post)

        with pytest.raises(ReportingHTTPError):
            push_report(_Settings(), {})

    def test_missing_uuid_raises_reporting_http_error(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return login_response
            response = MagicMock(status_code=200)
            response.json.return_value = {}
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(reporting_client.requests, "post", _fake_post)

        with pytest.raises(ReportingHTTPError):
            push_report(_Settings(), {})
