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

"""Weather-service history call via Gatekeeper.

Exercises the real `fetch_hourly_history`/`get_gatekeeper_token` call path
with only the HTTP boundary mocked (`requests.post`) — the URL
construction, GK-proxy path, auth header, and JSON body assembly all run
for real, mirroring test_pdm_client.py's pattern for the PDM call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gatekeeper
import ws_client
from ws_client import (
    REQUIRED_VARIABLES,
    WeatherServiceConnectionError,
    WeatherServiceHTTPError,
    fetch_hourly_forecast,
    fetch_hourly_history,
)


class _Settings:
    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    ws_service_name = "weather_data"
    ws_history_hourly_path = "/api/v1/history/hourly/"
    ws_forecast_hourly_path = "/api/v1/forecast/hourly/"
    ws_history_radius_km = 10.0
    ws_call_timeout_seconds = 15


def _login_ok() -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {"access": "test-token"}
    return response


class TestFetchHourlyHistoryRequestShape:
    def test_request_reaches_ws_through_the_gatekeeper_proxy_path(self, monkeypatch):
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "source": "openmeteo",
                "location": {"lat": 45.0, "lon": 10.0},
                "data": [],
            }
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        fetch_hourly_history(
            _Settings(),
            latitude=45.0,
            longitude=10.0,
            time_start="2026-06-01T00:00:00Z",
            time_end="2026-06-05T23:00:00Z",
        )

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/weather_data/api/v1/history/hourly/"
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"] == {
            "lat": 45.0,
            "lon": 10.0,
            "start": "2026-06-01",
            "end": "2026-06-05",
            "variables": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "radius_km": 10.0,
        }
        assert captured["timeout"] == 15

    def test_response_body_returned_verbatim(self, monkeypatch):
        expected_body = {"source": "openmeteo", "location": {"lat": 1.0, "lon": 2.0}, "data": [{"x": 1}]}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            response = MagicMock(status_code=200)
            response.json.return_value = expected_body
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        body = fetch_hourly_history(
            _Settings(),
            latitude=1.0,
            longitude=2.0,
            time_start="2026-06-01T00:00:00Z",
            time_end="2026-06-05T23:00:00Z",
        )

        assert body == expected_body


class TestFetchHourlyHistoryErrorHandling:
    def test_connection_failure_raises_weather_service_connection_error(self, monkeypatch):
        import requests as requests_module

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            raise requests_module.ConnectionError("boom")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        with pytest.raises(WeatherServiceConnectionError):
            fetch_hourly_history(
                _Settings(),
                latitude=45.0,
                longitude=10.0,
                time_start="2026-06-01T00:00:00Z",
                time_end="2026-06-05T23:00:00Z",
            )

    def test_http_error_raises_weather_service_http_error(self, monkeypatch):
        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            return MagicMock(status_code=502, text="upstream error")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        with pytest.raises(WeatherServiceHTTPError):
            fetch_hourly_history(
                _Settings(),
                latitude=45.0,
                longitude=10.0,
                time_start="2026-06-01T00:00:00Z",
                time_end="2026-06-05T23:00:00Z",
            )

    def test_invalid_time_start_raises(self, monkeypatch):
        monkeypatch.setattr(gatekeeper.requests, "post", lambda *a, **k: _login_ok())

        with pytest.raises(ws_client.WeatherServiceCallError):
            fetch_hourly_history(
                _Settings(),
                latitude=45.0,
                longitude=10.0,
                time_start="not-a-date",
                time_end="2026-06-05T23:00:00Z",
            )


class TestFetchHourlyForecastRequestShape:
    def test_request_reaches_ws_through_the_gatekeeper_proxy_path(self, monkeypatch):
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "source": "openmeteo",
                "location": {"lat": 45.0, "lon": 10.0},
                "data": [{"timestamp": "2026-06-01T00:00", "values": {}}],
            }
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        fetch_hourly_forecast(_Settings(), latitude=45.0, longitude=10.0, days=5)

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/weather_data/api/v1/forecast/hourly/"
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"] == {
            "lat": 45.0,
            "lon": 10.0,
            "days": 5,
            "variables": REQUIRED_VARIABLES + ["wind_speed_10m"],
        }
        assert captured["timeout"] == 15

    def test_response_body_returned_verbatim(self, monkeypatch):
        expected_body = {
            "source": "openmeteo", "location": {"lat": 1.0, "lon": 2.0}, "data": [{"x": 1}],
        }

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            response = MagicMock(status_code=200)
            response.json.return_value = expected_body
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        body = fetch_hourly_forecast(_Settings(), latitude=1.0, longitude=2.0, days=5)

        assert body == expected_body


class TestFetchHourlyForecastErrorHandling:
    def test_connection_failure_raises_weather_service_connection_error(self, monkeypatch):
        import requests as requests_module

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            raise requests_module.ConnectionError("boom")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        with pytest.raises(WeatherServiceConnectionError):
            fetch_hourly_forecast(_Settings(), latitude=45.0, longitude=10.0, days=5)

    def test_http_error_raises_weather_service_http_error(self, monkeypatch):
        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            return MagicMock(status_code=502, text="upstream error")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        with pytest.raises(WeatherServiceHTTPError):
            fetch_hourly_forecast(_Settings(), latitude=45.0, longitude=10.0, days=5)

    def test_a_response_without_rows_is_refused_not_invented(self, monkeypatch):
        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            response = MagicMock(status_code=200)
            response.json.return_value = {"source": "openmeteo", "location": {}, "data": []}
            return response

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)

        with pytest.raises(WeatherServiceHTTPError):
            fetch_hourly_forecast(_Settings(), latitude=45.0, longitude=10.0, days=5)
