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

"""The forecast outlook of a session
report. The Weather service, the PDM and GateKeeper are mocked at the HTTP
boundary; the clients, the fuzzy engine on the hourly frame, the payload
composition and the PDF renderer run for real.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import DateTime

import fuzzy_client
import gatekeeper
import pdm_client
import ws_client
from contract import CONTRACT_VERSION, DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput, SessionSubmission
from forecast_builder import build_forecast_outlook
from forecast_outlook import (
    FUZZY_NOT_EVALUATED,
    ForecastOutlookData,
    ForecastUnavailable,
    UNAVAILABLE_NO_KEY,
    UNAVAILABLE_NO_POSITION,
    UNAVAILABLE_RULES,
    UNAVAILABLE_SERVICE,
    compose_outlook_summary,
)
from fuzzy_client import FuzzyAssessmentError, assess_outlook_hours_via_fuzzy_engine
from models import RiskAssessmentRecord, SessionState
from pdm_client import (
    PdmCallError,
    PdmForecastResult,
    PdmHTTPError,
    assess_forecast_hours_via_pdm,
    assess_forecast_via_pdm,
)
from report_content import RECOMMENDATION_MARKER
from report_payload import compose_report_payload
from report_pdf.pest_session_pdf_core import render_pest_session_pdf
from reporting_client import ReportingConnectionError, ReportPushResult
from service import advance_session, intake_submission
from ws_client import (
    DAILY_HISTORY_VARIABLES,
    WeatherServiceConnectionError,
    WeatherServiceHTTPError,
    fetch_daily_forecast,
    fetch_daily_history,
)

_DAYS_RAW = [
    {"date": "2026-09-22", "temp_max": 24.0, "temp_min": 12.0, "humidity": 72.0,
     "precipitation": 0.4, "wind_speed": 3.1, "pop": 0.4, "description": "light rain"},
    {"date": "2026-09-23", "temp_max": 18.5, "temp_min": 9.0, "humidity": 80.0,
     "precipitation": 2.2, "wind_speed": 4.0, "pop": 0.8, "description": "rain"},
    {"date": "2026-09-24", "temp_max": 29.0, "temp_min": 15.0, "humidity": 55.0,
     "precipitation": 0.0, "wind_speed": 2.0, "pop": 0.0, "description": "clear sky"},
    {"date": "2026-09-25", "temp_max": 21.0, "temp_min": 11.0, "humidity": 74.0,
     "precipitation": 0.0, "wind_speed": 2.5, "pop": 0.1, "description": "few clouds"},
    {"date": "2026-09-26", "temp_max": 15.0, "temp_min": 6.0, "humidity": 66.0,
     "precipitation": 0.3, "wind_speed": 5.2, "pop": 0.3, "description": "scattered clouds"},
]
_FETCHED_AT = datetime.datetime(2026, 9, 21, 9, 30, 15, tzinfo=datetime.timezone.utc)


class _Settings:
    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    ws_service_name = "weather_data"
    ws_forecast_daily_path = "/api/v1/forecast/daily/"
    ws_forecast_hourly_path = "/api/v1/forecast/hourly/"
    ws_history_daily_path = "/api/v1/history/daily/"
    ws_history_hourly_path = "/api/v1/history/hourly/"
    ws_history_radius_km = 10.0
    ws_call_timeout_seconds = 15
    pdm_service_name = "pdm"
    pdm_forecast_risk_path = "/api/v1/risk-assessment/forecast/"
    pdm_call_timeout_seconds = 15
    max_outbound_retries = 3


def _login_ok() -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {"access": "test-token"}
    return response


def _route(monkeypatch, module, answer, captured):
    """Fake `requests.post` for GK login + one proxied call; `answer` is the
    proxied response (or an exception to raise)."""

    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith("/api/login/"):
            return _login_ok()
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
    monkeypatch.setattr(module.requests, "post", _fake_post)


def _response(status_code=200, body=None, text=""):
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = body
    return response


class TestFetchDailyForecast:
    def _forecast_body(self):
        return {"location": {"lat": 45.0, "lon": 10.0}, "data": _DAYS_RAW, "source": "openweathermap"}

    def test_request_goes_through_the_gatekeeper_proxy_and_rows_are_mapped(self, monkeypatch):
        captured = {}
        _route(monkeypatch, ws_client, _response(body=self._forecast_body()), captured)

        forecast = fetch_daily_forecast(_Settings(), 45.0, 10.0, days=5, now=_FETCHED_AT)

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/weather_data/api/v1/forecast/daily/"
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"] == {"lat": 45.0, "lon": 10.0, "days": 5}
        assert captured["timeout"] == 15
        assert forecast["source"] == "openweathermap"
        assert forecast["fetched_at"] == "2026-09-21T09:30:15+00:00"
        assert forecast["location"] == {"lat": 45.0, "lon": 10.0}
        assert len(forecast["days"]) == 5
        assert forecast["days"][1] == {
            "date": "2026-09-23",
            "temp_max_C": 18.5,
            "temp_min_C": 9.0,
            "humidity_pct": 80.0,
            "precipitation_mm": 2.2,
            "wind_speed_ms": 4.0,
        }

    def test_fetched_at_is_utc_whatever_the_clock_zone(self, monkeypatch):
        captured = {}
        _route(monkeypatch, ws_client, _response(body=self._forecast_body()), captured)
        utc_plus_3 = datetime.timezone(datetime.timedelta(hours=3))

        forecast = fetch_daily_forecast(
            _Settings(), 45.0, 10.0, now=datetime.datetime(2026, 9, 21, 12, 30, 15, tzinfo=utc_plus_3)
        )

        assert forecast["fetched_at"] == "2026-09-21T09:30:15+00:00"

    def test_an_http_error_is_raised_with_its_status_and_body(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(503, text="OpenWeatherMap API key is not configured."), {})

        with pytest.raises(WeatherServiceHTTPError) as caught:
            fetch_daily_forecast(_Settings(), 45.0, 10.0)

        assert caught.value.status_code == 503
        assert "API key" in caught.value.body

    def test_an_unreachable_service_is_a_connection_error(self, monkeypatch):
        import requests

        _route(monkeypatch, ws_client, requests.ConnectionError("refused"), {})

        with pytest.raises(WeatherServiceConnectionError):
            fetch_daily_forecast(_Settings(), 45.0, 10.0)

    def test_a_response_without_rows_is_refused_not_invented(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(body={"location": {}, "data": [], "source": "x"}), {})

        with pytest.raises(WeatherServiceHTTPError):
            fetch_daily_forecast(_Settings(), 45.0, 10.0)


_MARCH_DAYS = [
    ("2026-03-06", 11.0, 2.5, 78.0, 0.0, 4.2),
    ("2026-03-07", 9.5, 1.0, 84.0, 3.4, 6.1),
    ("2026-03-08", 12.0, 3.0, 70.0, 0.0, 3.3),
    ("2026-03-09", 14.5, 4.5, 65.0, 0.2, 2.8),
    ("2026-03-10", 8.0, 0.5, 88.0, 6.0, 7.0),
]


def _history_body(source="openmeteo", drop=()):
    return {
        "location": {"lat": 45.0, "lon": 10.0},
        "source": source,
        "data": [
            {
                "date": day,
                "values": {
                    key: value
                    for key, value in zip(DAILY_HISTORY_VARIABLES, (tmax, tmin, humidity, rain, wind))
                    if key not in drop
                },
            }
            for day, tmax, tmin, humidity, rain, wind in _MARCH_DAYS
        ],
    }


class TestFetchDailyHistory:
    def test_request_goes_through_the_gatekeeper_proxy_and_rows_are_mapped(self, monkeypatch):
        captured = {}
        _route(monkeypatch, ws_client, _response(body=_history_body()), captured)

        history = fetch_daily_history(
            _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
            DAILY_HISTORY_VARIABLES, now=_FETCHED_AT,
        )

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/weather_data/api/v1/history/daily/"
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"] == {
            "lat": 45.0,
            "lon": 10.0,
            "start": "2026-03-06",
            "end": "2026-03-10",
            "variables": [
                "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean",
                "precipitation_sum", "wind_speed_10m_max",
            ],
            "radius_km": 10.0,
        }
        assert captured["timeout"] == 15
        assert history["source"] == "openmeteo"
        assert history["fetched_at"] == "2026-09-21T09:30:15+00:00"
        assert history["location"] == {"lat": 45.0, "lon": 10.0}
        assert len(history["days"]) == 5
        assert history["days"][1] == {
            "date": "2026-03-07",
            "temp_max_C": 9.5,
            "temp_min_C": 1.0,
            "humidity_pct": 84.0,
            "precipitation_mm": 3.4,
            "wind_speed_ms": 6.1,
        }

    def test_a_variable_the_response_lacks_leaves_its_cell_empty(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(body=_history_body(drop=("wind_speed_10m_max",))), {})

        history = fetch_daily_history(
            _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
            DAILY_HISTORY_VARIABLES,
        )

        assert [day["wind_speed_ms"] for day in history["days"]] == [None] * 5
        assert history["days"][0]["temp_max_C"] == 11.0

    def test_a_variable_that_was_not_requested_is_empty_even_if_returned(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(body=_history_body()), {})

        history = fetch_daily_history(
            _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
            ["temperature_2m_max"],
        )

        assert history["days"][0]["temp_max_C"] == 11.0
        assert history["days"][0]["humidity_pct"] is None

    def test_an_http_error_is_raised_with_its_status_and_body(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(502, text="upstream failed"), {})

        with pytest.raises(WeatherServiceHTTPError) as caught:
            fetch_daily_history(
                _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
                DAILY_HISTORY_VARIABLES,
            )

        assert caught.value.status_code == 502

    def test_an_unreachable_service_is_a_connection_error(self, monkeypatch):
        import requests

        _route(monkeypatch, ws_client, requests.ConnectionError("refused"), {})

        with pytest.raises(WeatherServiceConnectionError):
            fetch_daily_history(
                _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
                DAILY_HISTORY_VARIABLES,
            )

    def test_a_response_without_rows_is_refused_not_invented(self, monkeypatch):
        _route(monkeypatch, ws_client, _response(body={"location": {}, "data": [], "source": "x"}), {})

        with pytest.raises(WeatherServiceHTTPError):
            fetch_daily_history(
                _Settings(), 45.0, 10.0, datetime.date(2026, 3, 6), datetime.date(2026, 3, 10),
                DAILY_HISTORY_VARIABLES,
            )


_MAPPED_ROWS = [
    {
        "date": row["date"],
        "temp_max_C": row["temp_max"],
        "temp_min_C": row["temp_min"],
        "humidity_pct": row["humidity"],
        "precipitation_mm": row["precipitation"],
        "wind_speed_ms": row["wind_speed"],
    }
    for row in _DAYS_RAW
]


class TestAssessForecastViaPdm:
    def test_request_reaches_the_pdm_forecast_route_through_the_gatekeeper(self, monkeypatch):
        captured = {}
        answer = {"days": [{"date": "2026-09-22", "level": "moderate", "rules_triggered": ["temperature_2m >= 20"]}]}
        _route(monkeypatch, pdm_client, _response(body=answer), captured)

        result = assess_forecast_via_pdm(_Settings(), "brassica-oleracea", _MAPPED_ROWS)

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/pdm/api/v1/risk-assessment/forecast/"
        assert captured["json"]["crop"] == "brassica-oleracea"
        assert captured["json"]["days"][0] == {
            "date": "2026-09-22",
            "temp_max": 24.0,
            "temp_min": 12.0,
            "humidity": 72.0,
            "precipitation": 0.4,
        }
        assert len(captured["json"]["days"]) == 5
        assert result.days == answer["days"]
        assert result.reason is None

    def test_the_named_reason_comes_back_with_no_days(self, monkeypatch):
        _route(monkeypatch, pdm_client, _response(body={"days": [], "reason": "no_pest_models_configured"}), {})

        result = assess_forecast_via_pdm(_Settings(), "brassica-oleracea", _MAPPED_ROWS)

        assert result.days == []
        assert result.reason == "no_pest_models_configured"

    def test_an_http_error_is_a_pdm_call_error(self, monkeypatch):
        _route(monkeypatch, pdm_client, _response(500, text="boom"), {})

        with pytest.raises(PdmHTTPError):
            assess_forecast_via_pdm(_Settings(), "brassica-oleracea", _MAPPED_ROWS)


def _hours_of(day, temperature, humidity, precipitation=0.0, rain_hour=6):
    """24 Weather-service hourly rows for one day: constant temperature and
    humidity, the day's precipitation in one hour."""
    return [
        {
            "timestamp": f"{day}T{hour:02d}:00",
            "values": {
                "temperature_2m": temperature,
                "relative_humidity_2m": humidity,
                "precipitation": precipitation if hour == rain_hour else 0.0,
            },
        }
        for hour in range(24)
    ]


def _five_march_days(temperature, humidity, precipitation=0.0):
    return [
        row
        for day in range(6, 11)
        for row in _hours_of(f"2026-03-{day:02d}", temperature, humidity, precipitation)
    ]


class TestFuzzyEngineOnTheHourlyFrame:
    """The engine is given the outlook's real hours through the session
    window's adapter and returns one score per calendar day."""

    def test_five_days_of_hours_give_five_daily_levels(self):
        levels = assess_outlook_hours_via_fuzzy_engine("brassica-oleracea", _five_march_days(14.0, 70.0, 0.5))

        assert list(levels) == [f"2026-03-{day:02d}" for day in range(6, 11)]
        assert set(levels.values()) <= {"Low", "Moderate", "High", "Critical"}

    def test_warm_humid_hours_give_a_day_above_low_and_cold_dry_hours_do_not(self):
        warm_humid = assess_outlook_hours_via_fuzzy_engine(
            "brassica-oleracea", _five_march_days(24.0, 92.0, 2.0)
        )
        cold_dry = assess_outlook_hours_via_fuzzy_engine("brassica-oleracea", _five_march_days(2.0, 40.0))

        assert any(level != "Low" for level in warm_humid.values())
        assert set(cold_dry.values()) == {"Low"}

    def test_no_hour_is_a_named_failure(self):
        with pytest.raises(FuzzyAssessmentError, match="no hourly observation"):
            assess_outlook_hours_via_fuzzy_engine("brassica-oleracea", [])

    def test_an_hour_the_adapter_refuses_is_a_named_failure(self):
        rows = _five_march_days(14.0, 70.0)
        rows[3]["values"]["relative_humidity_2m"] = None

        with pytest.raises(FuzzyAssessmentError, match="fuzzy engine pipeline failed"):
            assess_outlook_hours_via_fuzzy_engine("brassica-oleracea", rows)

    def test_a_crop_without_a_threat_record_is_a_named_failure(self):
        with pytest.raises(FuzzyAssessmentError, match="no ThreatDefinition"):
            assess_outlook_hours_via_fuzzy_engine("no-such-crop", _five_march_days(14.0, 70.0))


class TestAssessForecastHoursViaPdm:
    def test_the_hourly_form_reaches_the_forecast_route_with_the_hours_as_they_are(self, monkeypatch):
        captured = {}
        answer = {"days": [{"date": "2026-03-06", "level": "moderate", "rules_triggered": []}]}
        _route(monkeypatch, pdm_client, _response(body=answer), captured)
        hours = [
            {"time": "2026-03-06T00:00:00", "temperature_2m": 11.0, "relative_humidity_2m": 78.0, "precipitation": 0.0},
            {"time": "2026-03-06T01:00:00", "temperature_2m": 10.5, "relative_humidity_2m": 80.0, "precipitation": 0.2},
        ]

        result = assess_forecast_hours_via_pdm(
            _Settings(), "brassica-oleracea", [{"date": "2026-03-06", "hours": hours}]
        )

        assert captured["url"] == "http://gatekeeper:8000/api/proxy/pdm/api/v1/risk-assessment/forecast/"
        assert captured["json"] == {
            "crop": "brassica-oleracea",
            "days": [{"date": "2026-03-06", "hours": hours}],
        }
        assert result.days == answer["days"]

    def test_an_http_error_is_a_pdm_call_error(self, monkeypatch):
        _route(monkeypatch, pdm_client, _response(500, text="boom"), {})

        with pytest.raises(PdmHTTPError):
            assess_forecast_hours_via_pdm(_Settings(), "brassica-oleracea", [{"date": "2026-03-06", "hours": []}])


def _ws_ok(settings, latitude, longitude, days):
    return {
        "source": "openweathermap",
        "fetched_at": "2026-09-21T09:30:15+00:00",
        "location": {"lat": latitude, "lon": longitude},
        "days": _MAPPED_ROWS[:days],
    }


def _pdm_levels(settings, crop, rows):
    levels = ["moderate", "low", "high", "low", "low"]
    return PdmForecastResult(
        days=[{"date": r["date"], "level": levels[i], "rules_triggered": []} for i, r in enumerate(rows)]
    )


def _no_hourly_forecast(*args, **kwargs):
    """The default hourly-forecast stand-in for `_build`/`_build_at`: always
    unavailable, so the outlook falls back to the daily forecast unless a
    test overrides it -- the scenario every pre-existing "not evaluated"
    cell exercises."""
    raise WeatherServiceHTTPError(503, "no hourly forecast in this test")


def _build(**overrides):
    kwargs = dict(ws_call=_ws_ok, pdm_call=_pdm_levels, hourly_forecast_call=_no_hourly_forecast)
    kwargs.update(overrides)
    return build_forecast_outlook(_Settings(), 45.0, 10.0, "brassica-oleracea", None, **kwargs)


def _pdm_by_index(levels):
    def _call(settings, crop, rows):
        return PdmForecastResult(
            days=[{"date": r["date"], "level": levels[i], "rules_triggered": []} for i, r in enumerate(rows)]
        )

    return _call


def _pdm_all(level):
    return _pdm_by_index([level] * 5)


def _days(*pairs):
    return [{"date": date, "risk_level": level} for date, level in pairs]


def _raiser(exc):
    def _call(*args, **kwargs):
        raise exc

    return _call


class TestBuildForecastOutlook:
    def test_five_days_carry_the_rule_level_the_source_and_the_fetch_time(self):
        outlook = _build()

        assert isinstance(outlook, ForecastOutlookData)
        assert outlook.source == "openweathermap"
        assert outlook.fetched_at == "2026-09-21T09:30:15+00:00"
        assert outlook.location == {"lat": 45.0, "lon": 10.0}
        assert [d["date"] for d in outlook.days] == [r["date"] for r in _DAYS_RAW]
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert all("comparative_level" not in d for d in outlook.days)
        assert [d["fuzzy_level"] for d in outlook.days] == [FUZZY_NOT_EVALUATED] * 5
        assert outlook.days[0]["temp_max_C"] == 24.0

    def test_a_missing_provider_key_is_named(self):
        exc = WeatherServiceHTTPError(503, "OpenWeatherMap API key is not configured.")

        assert _build(ws_call=_raiser(exc)) == ForecastUnavailable(UNAVAILABLE_NO_KEY)

    @pytest.mark.parametrize(
        "exc",
        [
            WeatherServiceHTTPError(502, "OpenWeatherMap API error: HTTP 500"),
            WeatherServiceHTTPError(503, "bad gateway"),
            WeatherServiceConnectionError("refused"),
            gatekeeper.GatekeeperAuthError("login refused"),
        ],
    )
    def test_any_other_fetch_failure_is_service_unavailable(self, exc):
        assert _build(ws_call=_raiser(exc)) == ForecastUnavailable(UNAVAILABLE_SERVICE)

    def test_no_position_means_no_call_at_all(self):
        def _refuse(*args, **kwargs):
            raise AssertionError("no fetch without a position")

        for latitude, longitude in ((None, 10.0), (45.0, None), (None, None)):
            result = build_forecast_outlook(
                _Settings(), latitude, longitude, "brassica-oleracea", None,
                ws_call=_refuse, pdm_call=_refuse,
            )
            assert result == ForecastUnavailable(UNAVAILABLE_NO_POSITION)

    def test_a_failed_rule_evaluation_leaves_no_table(self):
        result = _build(pdm_call=_raiser(PdmCallError("down")))

        assert result == ForecastUnavailable(UNAVAILABLE_RULES)

    def test_pdm_naming_no_pest_models_is_carried_in_the_reason(self):
        def _no_models(settings, crop, rows):
            return PdmForecastResult(days=[], reason="no_pest_models_configured")

        result = _build(pdm_call=_no_models)

        assert isinstance(result, ForecastUnavailable)
        assert result.reason == f"{UNAVAILABLE_RULES} (no_pest_models_configured)"

    def test_an_incomplete_rule_answer_is_not_papered_over(self):
        def _short(settings, crop, rows):
            return PdmForecastResult(days=[{"date": rows[0]["date"], "level": "low", "rules_triggered": []}])

        assert isinstance(_build(pdm_call=_short), ForecastUnavailable)

    def test_a_failed_hourly_forecast_falls_back_and_fuzzy_cells_say_not_evaluated(self):
        def _refuse(*args, **kwargs):
            raise AssertionError("no fuzzy evaluation on the daily forecast")

        outlook = _build(fuzzy_call=_refuse)

        assert isinstance(outlook, ForecastOutlookData)
        assert [d["fuzzy_level"] for d in outlook.days] == ["not evaluated"] * 5
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert outlook.days_source == "forecast"
        assert outlook.fallback_reason == "hourly forecast unavailable: WeatherServiceHTTPError"


class _Spy:
    """A stand-in for a Weather service call that records how it was called."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _march_hours(extra_days=("2026-03-05", "2026-03-11")):
    """The hourly history the Weather service returns for the March window:
    24 hours a day whose maximum, minimum, humidity and precipitation are the
    `_MARCH_DAYS` values (the maximum at 15:00, the precipitation at 06:00),
    plus a full day on either side that the outlook must not use."""
    rows = []
    for day, tmax, tmin, humidity, rain, _wind in _MARCH_DAYS:
        rows += [
            {
                "timestamp": f"{day}T{hour:02d}:00",
                "values": {
                    "temperature_2m": tmax if hour == 15 else tmin,
                    "relative_humidity_2m": humidity,
                    "precipitation": rain if hour == 6 else 0.0,
                },
            }
            for hour in range(24)
        ]
    for day in extra_days:
        rows += _hours_of(day, 30.0, 99.0, 9.0)
    return rows


def _hourly_answer(rows=None):
    return {
        "source": "openmeteo",
        "location": {"lat": 45.0, "lon": 10.0},
        "data": _march_hours() if rows is None else rows,
    }


_NOW = datetime.datetime(2026, 9, 21, 9, 30, 15, tzinfo=datetime.timezone.utc)
_FUZZY_MARCH = {day[0]: level for day, level in zip(_MARCH_DAYS, ("Low", "Moderate", "Moderate", "Low", "High"))}


def _pdm_hours_levels(settings, crop, days):
    levels = ["moderate", "low", "high", "low", "low"]
    return PdmForecastResult(
        days=[{"date": d["date"], "level": levels[i], "rules_triggered": []} for i, d in enumerate(days)]
    )


def _forecast_hours_of(day, temperature, humidity, wind, precipitation=0.0, rain_hour=6):
    """24 hourly forecast rows for one day, wind included (the hourly
    forecast route is asked for it, unlike the hourly history route)."""
    return [
        {
            "timestamp": f"{day}T{hour:02d}:00",
            "values": {
                "temperature_2m": temperature,
                "relative_humidity_2m": humidity,
                "precipitation": precipitation if hour == rain_hour else 0.0,
                "wind_speed_10m": wind,
            },
        }
        for hour in range(24)
    ]


def _build_at(time_end, forecast=None, hourly=None, hourly_forecast=None, **overrides):
    forecast = forecast or _Spy(_ws_ok(_Settings(), 45.0, 10.0, 5))
    hourly = hourly or _Spy(_hourly_answer())
    hourly_forecast = hourly_forecast or _Spy(WeatherServiceHTTPError(503, "no hourly forecast in this test"))
    kwargs = dict(
        ws_call=forecast, hourly_call=hourly, hourly_forecast_call=hourly_forecast,
        pdm_call=_pdm_levels, pdm_hours_call=_pdm_hours_levels,
        fuzzy_call=lambda crop, observations, pest: dict(_FUZZY_MARCH), time_end=time_end, now=_NOW,
    )
    kwargs.update(overrides)
    outlook = build_forecast_outlook(_Settings(), 45.0, 10.0, "brassica-oleracea", None, **kwargs)
    return outlook, forecast, hourly


class TestTheOutlookFollowsTheSessionsDay:
    def test_a_session_that_ended_two_hours_ago_gets_the_forecast(self):
        outlook, forecast, hourly = _build_at("2026-09-21T07:30:15Z")

        assert isinstance(outlook, ForecastOutlookData)
        assert len(forecast.calls) == 1 and hourly.calls == []
        assert [d["date"] for d in outlook.days] == [r["date"] for r in _DAYS_RAW]
        assert outlook.source == "openweathermap"

    def test_a_failed_hourly_forecast_falls_back_to_the_daily_one_and_the_fuzzy_cells_say_not_evaluated(self):
        hourly_forecast_spy = _Spy(WeatherServiceHTTPError(503, "hourly forecast down"))

        outlook, forecast, hourly = _build_at("2026-09-21T07:30:15Z", hourly_forecast=hourly_forecast_spy)

        assert len(hourly_forecast_spy.calls) == 1
        assert len(forecast.calls) == 1 and hourly.calls == []
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert [d["fuzzy_level"] for d in outlook.days] == ["not evaluated"] * 5
        assert outlook.days_source == "forecast"
        assert outlook.fallback_reason == "hourly forecast unavailable: WeatherServiceHTTPError"

    def test_a_live_sessions_hourly_forecast_is_evaluated_on_both_paths(self):
        tomorrow = _NOW.date() + datetime.timedelta(days=1)
        live_metrics = [(22.0, 65.0, 3.2), (19.0, 70.0, 4.1), (25.0, 55.0, 2.8), (20.0, 72.0, 3.6), (16.0, 80.0, 5.0)]
        live_days = [
            ((tomorrow + datetime.timedelta(days=i)).isoformat(), temperature, humidity, wind)
            for i, (temperature, humidity, wind) in enumerate(live_metrics)
        ]
        rows = []
        for day, temperature, humidity, wind in live_days:
            rows += _forecast_hours_of(day, temperature, humidity, wind, precipitation=0.5)
        hourly_forecast_spy = _Spy({
            "source": "openmeteo",
            "location": {"lat": 45.0, "lon": 10.0},
            "data": rows,
        })
        fuzzy_levels = dict(zip((d for d, *_ in live_days), ("Low", "Moderate", "Moderate", "Low", "High")))

        outlook, forecast, hourly = _build_at(
            "2026-09-21T07:30:15Z",
            hourly_forecast=hourly_forecast_spy,
            fuzzy_call=lambda crop, observations, pest: dict(fuzzy_levels),
        )

        assert len(hourly_forecast_spy.calls) == 1
        assert hourly_forecast_spy.calls[0][-1] == 5
        assert forecast.calls == [] and hourly.calls == []
        assert isinstance(outlook, ForecastOutlookData)
        assert [d["date"] for d in outlook.days] == [
            (tomorrow + datetime.timedelta(days=i)).isoformat() for i in range(5)
        ]
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert [d["fuzzy_level"] for d in outlook.days] == ["Low", "Moderate", "Moderate", "Low", "High"]
        assert all(level != "not evaluated" for level in (d["fuzzy_level"] for d in outlook.days))
        assert [d["wind_speed_ms"] for d in outlook.days] == [w for *_, w in live_days]
        assert outlook.days_source == "forecast"
        assert outlook.fallback_reason is None
        assert outlook.source == "openmeteo"

    def test_a_session_that_ended_late_yesterday_still_gets_the_next_five_days(self):
        tomorrow = _NOW.date() + datetime.timedelta(days=1)
        live_metrics = [(22.0, 65.0, 3.2), (19.0, 70.0, 4.1), (25.0, 55.0, 2.8), (20.0, 72.0, 3.6), (16.0, 80.0, 5.0)]
        live_days = [
            ((tomorrow + datetime.timedelta(days=i)).isoformat(), temperature, humidity, wind)
            for i, (temperature, humidity, wind) in enumerate(live_metrics)
        ]
        rows = []
        for day, temperature, humidity, wind in live_days:
            rows += _forecast_hours_of(day, temperature, humidity, wind, precipitation=0.5)
        hourly_forecast_spy = _Spy({
            "source": "openmeteo",
            "location": {"lat": 45.0, "lon": 10.0},
            "data": rows,
        })
        fuzzy_levels = dict(zip((d for d, *_ in live_days), ("Low", "Moderate", "Moderate", "Low", "High")))

        outlook, forecast, hourly = _build_at(
            "2026-09-20T23:00:00Z",
            hourly_forecast=hourly_forecast_spy,
            fuzzy_call=lambda crop, observations, pest: dict(fuzzy_levels),
        )

        assert forecast.calls == [] and hourly.calls == []
        assert isinstance(outlook, ForecastOutlookData)
        dates = [d["date"] for d in outlook.days]
        assert dates == [(tomorrow + datetime.timedelta(days=i)).isoformat() for i in range(5)]
        assert _NOW.date().isoformat() not in dates
        assert (_NOW.date() - datetime.timedelta(days=1)).isoformat() not in dates
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert [d["fuzzy_level"] for d in outlook.days] == ["Low", "Moderate", "Moderate", "Low", "High"]
        assert outlook.days_source == "forecast"

    def test_the_boundary_is_twenty_four_hours_inclusive(self):
        _, forecast, hourly = _build_at("2026-09-20T09:30:15Z")
        assert len(forecast.calls) == 1 and hourly.calls == []

        _, forecast, hourly = _build_at("2026-09-20T09:30:14Z")
        assert forecast.calls == [] and len(hourly.calls) == 1

    def test_a_session_still_running_or_without_a_usable_end_gets_the_forecast(self):
        for time_end in ("2026-09-22T00:00:00Z", None, "", "not a date"):
            _, forecast, hourly = _build_at(time_end)
            assert len(forecast.calls) == 1 and hourly.calls == [], time_end

    def test_a_session_from_march_gets_the_hourly_weather_of_the_five_days_that_followed_it(self):
        outlook, forecast, hourly = _build_at("2026-03-05T16:20:00Z")

        assert forecast.calls == []
        (call,) = hourly.calls
        assert call[1:] == (45.0, 10.0, "2026-03-06T00:00:00", "2026-03-10T23:00:00")
        assert isinstance(outlook, ForecastOutlookData)
        assert [d["date"] for d in outlook.days] == [day[0] for day in _MARCH_DAYS]
        assert outlook.source == "openmeteo"
        assert outlook.fetched_at == "2026-09-21T09:30:15+00:00"
        assert outlook.location == {"lat": 45.0, "lon": 10.0}
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]
        assert [d["fuzzy_level"] for d in outlook.days] == ["Low", "Moderate", "Moderate", "Low", "High"]

    def test_each_days_row_is_aggregated_from_its_own_hours_and_wind_is_empty(self):
        outlook, _, _ = _build_at("2026-03-05T16:20:00Z")

        for day, expected in zip(outlook.days, _MARCH_DAYS):
            date, tmax, tmin, humidity, rain, _wind = expected
            assert (day["date"], day["temp_max_C"], day["temp_min_C"], day["humidity_pct"], day["precipitation_mm"]) == (
                date, tmax, tmin, humidity, rain,
            )
            assert day["wind_speed_ms"] is None

    def test_the_hours_of_the_days_around_the_window_are_not_used(self):
        seen = {}

        def _pdm_hours(settings, crop, days):
            seen["days"] = days
            return _pdm_hours_levels(settings, crop, days)

        def _fuzzy(crop, observations, pest):
            seen["observations"] = observations
            return {}

        outlook, _, _ = _build_at("2026-03-05T16:20:00Z", pdm_hours_call=_pdm_hours, fuzzy_call=_fuzzy)

        assert [d["date"] for d in seen["days"]] == [day[0] for day in _MARCH_DAYS]
        assert len(seen["observations"]) == 5 * 24
        assert {o["timestamp"][:10] for o in seen["observations"]} == {day[0] for day in _MARCH_DAYS}
        assert outlook.days[0]["temp_max_C"] == 11.0

    def test_the_day_after_is_taken_from_the_end_in_utc(self):
        local_end = "2026-03-05T23:30:00+02:00"

        _, _, hourly = _build_at(local_end)
        assert hourly.calls[0][3] == "2026-03-06T00:00:00"

        _, _, hourly = _build_at("2026-03-06T00:30:00+02:00")
        assert hourly.calls[0][3] == "2026-03-06T00:00:00"

    def test_the_rules_are_evaluated_per_day_over_that_days_hours_by_the_hourly_form(self):
        seen = {}

        def _pdm_hours(settings, crop, days):
            seen["crop"], seen["days"] = crop, days
            return _pdm_hours_levels(settings, crop, days)

        _build_at("2026-03-05T16:20:00Z", pdm_call=_raiser(AssertionError("daily form")), pdm_hours_call=_pdm_hours)

        assert seen["crop"] == "brassica-oleracea"
        first = seen["days"][0]
        assert first["date"] == "2026-03-06"
        assert len(first["hours"]) == 24
        assert first["hours"][0] == {
            "time": "2026-03-06T00:00", "temperature_2m": 2.5, "relative_humidity_2m": 78.0, "precipitation": 0.0,
        }
        assert first["hours"][15]["temperature_2m"] == 11.0

    def test_the_fuzzy_engine_is_given_the_hourly_observations(self):
        seen = {}

        def _fuzzy(crop, observations, pest):
            seen.update(crop=crop, observations=observations, pest=pest)
            return dict(_FUZZY_MARCH)

        _build_at("2026-03-05T16:20:00Z", fuzzy_call=_fuzzy)

        assert seen["crop"] == "brassica-oleracea" and seen["pest"] is None
        assert len(seen["observations"]) == 5 * 24
        assert seen["observations"][0] == {
            "timestamp": "2026-03-06T00:00",
            "values": {"temperature_2m": 2.5, "relative_humidity_2m": 78.0, "precipitation": 0.0},
        }

    def test_only_the_hourly_route_and_the_rule_route_are_called_over_http(self, monkeypatch):
        posted = []

        def _fake_post(url, headers=None, json=None, timeout=None):
            if url.endswith("/api/login/"):
                return _login_ok()
            posted.append((url, json))
            if "/history/hourly/" in url:
                return _response(body=_hourly_answer())
            if "/risk-assessment/forecast/" in url:
                return _response(body={
                    "days": [{"date": d["date"], "level": "low", "rules_triggered": []} for d in json["days"]]
                })
            raise AssertionError(f"unexpected call: {url}")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)
        monkeypatch.setattr(ws_client.requests, "post", _fake_post)
        monkeypatch.setattr(pdm_client.requests, "post", _fake_post)

        outlook = build_forecast_outlook(
            _Settings(), 45.0, 10.0, "brassica-oleracea", None, time_end="2026-03-05T16:20:00Z", now=_NOW
        )

        assert isinstance(outlook, ForecastOutlookData)
        (hourly_url, hourly_body), (pdm_url, pdm_body) = posted
        assert hourly_url == "http://gatekeeper:8000/api/proxy/weather_data/api/v1/history/hourly/"
        assert (hourly_body["start"], hourly_body["end"]) == ("2026-03-06", "2026-03-10")
        assert hourly_body["variables"] == ["temperature_2m", "relative_humidity_2m", "precipitation"]
        assert pdm_url == "http://gatekeeper:8000/api/proxy/pdm/api/v1/risk-assessment/forecast/"
        assert [d["date"] for d in pdm_body["days"]] == [day[0] for day in _MARCH_DAYS]
        assert all(len(d["hours"]) == 24 and "temp_max" not in d for d in pdm_body["days"])
        assert not [url for url, _ in posted if "/forecast/daily/" in url or "/history/daily/" in url]

    def test_the_fuzzy_levels_are_the_engines_daily_scores_on_the_hourly_frame(self):
        hours = [
            row
            for day in range(6, 11)
            for row in _hours_of(f"2026-03-{day:02d}", 24.0, 92.0, 2.0)
        ]
        outlook, _, _ = _build_at(
            "2026-03-05T16:20:00Z", hourly=_Spy(_hourly_answer(hours)), fuzzy_call=assess_outlook_hours_via_fuzzy_engine,
        )

        directly = assess_outlook_hours_via_fuzzy_engine("brassica-oleracea", hours)
        assert [d["fuzzy_level"] for d in outlook.days] == [directly[d["date"]] for d in outlook.days]
        assert any(d["fuzzy_level"] != "Low" for d in outlook.days)

    def test_a_failed_fuzzy_evaluation_leaves_its_cells_empty_and_keeps_the_rule_table(self):
        outlook, _, _ = _build_at("2026-03-05T16:20:00Z", fuzzy_call=_raiser(FuzzyAssessmentError("engine down")))

        assert isinstance(outlook, ForecastOutlookData)
        assert [d["fuzzy_level"] for d in outlook.days] == [None] * 5
        assert [d["risk_level"] for d in outlook.days] == ["Moderate", "Low", "High", "Low", "Low"]

    def test_an_hour_without_all_three_values_is_left_out_and_a_day_without_hours_has_no_row(self):
        rows = _march_hours()
        for row in rows:
            if row["timestamp"].startswith("2026-03-08"):
                row["values"]["precipitation"] = None
            if row["timestamp"] == "2026-03-06T15:00":
                row["values"]["temperature_2m"] = None
        seen = {}

        def _fuzzy(crop, observations, pest):
            seen["observations"] = observations
            return {}

        outlook, _, _ = _build_at(
            "2026-03-05T16:20:00Z", hourly=_Spy(_hourly_answer(rows)), fuzzy_call=_fuzzy,
            pdm_hours_call=lambda settings, crop, days: PdmForecastResult(
                days=[{"date": d["date"], "level": "low", "rules_triggered": []} for d in days]
            ),
        )

        assert [d["date"] for d in outlook.days] == ["2026-03-06", "2026-03-07", "2026-03-09", "2026-03-10"]
        assert outlook.days[0]["temp_max_C"] == 2.5
        assert len(seen["observations"]) == 4 * 24 - 1

    def test_no_complete_hour_at_all_is_service_unavailable_not_an_empty_table(self):
        rows = _march_hours()
        for row in rows:
            row["values"]["relative_humidity_2m"] = None

        outlook, _, _ = _build_at("2026-03-05T16:20:00Z", hourly=_Spy(_hourly_answer(rows)))

        assert outlook == ForecastUnavailable(UNAVAILABLE_SERVICE)

    def test_a_response_without_a_data_list_is_service_unavailable(self):
        outlook, _, _ = _build_at("2026-03-05T16:20:00Z", hourly=_Spy({"source": "openmeteo"}))

        assert outlook == ForecastUnavailable(UNAVAILABLE_SERVICE)

    def test_a_fetch_failure_keeps_the_reason_and_the_recommendation(self):
        for exc in (
            WeatherServiceHTTPError(502, "upstream failed"),
            WeatherServiceConnectionError("refused"),
            gatekeeper.GatekeeperAuthError("login refused"),
        ):
            outlook, forecast, hourly = _build_at("2026-03-05T16:20:00Z", hourly=_Spy(exc))

            assert outlook == ForecastUnavailable(UNAVAILABLE_SERVICE)
            assert forecast.calls == []
            payload = _payload(outlook)["forecast_outlook"]
            assert set(payload) == {"unavailable_reason", "recommendation"}
            assert payload["unavailable_reason"] == UNAVAILABLE_SERVICE
            assert payload["recommendation"].startswith(RECOMMENDATION_MARKER)

    def test_a_failed_rule_evaluation_of_the_hours_leaves_no_table(self):
        outlook, _, _ = _build_at("2026-03-05T16:20:00Z", pdm_hours_call=_raiser(PdmCallError("down")))

        assert outlook == ForecastUnavailable(UNAVAILABLE_RULES)

    def test_the_hourly_outlook_renders_the_march_dates_both_levels_and_its_source(self):
        outlook, _, _ = _build_at("2026-03-05T16:20:00Z")

        lines = _layout_lines(_payload(outlook))

        rows = [line for line in lines if line.startswith("2026-03-")]
        assert [row.split()[0] for row in rows] == [day[0] for day in _MARCH_DAYS]
        assert rows[0] == "2026-03-06 11.0 2.5 78.0 0.0 n/a Moderate Low"
        assert rows[4] == "2026-03-10 8.0 0.5 88.0 6.0 n/a Low High"
        assert "Comparative" not in "\n".join(_table_lines(lines))
        assert "Risk outlook: the five days after the session" in lines
        assert "Source openmeteo" in lines


def _risk_input():
    return RiskInput(
        session_id="sess-fc",
        infestation_score=0.8,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )


def _envelope(designation_status="designated"):
    return ReportingEnvelope(
        session_id="sess-fc",
        best_image_id="img-1",
        image_count=3,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status=designation_status,
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
        ),
    )


def _row(path, fused_verdict):
    return RiskAssessmentRecord(
        session_id="sess-fc", path=path, status="present", level="moderate",
        provenance_json=json.dumps({}), fused_verdict=fused_verdict, fusion_rule_version="v1",
        engine_identity="engine", engine_version="1", weather_window="w",
    )


def _payload(forecast, designation_status="designated"):
    return compose_report_payload(
        risk_input=_risk_input(),
        envelope=_envelope(designation_status),
        pdm_assessment=_row("pdm", "High"),
        fuzzy_assessment=None,
        contract_version=CONTRACT_VERSION,
        forecast=forecast,
    )


class TestForecastOutlookInThePayload:
    def test_a_forecast_becomes_the_outlook_with_the_recommendation(self):
        payload = _payload(_build())

        outlook = payload["forecast_outlook"]
        assert set(outlook) == {
            "source", "fetched_at", "location", "days", "days_source", "outlook_summary", "recommendation",
            "fallback_reason",
        }
        assert outlook["source"] == "openweathermap"
        assert outlook["fetched_at"] == "2026-09-21T09:30:15+00:00"
        assert len(outlook["days"]) == 5
        assert set(outlook["days"][0]) == {
            "date", "temp_max_C", "temp_min_C", "humidity_pct", "precipitation_mm",
            "wind_speed_ms", "risk_level", "fuzzy_level",
        }
        assert outlook["recommendation"].startswith(RECOMMENDATION_MARKER)
        assert outlook["outlook_summary"] == (
            "The weather rules give High risk on 1 of the 5 days (24 September), "
            "Moderate on the others."
        )

    def test_a_missing_forecast_is_the_recommendation_and_the_reason(self):
        payload = _payload(ForecastUnavailable(UNAVAILABLE_NO_KEY))

        outlook = payload["forecast_outlook"]
        assert set(outlook) == {"unavailable_reason", "recommendation"}
        assert outlook["unavailable_reason"] == UNAVAILABLE_NO_KEY
        assert outlook["recommendation"].startswith(RECOMMENDATION_MARKER)
        assert "days" not in outlook

    def test_no_forecast_attempted_keeps_the_recommendation_only_outlook(self):
        outlook = _payload(None)["forecast_outlook"]

        assert set(outlook) == {"recommendation"}

    def test_a_stock_model_session_never_carries_an_outlook(self):
        assert _payload(_build(), designation_status="stub")["forecast_outlook"] is None

    @pytest.mark.parametrize("level", ["low", "moderate", "high"])
    def test_five_equal_days_summarise_as_one_level(self, level):
        equal = _build(pdm_call=_pdm_all(level))

        summary = _payload(equal)["forecast_outlook"]["outlook_summary"]

        assert summary == f"The weather rules give {level.capitalize()} risk on all 5 days."
        assert " or " not in summary and "highest" not in summary

    def test_two_high_days_are_named_by_date_and_the_others_by_their_level(self):
        mixed = _build(pdm_call=_pdm_by_index(["high", "high", "moderate", "moderate", "moderate"]))

        summary = _payload(mixed)["forecast_outlook"]["outlook_summary"]

        assert summary == (
            "The weather rules give High risk on 2 of the 5 days (22 and 23 September), "
            "Moderate on the others."
        )

    def test_the_summary_never_names_a_level_no_day_carries(self):
        mixed = _build(pdm_call=_pdm_by_index(["moderate", "low", "low", "low", "low"]))

        summary = _payload(mixed)["forecast_outlook"]["outlook_summary"]

        assert summary == (
            "The weather rules give Moderate risk on 1 of the 5 days (22 September), Low on the others."
        )
        assert "High" not in summary

    def test_dates_join_with_commas_from_three_and_carry_their_month_across_months(self):
        assert compose_outlook_summary(_days(
            ["2026-09-22", "High"], ["2026-09-23", "High"], ["2026-09-24", "High"], ["2026-09-25", "Low"],
        )) == "The weather rules give High risk on 3 of the 4 days (22, 23, 24 September), Low on the others."
        assert compose_outlook_summary(_days(
            ["2026-09-30", "High"], ["2026-10-01", "High"], ["2026-10-02", "Low"],
        )) == "The weather rules give High risk on 2 of the 3 days (30 September and 1 October), Low on the others."

    def test_no_day_rows_give_no_summary(self):
        assert compose_outlook_summary([]) == ""


def _layout_lines(payload) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "outlook.pdf")
        render_pest_session_pdf(payload, lambda variant: None, path)
        out = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, check=True).stdout
    return [" ".join(line.split()) for line in out.splitlines()]


def _table_lines(lines: list) -> list:
    """The outlook table's header and day rows out of the layout lines."""
    return [line for line in lines if line.startswith(("Date Max", "2026-"))]


class TestOutlookRendersInThePdf:
    def test_five_days_render_five_rows_with_both_levels_the_source_and_the_time(self):
        lines = _layout_lines(_payload(_build()))

        assert (
            "Date Max °C Min °C Humidity % Precip mm Wind m/s Risk (weather rules) Risk (fuzzy path)" in lines
        )
        assert "Comparative" not in "\n".join(_table_lines(lines))
        rows = [line for line in lines if line.startswith("2026-09-2")]
        assert len(rows) == 5
        assert rows[0] == "2026-09-22 24.0 12.0 72.0 0.4 3.1 Moderate not evaluated"
        assert rows[2] == "2026-09-24 29.0 15.0 55.0 0.0 2.0 High not evaluated"
        assert "Forecast risk outlook: the next 5 days" in lines
        assert "Source openweathermap" in lines
        assert "Data fetched at 2026-09-21T09:30:15+00:00" in lines
        assert any(line.startswith(RECOMMENDATION_MARKER) for line in lines)

    def test_five_equal_days_render_the_one_level_sentence_and_the_header(self):
        lines = _layout_lines(_payload(_build(pdm_call=_pdm_all("moderate"))))
        text = " ".join(lines)

        assert "Outlook Summary: The weather rules give Moderate risk on all 5 days." in text
        assert "Risk (weather rules)" in text and "Risk (fuzzy path)" in text
        assert "Comparative" not in "\n".join(_table_lines(lines))

    def test_a_day_with_no_fuzzy_level_prints_n_a(self):
        outlook = _build()
        days = [{**d, "fuzzy_level": None} if i == 1 else d for i, d in enumerate(outlook.days)]

        lines = _layout_lines(
            _payload(ForecastOutlookData(outlook.source, outlook.fetched_at, outlook.location, days, outlook.days_source))
        )

        rows = [line for line in lines if line.startswith("2026-09-2")]
        assert rows[1] == "2026-09-23 18.5 9.0 80.0 2.2 4.0 Low n/a"

    def test_two_high_days_render_the_mixed_sentence(self):
        mixed = _build(pdm_call=_pdm_by_index(["high", "high", "moderate", "moderate", "moderate"]))

        text = " ".join(_layout_lines(_payload(mixed)))

        assert (
            "Outlook Summary: The weather rules give High risk on 2 of the 5 days "
            "(22 and 23 September), Moderate on the others."
        ) in text

    def test_a_missing_forecast_renders_no_table_but_keeps_the_recommendation(self):
        lines = _layout_lines(_payload(ForecastUnavailable(UNAVAILABLE_SERVICE)))
        text = "\n".join(lines)

        assert "Wind m/s" not in text
        assert "Risk (weather rules)" not in text and "Risk (fuzzy path)" not in text
        assert any(line.startswith(RECOMMENDATION_MARKER) for line in lines)

    def test_a_reason_alone_renders_no_empty_section(self):
        payload = _payload(ForecastUnavailable(UNAVAILABLE_SERVICE))
        del payload["forecast_outlook"]["recommendation"]

        lines = _layout_lines(payload)

        assert "Forecast risk outlook: the next 5 days" not in lines
        assert "Risk outlook: the five days after the session" not in lines


class _RecordingForecast:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, settings, latitude, longitude, crop, pest, time_end=None):
        self.calls.append((latitude, longitude, crop, pest, time_end))
        return self.result


def _submission(session_id, designation_status="designated"):
    risk_input = _risk_input().model_copy(update={"session_id": session_id})
    envelope = _envelope(designation_status).model_copy(update={"session_id": session_id})
    return SessionSubmission(
        contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope
    )


def _pdm_ok(settings, risk_input):
    from pdm_client import PdmAssessmentResult

    return PdmAssessmentResult(
        level="High",
        details="Weather-rule-based risk assessment.",
        provenance={"weather_rule_based": True, "weather_rule_level_raw": "moderate",
                    "ruleset_version": "1.0.0", "pdm_app_version": "0.2.0"},
    )


def _fuzzy_ok(settings, risk_input, **kwargs):
    from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary

    return FuzzyAssessmentResult(
        dominant_band="moderate",
        daily_scores=[{"date": "2026-01-01", "value": 40.0, "risk_level": "moderate"}],
        engine_version="2.0.0", threat_definition_version="1.0.0", dossier_version="v2",
        weather_summary=WeatherWindowSummary(
            temperature_min_C=11.0, temperature_max_C=22.0, relative_humidity_avg_pct=58.0, source="openmeteo"
        ),
    )


class TestForecastIsFetchedOncePerSessionAndNeverStored:
    def test_the_forecast_reaches_the_pushed_payload_and_is_fetched_once_across_report_retries(self, db):
        forecast = _RecordingForecast(_build())
        pushed = []

        def _report(settings, payload):
            pushed.append(payload)
            if len(pushed) == 1:
                raise ReportingConnectionError("Reporting service down")
            return ReportPushResult(report_id="report-1")

        intake_submission(db, _submission("sess-once"))
        result = advance_session(
            db, "sess-once", settings=_Settings(), pdm_call=_pdm_ok, fuzzy_call=_fuzzy_ok,
            report_call=_report, forecast_call=forecast,
        )

        assert result.state == SessionState.REPORTED.value
        assert len(pushed) == 2
        assert forecast.calls == [(45.0, 10.0, "brassica-oleracea", None, "2026-01-02T00:00:00Z")]
        for payload in pushed:
            assert len(payload["forecast_outlook"]["days"]) == 5

    def test_the_stored_assessment_records_carry_no_forecast(self, db):
        forecast = _RecordingForecast(_build())

        intake_submission(db, _submission("sess-stored"))
        advance_session(
            db, "sess-stored", settings=_Settings(), pdm_call=_pdm_ok, fuzzy_call=_fuzzy_ok,
            report_call=lambda settings, payload: ReportPushResult(report_id="r"), forecast_call=forecast,
        )

        rows = db.query(RiskAssessmentRecord).filter_by(session_id="sess-stored").all()
        assert {row.path for row in rows} == {"pdm", "fuzzy"}
        forecast_dates = [day["date"] for day in _DAYS_RAW]
        non_timestamp_columns = [
            column for column in RiskAssessmentRecord.__table__.columns if not isinstance(column.type, DateTime)
        ]
        for row in rows:
            stored = " ".join(str(getattr(row, column.name)) for column in non_timestamp_columns)
            assert "forecast" not in stored.lower()
            for date in forecast_dates:
                assert date not in stored

    def test_a_stock_model_session_fetches_no_forecast(self, db):
        forecast = _RecordingForecast(_build())
        pushed = []

        def _report(settings, payload):
            pushed.append(payload)
            return ReportPushResult(report_id="r")

        intake_submission(db, _submission("sess-stock", designation_status="stub"))
        advance_session(
            db, "sess-stock", settings=_Settings(), pdm_call=_pdm_ok, fuzzy_call=_fuzzy_ok,
            report_call=_report, forecast_call=forecast,
        )

        assert forecast.calls == []
        assert pushed[0]["forecast_outlook"] is None

    def test_an_unavailable_forecast_still_reports_and_records_the_reason(self, db):
        forecast = _RecordingForecast(ForecastUnavailable(UNAVAILABLE_NO_KEY))
        pushed = []

        def _report(settings, payload):
            pushed.append(payload)
            return ReportPushResult(report_id="r")

        intake_submission(db, _submission("sess-nokey"))
        result = advance_session(
            db, "sess-nokey", settings=_Settings(), pdm_call=_pdm_ok, fuzzy_call=_fuzzy_ok,
            report_call=_report, forecast_call=forecast,
        )

        assert result.state == SessionState.REPORTED.value
        assert pushed[0]["forecast_outlook"]["unavailable_reason"] == UNAVAILABLE_NO_KEY
        assert pushed[0]["forecast_outlook"]["recommendation"].startswith(RECOMMENDATION_MARKER)
