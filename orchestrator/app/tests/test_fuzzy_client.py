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
In-process fuzzy engine invocation wiring (the Weather service fetch -> adapter ->
engine -> dominant band). The Weather-service fetch itself is stubbed at the
`fuzzy_client.fetch_hourly_history` boundary (test_ws_client.py already
covers that call's own construction/error handling) so these tests focus
on the pipeline wiring: vision-blindness and failure isolation.
"""
from __future__ import annotations

import datetime
import json
import os

import pytest

import fuzzy_client
import report_content
from contract import GrowthStage, RiskInput
from fuzzy_client import FuzzyAssessmentError, assess_via_fuzzy_engine

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "fuzzy_evaluation_vectors")


class _Settings:
    pass


def _load_ws_response(name: str) -> dict:
    with open(os.path.join(_FIXTURES_DIR, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)["ws_response"]


def _risk_input(**overrides) -> RiskInput:
    fields = dict(
        session_id="sess-1",
        infestation_score=0.05,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-06-01T00:00:00Z",
        time_end="2026-06-05T23:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    fields.update(overrides)
    return RiskInput(**fields)


class TestAssessViaFuzzyEngineIsVisionBlind:
    def test_infestation_score_never_reaches_the_ws_fetch(self, monkeypatch):
        captured = {}

        def _fake_fetch(settings, latitude, longitude, time_start, time_end):
            captured["kwargs"] = {
                "latitude": latitude,
                "longitude": longitude,
                "time_start": time_start,
                "time_end": time_end,
            }
            return _load_ws_response("favorable_dry_low_infestation")

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input(infestation_score=0.99))

        assert set(captured["kwargs"]) == {"latitude", "longitude", "time_start", "time_end"}
        assert result.dominant_band in ("low", "moderate", "high", "critical")
        assert result.engine_version
        assert result.threat_definition_version
        assert result.dossier_version


class TestAssessViaFuzzyEngineFailureModes:
    """Any failure anywhere in this pipeline raises
    FuzzyAssessmentError uniformly, never fabricating a result."""

    def test_missing_latitude_raises(self):
        with pytest.raises(FuzzyAssessmentError):
            assess_via_fuzzy_engine(_Settings(), _risk_input(latitude=None))

    def test_missing_longitude_raises(self):
        with pytest.raises(FuzzyAssessmentError):
            assess_via_fuzzy_engine(_Settings(), _risk_input(longitude=None))

    def test_ws_fetch_failure_wraps_as_fuzzy_assessment_error(self, monkeypatch):
        def _fake_fetch(*args, **kwargs):
            raise RuntimeError("Weather service down (test)")

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        with pytest.raises(FuzzyAssessmentError):
            assess_via_fuzzy_engine(_Settings(), _risk_input())

    def test_unknown_crop_raises_fuzzy_assessment_error(self, monkeypatch):
        def _fake_fetch(*args, **kwargs):
            return _load_ws_response("favorable_dry_low_infestation")

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        with pytest.raises(FuzzyAssessmentError):
            assess_via_fuzzy_engine(_Settings(), _risk_input(crop="unknown-crop"))

    def test_malformed_ws_response_wraps_as_fuzzy_assessment_error(self, monkeypatch):
        def _fake_fetch(*args, **kwargs):
            return {"source": "openmeteo", "location": {"lat": 1.0, "lon": 2.0}, "data": []}

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        with pytest.raises(FuzzyAssessmentError):
            assess_via_fuzzy_engine(_Settings(), _risk_input())


class TestAssessViaFuzzyEngineSuccess:
    def test_success_records_engine_and_threat_versions(self, monkeypatch):
        def _fake_fetch(*args, **kwargs):
            return _load_ws_response("favorable_dry_low_infestation")

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input())

        assert result.threat_definition_version == "1.0.0"
        assert result.dossier_version == "v2"
        assert result.dominant_band == "moderate"
        assert len(result.daily_scores) == 5

    def test_cucumber_uses_myzupe_by_default_and_aphigo_when_the_pair_is_given(self, monkeypatch):
        selected = []
        original = fuzzy_client.resolve_threat_record

        def _record(crop, pest=None):
            selected.append((crop, pest))
            return original(crop, pest)

        monkeypatch.setattr(fuzzy_client, "resolve_threat_record", _record)
        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", lambda *args, **kwargs: _load_ws_response("favorable_dry_low_infestation"))
        cucumber = _risk_input(crop="cucumis-sativus")
        assess_via_fuzzy_engine(_Settings(), cucumber)
        assess_via_fuzzy_engine(_Settings(), cucumber, pest="APHIGO")
        assert selected == [("cucumis-sativus", None), ("cucumis-sativus", "APHIGO")]


class TestAssessViaFuzzyEngineWeatherSummary:
    """The
    window-weather summary is computed from the SAME `WeatherDataFrame`
    the engine consumed -- pinned against the fixture's own known values
    (not invented), so a wrong aggregation cannot pass."""

    def test_weather_summary_pins_exact_aggregates_from_known_ws_response(self, monkeypatch):
        ws_response = _load_ws_response("favorable_dry_low_infestation")

        def _fake_fetch(*args, **kwargs):
            return ws_response

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input())

        temperatures = [obs["values"]["temperature_2m"] for obs in ws_response["data"]]
        humidities = [obs["values"]["relative_humidity_2m"] for obs in ws_response["data"]]

        assert result.weather_summary.temperature_min_C == pytest.approx(min(temperatures))
        assert result.weather_summary.temperature_max_C == pytest.approx(max(temperatures))
        assert result.weather_summary.relative_humidity_avg_pct == pytest.approx(
            sum(humidities) / len(humidities)
        )
        assert result.weather_summary.source == ws_response["source"]

    def test_weather_summary_rounds_repeating_decimal_aggregates_to_one_decimal(self, monkeypatch):
        """The summary object rounds
        temp min/max/humidity avg to one decimal at construction, before
        the composer or renderer ever see them -- derived from the known
        fixture (not invented) with values perturbed to carry repeating
        decimals, e.g. 89.66666666666667 -> 89.7."""
        ws_response = _load_ws_response("favorable_dry_low_infestation")
        for obs in ws_response["data"]:
            obs["values"]["relative_humidity_2m"] = 89.66666666666667
        ws_response["data"][0]["values"]["temperature_2m"] = 12.133333333333333
        ws_response["data"][1]["values"]["temperature_2m"] = 25.866666666666667

        def _fake_fetch(*args, **kwargs):
            return ws_response

        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", _fake_fetch)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input())

        assert result.weather_summary.temperature_min_C == 12.1
        assert result.weather_summary.temperature_max_C == 25.9
        assert result.weather_summary.relative_humidity_avg_pct == 89.7


class TestFuzzyPathResolvesThroughTheSharedFunction:
    """The fuzzy path and the report
    content read ONE catalogue record, resolved by the same function."""

    @staticmethod
    def _spy(monkeypatch):
        import report_content

        calls = []
        real = report_content.resolve_threat_record

        def _spy(crop, pest=None):
            calls.append((crop, pest))
            return real(crop, pest)

        monkeypatch.setattr(fuzzy_client, "resolve_threat_record", _spy)
        monkeypatch.setattr(
            fuzzy_client,
            "fetch_hourly_history",
            lambda *a, **k: _load_ws_response("favorable_dry_low_infestation"),
        )
        return calls

    def test_a_session_with_a_pest_resolves_by_the_pair(self, monkeypatch):
        calls = self._spy(monkeypatch)
        result = assess_via_fuzzy_engine(_Settings(), _risk_input(), pest="BRVCBR")
        assert calls == [("brassica-oleracea", "BRVCBR")]
        assert result.threat_definition_version == "1.0.0"

    def test_a_session_without_a_pest_resolves_by_crop_as_before(self, monkeypatch):
        calls = self._spy(monkeypatch)
        assess_via_fuzzy_engine(_Settings(), _risk_input())
        assert calls == [("brassica-oleracea", None)]

    def test_the_fuzzy_record_is_the_record_the_report_names(self, monkeypatch):
        import report_content

        self._spy(monkeypatch)
        result = assess_via_fuzzy_engine(_Settings(), _risk_input(), pest="BRVCBR")
        record = report_content.resolve_threat_record("brassica-oleracea", "BRVCBR")
        identity = report_content.resolve_pest_identity("brassica-oleracea", "BRVCBR")
        assert result.threat_definition_version == record.version
        assert result.dossier_version == record.dossier_version
        assert identity.eppo_code == record.eppo_code == "BRVCBR"

    def test_an_unknown_pair_fails_the_fuzzy_path_by_name(self, monkeypatch):
        self._spy(monkeypatch)
        with pytest.raises(FuzzyAssessmentError, match="MYZUPE"):
            assess_via_fuzzy_engine(_Settings(), _risk_input(), pest="MYZUPE")


class TestAssessViaFuzzyEngineExplanation:
    """The
    explanation's per-day aggregates, sub-model results and rule matches,
    computed from the SAME frame and threat record the engine consumed --
    pinned against a constructed 24-hour, single-day frame so the engine
    yields exactly one daily score, and against the real
    `HumidityStreak`/`LeafWetnessRH` sub-models the engine itself calls
    (never re-derived by hand: a wrong aggregation would then still pass).

    `LeafWetnessRH` gives 18.0 h for RH >= 85 with >= 1 mm rain (14 base + 4 rain), and BRVCBR's
    "low" rule matches this frame, so 1 of 1 days matches.
    """

    def _single_day_ws_response(self):
        data = []
        base = datetime.datetime(2026, 3, 5, 0, 0, 0)
        for hour in range(24):
            data.append(
                {
                    "timestamp": (base + datetime.timedelta(hours=hour)).isoformat() + "Z",
                    "values": {
                        "temperature_2m": 5.6,
                        "relative_humidity_2m": 92.0,
                        "precipitation": 1.2 / 24,
                    },
                }
            )
        return {"source": "openmeteo", "location": {"lat": 45.0, "lon": 10.0}, "data": data}

    def test_a_24_hour_frame_gives_the_explanations_own_facts_and_the_engines_own_score(self, monkeypatch):
        ws_response = self._single_day_ws_response()
        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", lambda *a, **k: ws_response)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input())

        assert len(result.daily_scores) == 1
        explanation = result.explanation
        assert len(explanation["days"]) == 1
        day = explanation["days"][0]
        assert day["temperature_avg_C"] == pytest.approx(5.6)
        assert day["humidity_max_pct"] == pytest.approx(92.0)
        assert day["rain_mm"] == pytest.approx(1.2)
        assert day["matched_rule_levels"] == ["low"]
        assert explanation["humidity_streak_days"] == 1
        assert explanation["min_streak_days"] == 2
        assert explanation["wettest_day_wet_hours"] == pytest.approx(18.0)
        assert explanation["min_wetness_hours_high"] == pytest.approx(24.0)
        assert explanation["rule_count"] == 3
        assert explanation["dominant_band"] == result.dominant_band
        assert explanation["score_low"] == explanation["score_high"] == result.daily_scores[0]["value"]

    def test_the_composed_paragraph_names_every_fact_in_order_with_one_decimal(self, monkeypatch):
        ws_response = self._single_day_ws_response()
        monkeypatch.setattr(fuzzy_client, "fetch_hourly_history", lambda *a, **k: ws_response)

        result = assess_via_fuzzy_engine(_Settings(), _risk_input())
        paragraph = report_content.compose_fuzzy_details(result.explanation)

        score = result.daily_scores[0]["value"]
        assert paragraph == (
            "Fuzzy model (Mamdani) over the same window: daily average temperature 5.6 °C, "
            "daily maximum relative humidity 92.0 %, 1.2 mm of rain in total. Humid days in a row "
            "(relative humidity of at least 80 %): 1 of the 2 the model requires. Leaf wetness: "
            "18.0 hours on the wettest day, 24.0 required for High. 1 of the 1 days had values "
            "within the ranges of at least one of the model's 3 rules (Low). The model's daily "
            f"score is {score:.1f} of 100; the resulting band is {result.dominant_band.capitalize()}."
        )

    def test_a_failed_fuzzy_path_composes_the_not_evaluated_sentence(self):
        reason = "session coordinates are absent; the fuzzy path requires lat/lon for the Weather service history call"
        assert report_content.compose_fuzzy_not_evaluated(reason) == (
            f"The fuzzy model was not evaluated: {reason}."
        )
