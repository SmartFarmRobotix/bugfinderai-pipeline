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

"""Fixed evaluation vectors.

Two things are pinned in `app/tests/fixtures/fuzzy_evaluation_vectors/`:

1. Raw environmental pressure. Reproduced and asserted exactly
   below, every run.

2. Each fixture's low/high `infestation_score` and its expected fused
   verdict, computed by hand from the v1 fusion rule; `TestFusedVerdictReproduction` below calls the fusion
   function (`app/fusion.py`) on each fixture and asserts that its output equals `expected_verdict`.
"""
from __future__ import annotations

import json
import os

import pytest

from agstack_pnd.foundation.types import RiskLevel
from agstack_pnd.models.disease.fuzzy_mamdani import FuzzyMamdaniRisk
from fusion import dominant_band, fuse, fuse_principled
from fuzzy_threats import get_threat
from weather_adapter import ws_history_to_weather_dataframe

_FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "fuzzy_evaluation_vectors"
)
_VECTOR_NAMES = [
    "favorable_dry_low_infestation",
    "favorable_dry_high_infestation",
    "unfavorable_cold_wet",
]


def _load_fixture(name: str) -> dict:
    with open(os.path.join(_FIXTURES_DIR, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _run_engine(fixture: dict):
    wdf = ws_history_to_weather_dataframe(fixture["ws_response"])
    threat = get_threat(fixture["threat_crop"], fixture["threat_scientific_name"])
    return FuzzyMamdaniRisk().calculate(wdf, threat=threat)


class TestRawEnvironmentalPressureReproduction:
    """Runs NOW: adapter + engine + ThreatDefinition, no fusion involved."""

    @pytest.mark.parametrize("vector_name", _VECTOR_NAMES)
    def test_daily_scores_reproduce_exactly(self, vector_name: str) -> None:
        fixture = _load_fixture(vector_name)
        result = _run_engine(fixture)

        expected = fixture["expected_environmental_pressure"]["daily_scores"]
        assert len(result.daily_scores) == len(expected)
        for actual_score, expected_score in zip(result.daily_scores, expected):
            assert str(actual_score.date) == expected_score["date"]
            assert actual_score.value == pytest.approx(expected_score["value"])
            assert actual_score.risk_level.value == expected_score["risk_level"]

    def test_engine_version_pinned_in_fixtures_matches_installed_engine(self) -> None:
        for vector_name in _VECTOR_NAMES:
            fixture = _load_fixture(vector_name)
            assert fixture["engine_version"] == FuzzyMamdaniRisk.metadata.version

    def test_low_and_high_infestation_variants_share_identical_raw_pressure(self) -> None:
        """The raw Mamdani stage is weather-only by API and never sees
        the session score -- the low/high fixture pair's weather windows
        are byte-identical, so their raw environmental pressure must be
        identical too, regardless of infestation_score."""
        low = _load_fixture("favorable_dry_low_infestation")
        high = _load_fixture("favorable_dry_high_infestation")

        assert low["ws_response"] == high["ws_response"]
        assert (
            low["expected_environmental_pressure"]
            == high["expected_environmental_pressure"]
        )
        assert low["infestation_score"] != high["infestation_score"]

    def test_favorable_and_unfavorable_vectors_are_distinguishable(self) -> None:
        favorable = _run_engine(_load_fixture("favorable_dry_low_infestation"))
        unfavorable = _run_engine(_load_fixture("unfavorable_cold_wet"))

        assert favorable.summary["max_score"] > unfavorable.summary["max_score"]
        assert any(
            s.risk_level != RiskLevel.LOW for s in favorable.daily_scores
        ), "favorable fixture should reach above LOW at least one day"
        assert all(
            s.risk_level == RiskLevel.LOW for s in unfavorable.daily_scores
        ), "unfavorable fixture should stay LOW throughout"

    def test_unfavorable_fixture_score_is_rain_invariant(self) -> None:
        """This ThreatDefinition's rain_min=0.0 on every rule means recorded precipitation never
        changes the score. Confirms the unfavorable_cold_wet fixture's
        rain hours are inert, not silently ignored by a bug."""
        fixture = _load_fixture("unfavorable_cold_wet")
        ws_response = fixture["ws_response"]

        zero_rain_response = json.loads(json.dumps(ws_response))
        for obs in zero_rain_response["data"]:
            obs["values"]["precipitation"] = 0.0

        threat = get_threat(fixture["threat_crop"], fixture["threat_scientific_name"])
        with_rain = FuzzyMamdaniRisk().calculate(
            ws_history_to_weather_dataframe(ws_response), threat=threat
        )
        without_rain = FuzzyMamdaniRisk().calculate(
            ws_history_to_weather_dataframe(zero_rain_response), threat=threat
        )

        for a, b in zip(with_rain.daily_scores, without_rain.daily_scores):
            assert a.value == pytest.approx(b.value)


class TestFusedVerdictReproduction:
    """The shared fusion function reproduces each low/high fixture's hand-authored `expected_verdict`
    by running the real pipeline (adapter + engine + `dominant_band` + `fuse`) and asserting exact
    equality."""

    _PAIR = ("favorable_dry_low_infestation", "favorable_dry_high_infestation")

    @pytest.mark.parametrize("vector_name", _PAIR)
    def test_fused_verdict_reproduces_fixture_exactly(self, vector_name: str) -> None:
        fixture = _load_fixture(vector_name)
        result = _run_engine(fixture)
        weather_dominant = dominant_band([s.risk_level.value for s in result.daily_scores])
        fused = fuse(fixture["infestation_score"], weather_dominant)

        expected = fixture["fusion"]["expected_verdict"]
        assert fused.nn_verdict == expected["nn_verdict"]
        assert fused.weather_dominant_level == expected["weather_dominant_level"]
        assert fused.combined_verdict == expected["combined_verdict"]
        assert fused.risk_assessment_level == expected["risk_assessment_level"]
        assert fused.fusion_rule_version == fixture["fusion"]["fusion_rule_version"]

    def test_low_and_high_fused_verdicts_are_distinct(self) -> None:
        """The vision-blind-verdict case, against the
        real fusion function: byte-identical weather (already asserted in
        TestRawEnvironmentalPressureReproduction), yet the low/high pair's
        REPRODUCED fused verdicts must differ -- a fusion function that
        silently ignored the score would fail this."""
        results = {}
        for vector_name in self._PAIR:
            fixture = _load_fixture(vector_name)
            result = _run_engine(fixture)
            weather_dominant = dominant_band([s.risk_level.value for s in result.daily_scores])
            results[vector_name] = fuse(fixture["infestation_score"], weather_dominant)

        low = results["favorable_dry_low_infestation"]
        high = results["favorable_dry_high_infestation"]
        assert low.nn_verdict != high.nn_verdict
        assert low.combined_verdict != high.combined_verdict
        assert low.risk_assessment_level != high.risk_assessment_level
        assert low.weather_dominant_level == high.weather_dominant_level


class TestPrincipledFusedVerdictReproduction:
    """The same
    low/high fixture pair, extended with a sibling `fusion.principled`
    oracle (hand-computed from `fuse_principled`'s decided pressure
    fractions/floors) alongside the existing v1 `fusion.expected_verdict`
    proving the principled rule reproduces exactly against the same fixtures that already pin v1,
    not a free-standing set of numbers."""

    _PAIR = ("favorable_dry_low_infestation", "favorable_dry_high_infestation")

    @pytest.mark.parametrize("vector_name", _PAIR)
    def test_principled_verdict_reproduces_fixture_exactly(self, vector_name: str) -> None:
        fixture = _load_fixture(vector_name)
        result = _run_engine(fixture)
        weather_dominant = dominant_band([s.risk_level.value for s in result.daily_scores])
        fused = fuse_principled(fixture["infestation_score"], weather_dominant)

        expected = fixture["fusion"]["principled"]["expected_verdict"]
        assert fused.nn_verdict == expected["nn_verdict"]
        assert fused.weather_dominant_level == expected["weather_dominant_level"]
        assert fused.combined_verdict == expected["combined_verdict"]
        assert fused.risk_assessment_level == expected["risk_assessment_level"]
        assert fused.fusion_rule_version == fixture["fusion"]["principled"]["fusion_rule_version"]

    def test_principled_rule_differs_from_v1_on_the_same_fixture(self) -> None:
        """The two rules are genuinely different, not just differently
        labeled: on the same byte-identical weather + score, v1's
        max-of-bands and the principled product rule disagree at least
        once across this pair (low_infestation: v1=Moderate vs
        principled=Low)."""
        low = _load_fixture("favorable_dry_low_infestation")
        v1_combined = low["fusion"]["expected_verdict"]["combined_verdict"]
        principled_combined = low["fusion"]["principled"]["expected_verdict"]["combined_verdict"]
        assert v1_combined != principled_combined

    def test_low_and_high_principled_verdicts_are_distinct(self) -> None:
        """The vision-blind-verdict case, reproduced
        against the principled rule too: byte-identical weather, yet the
        low/high pair's principled verdicts must still differ."""
        results = {}
        for vector_name in self._PAIR:
            fixture = _load_fixture(vector_name)
            result = _run_engine(fixture)
            weather_dominant = dominant_band([s.risk_level.value for s in result.daily_scores])
            results[vector_name] = fuse_principled(fixture["infestation_score"], weather_dominant)

        low = results["favorable_dry_low_infestation"]
        high = results["favorable_dry_high_infestation"]
        assert low.combined_verdict != high.combined_verdict
        assert low.risk_assessment_level != high.risk_assessment_level
        assert low.weather_dominant_level == high.weather_dominant_level


class TestFusedVerdictFixtureWellFormed:
    """Structural well-formedness of the fixtures' authored oracle itself
    (kept alongside TestFusedVerdictReproduction above, which checks the
    real implementation against it) -- validated by inspection alone, no
    fusion function involved."""

    _PAIR = ("favorable_dry_low_infestation", "favorable_dry_high_infestation")

    def test_fixtures_record_well_formed_fusion_expectations(self) -> None:
        for vector_name in self._PAIR:
            fixture = _load_fixture(vector_name)
            assert fixture["infestation_score"] is not None

            fusion = fixture["fusion"]
            assert fusion["status"] == "reproduced"
            assert fusion["fusion_rule_version"] == "v1-max-of-bands"
            assert fusion["note"]

            expected = fusion["expected_verdict"]
            for key in (
                "nn_verdict",
                "weather_dominant_level",
                "combined_verdict",
                "risk_assessment_level",
            ):
                assert expected.get(key), f"{vector_name}: expected_verdict.{key} missing/empty"
            assert expected["nn_verdict"] in ("Low", "Moderate", "High")
            assert expected["combined_verdict"] in ("Low", "Moderate", "High")
            assert expected["risk_assessment_level"] in ("Low", "Medium", "High")

            daily_levels = [
                s["risk_level"]
                for s in fixture["expected_environmental_pressure"]["daily_scores"]
            ]
            ordinal = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
            worst = max(daily_levels, key=lambda level: ordinal[level])
            assert worst == expected["weather_dominant_level"]

    def test_low_and_high_infestation_expectations_are_distinct(self) -> None:
        """The distinctness oracle: given byte-identical weather (already asserted in
        TestRawEnvironmentalPressureReproduction), the low/high pair's
        *authored* fused-verdict expectations must differ -- otherwise the
        oracle itself would be unable to catch a vision-blind verdict."""
        low = _load_fixture("favorable_dry_low_infestation")["fusion"]["expected_verdict"]
        high = _load_fixture("favorable_dry_high_infestation")["fusion"]["expected_verdict"]

        assert low["nn_verdict"] != high["nn_verdict"]
        assert low["combined_verdict"] != high["combined_verdict"]
        assert low["risk_assessment_level"] != high["risk_assessment_level"]
        assert low["weather_dominant_level"] == high["weather_dominant_level"]
