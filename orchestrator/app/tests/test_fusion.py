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

"""The shared fusion function.

Covers the one-shared-fusion-function requirement directly: determinism/identical-across-paths,
weather-side band differences surfacing in verdicts, and the vision-blind
detection scenario -- independent of any engine/HTTP wiring (see
test_engine_failure_isolation.py and test_fuzzy_client.py for those).
"""
from __future__ import annotations

import pytest

from fusion import (
    FUSION_RULE_PRINCIPLED,
    FUSION_RULE_VERSION,
    UnknownBandError,
    UnknownFusionRuleError,
    dominant_band,
    fuse,
    fuse_principled,
    nn_verdict,
    select_fusion_rule,
)


class TestNnVerdictThresholds:
    """Reused verbatim from PDM's own nn_verdict (fusion.py module
    docstring) -- score >= 0.70 High, >= 0.40 Moderate, else Low."""

    def test_low_below_040(self) -> None:
        assert nn_verdict(0.0) == "Low"
        assert nn_verdict(0.39) == "Low"

    def test_moderate_between_040_and_070(self) -> None:
        assert nn_verdict(0.40) == "Moderate"
        assert nn_verdict(0.69) == "Moderate"

    def test_high_at_or_above_070(self) -> None:
        assert nn_verdict(0.70) == "High"
        assert nn_verdict(1.0) == "High"


class TestFuseMaxOfBands:
    @pytest.mark.parametrize(
        "score,weather_level,expected_combined,expected_risk_level",
        [
            (0.05, "low", "Low", "Low"),
            (0.05, "moderate", "Moderate", "Medium"),
            (0.95, "moderate", "High", "High"),
            (0.05, "critical", "Critical", "High"),
            (0.95, "low", "High", "High"),
            (0.5, "Medium", "Moderate", "Medium"),
            (0.05, "High", "High", "High"),
        ],
    )
    def test_max_of_bands(self, score, weather_level, expected_combined, expected_risk_level) -> None:
        fused = fuse(score, weather_level)
        assert fused.combined_verdict == expected_combined
        assert fused.risk_assessment_level == expected_risk_level
        assert fused.fusion_rule_version == FUSION_RULE_VERSION

    def test_unknown_band_raises(self) -> None:
        with pytest.raises(UnknownBandError):
            fuse(0.5, "extreme")


class TestFusionDeterministicAndIdenticalAcrossPaths:
    """Identical inputs (same score band, same
    weather-side band) -> identical fused verdicts + the same fusion-rule
    version on both paths."""

    def test_identical_inputs_identical_outputs(self) -> None:
        assert fuse(0.5, "moderate") == fuse(0.5, "moderate")

    def test_pdm_and_fuzzy_vocabularies_agree_at_the_same_ordinal(self) -> None:
        pdm_style = fuse(0.5, "Medium")
        fuzzy_style = fuse(0.5, "moderate")
        assert pdm_style.combined_verdict == fuzzy_style.combined_verdict
        assert pdm_style.risk_assessment_level == fuzzy_style.risk_assessment_level
        assert pdm_style.fusion_rule_version == fuzzy_style.fusion_rule_version


class TestWeatherSideBandDifferencesSurfaceInVerdicts:
    """When the score band does not dominate both paths,
    differing weather-side bands must produce differing fused verdicts --
    the point of the PDM-vs-fuzzy comparison."""

    def test_low_weather_side_ordinal_lets_score_band_dominate_identically(self) -> None:
        assert fuse(0.95, "low").combined_verdict == fuse(0.95, "moderate").combined_verdict == "High"

    def test_high_weather_side_ordinal_diverges_the_paths(self) -> None:
        low_weather = fuse(0.05, "low")
        high_weather = fuse(0.05, "critical")
        assert low_weather.combined_verdict != high_weather.combined_verdict


class TestVisionBlindVerdictIsCaught:
    """Vision-blind verdict is caught: identical weather-side bands
    with a low vs. a high infestation_score must NOT produce the same
    fused verdict."""

    def test_low_vs_high_score_diverges_with_identical_weather(self) -> None:
        low = fuse(0.05, "moderate")
        high = fuse(0.95, "moderate")
        assert low.risk_assessment_level != high.risk_assessment_level
        assert low.weather_dominant_level == high.weather_dominant_level


class TestPrincipledFusionRule:
    """Environmental pressure x confirmed presence, a versioned
    alternative behind the same `(score, weather_level) -> FusedVerdict`
    interface. Not the active default -- these
    tests exercise it directly, independent of which rule `Settings`
    selects."""

    def test_low_pressure_and_low_presence_stays_low(self) -> None:
        fused = fuse_principled(0.1, "low")
        assert fused.combined_verdict == "Low"
        assert fused.risk_assessment_level == "Low"
        assert fused.fusion_rule_version == FUSION_RULE_PRINCIPLED

    def test_high_pressure_alone_does_not_dominate(self) -> None:
        fused = fuse_principled(0.02, "critical")
        assert fused.combined_verdict == "Low"

    def test_high_score_alone_does_not_dominate(self) -> None:
        fused = fuse_principled(0.95, "low")
        assert fused.combined_verdict == "Low"
        assert fused.nn_verdict == "High"

    def test_both_jointly_elevated_reaches_high(self) -> None:
        fused = fuse_principled(0.9, "high")
        assert fused.combined_verdict == "High"
        assert fused.risk_assessment_level == "High"

    def test_boundary_at_moderate_floor(self) -> None:
        fused = fuse_principled(0.5, "moderate")
        assert fused.combined_verdict == "Moderate"

    def test_unknown_band_raises(self) -> None:
        with pytest.raises(UnknownBandError):
            fuse_principled(0.5, "extreme")

    def test_fusion_rule_version_recorded(self) -> None:
        assert fuse_principled(0.5, "high").fusion_rule_version == FUSION_RULE_PRINCIPLED
        assert fuse_principled(0.5, "high").fusion_rule_version != FUSION_RULE_VERSION


class TestSelectFusionRule:
    """Config-selectable dispatch -- the mechanism `service.py` uses to turn
    a `Settings.fusion_rule_version` string into a callable rule."""

    def test_selects_v1_by_default_version_string(self) -> None:
        assert select_fusion_rule(FUSION_RULE_VERSION) is fuse

    def test_selects_principled_by_its_version_string(self) -> None:
        assert select_fusion_rule(FUSION_RULE_PRINCIPLED) is fuse_principled

    def test_unregistered_version_raises(self) -> None:
        with pytest.raises(UnknownFusionRuleError):
            select_fusion_rule("v99-invented")


class TestDominantBand:
    def test_worst_band_wins(self) -> None:
        assert dominant_band(["low", "moderate", "high", "low"]) == "high"

    def test_single_value(self) -> None:
        assert dominant_band(["critical"]) == "critical"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            dominant_band([])
