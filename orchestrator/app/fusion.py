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
The single shared fusion function: one versioned rule producing EACH risk path's session verdict from
the session's `infestation_score` and that path's weather-side band:
`fuse(score, PDM weather-rule level)` and `fuse(score, Mamdani environmental
band)` call this exact same function, so the only variable between the two
paths is the weather-risk producer (the point of comparing the two paths). No
path's stored verdict may bypass this step.

v1 = max of bands, and it is deliberately not a new invention: it reuses
the OpenAgri PDM service's own `nn_verdict` thresholds and `VERDICT_TO_RISK_LEVEL`
mapping verbatim, so
v1's max-of-bands OUTPUT matches PDM's own in-PDM combination. The INPUT
does not, though, and that distinction is load-bearing:
PDM's own `/api/v1/risk-assessment/` response returns a `level` that is
ALREADY that same combination -- but this orchestrator's central
fusion must be called with PDM's separately-reported RAW, pre-score
weather-rule band (`pdm_client.PdmAssessmentResult.raw_level`,
PDM's `provenance["weather_rule_level_raw"]`), never `level` itself. Fusing
`level` again would fuse the session score against PDM's own output a
second time -- harmless under today's max (max is idempotent: fusing an
already-≥-NN value changes nothing), but it corrupts the STORED raw
component (the raw components stay stored for later analysis, and comparing the paths needs a
genuine weather-only reading, not one already blended with vision) and stops being harmless the moment `fuse`
grows a non-max rule that
is not idempotent under re-application. The point of extracting fusion
here, once, is that both paths -- PDM's raw weather-rule band and the
fuzzy path's raw Mamdani band, neither of which has seen the score -- go
through the exact same `fuse(score, raw_band)` call.

Two related but distinct band vocabularies feed this function:
  the four-band weather-side scale ("low"/"moderate"/"high"/"critical")
    that both PDM's pest-model rule engine and the ported Mamdani engine
    use (PDM's own rules only ever reach "high", but the Mamdani engine's band
    thresholds do reach "critical"), and
  PDM's report-facing three-band vocabulary ("Low"/"Medium"/"High",
    `VERDICT_TO_RISK_LEVEL`'s output), which is what PDM's
    `/api/v1/risk-assessment/` endpoint actually returns as
    `risk_assessment.level` (already fused internally -- see above).
`_BAND_ORDINAL` accepts both, case-insensitively, aliasing "medium" to the
same ordinal as "moderate".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

FUSION_RULE_VERSION = "v1-max-of-bands"

_BAND_ORDINAL = {"low": 1, "moderate": 2, "medium": 2, "high": 3, "critical": 4}

_ORDINAL_BAND = {1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}

_NN_HIGH_FLOOR = 0.70
_NN_MODERATE_FLOOR = 0.40

_VERDICT_TO_RISK_LEVEL = {
    "Low": "Low",
    "Moderate": "Medium",
    "High": "High",
    "Critical": "High",
}


class UnknownBandError(ValueError):
    """A weather-side band string is not one of low/moderate/medium/high/critical."""

    def __init__(self, band: str) -> None:
        super().__init__(
            f"{band!r} is not one of low/moderate/medium/high/critical"
        )


def nn_verdict(infestation_score: float) -> str:
    """Bucket the session's infestation_score into the three-band NN verdict
    (thresholds reused verbatim from PDM's own `nn_verdict`, module docstring)."""
    if infestation_score >= _NN_HIGH_FLOOR:
        return "High"
    if infestation_score >= _NN_MODERATE_FLOOR:
        return "Moderate"
    return "Low"


def _band_ordinal(band: str) -> int:
    try:
        return _BAND_ORDINAL[band.strip().lower()]
    except (KeyError, AttributeError):
        raise UnknownBandError(band) from None


@dataclass(frozen=True)
class FusedVerdict:
    nn_verdict: str
    weather_dominant_level: str
    combined_verdict: str
    risk_assessment_level: str
    fusion_rule_version: str = FUSION_RULE_VERSION


def fuse(infestation_score: float, weather_level: str) -> FusedVerdict:
    """The single shared fusion rule: the fused band is the worse
    (higher-ordinal) of the NN-score band and the weather-side band. Called
    identically for both risk paths -- no bypass. Raises `UnknownBandError`
    if `weather_level` is not a recognized band string; callers must treat
    that as this path's assessment failing, not as a valid absent fusion."""
    nn = nn_verdict(infestation_score)
    combined_ordinal = max(_BAND_ORDINAL[nn.lower()], _band_ordinal(weather_level))
    combined = _ORDINAL_BAND[combined_ordinal]
    return FusedVerdict(
        nn_verdict=nn,
        weather_dominant_level=weather_level,
        combined_verdict=combined,
        risk_assessment_level=_VERDICT_TO_RISK_LEVEL[combined],
    )


FUSION_RULE_PRINCIPLED = "v2-pressure-times-presence"

_PRESSURE_FRACTION = {1: 0.10, 2: _NN_MODERATE_FLOOR, 3: _NN_HIGH_FLOOR, 4: 1.0}

_PRINCIPLED_HIGH_FLOOR = 0.50
_PRINCIPLED_MODERATE_FLOOR = 0.20


def fuse_principled(infestation_score: float, weather_level: str) -> FusedVerdict:
    """The principled fusion rule: environmental
    pressure * confirmed presence, a versioned alternative to `fuse`'s
    max-of-bands behind the identical `(score, weather_level) ->
    FusedVerdict` interface. Where max-of-bands treats either signal alone
    as sufficient (worst-case), this rule requires BOTH the weather-side
    pressure and the vision-confirmed presence to be jointly elevated:
    a "critical" weather band with a near-zero `infestation_score` does
    NOT, by itself, produce an elevated verdict, and a high score under
    quiet weather does not either. Not asserted to be more "correct" than
    v1 -- only a genuinely different, versioned,
    comparison-ready rule to weigh against v1's
    behavior on real sessions. Raises `UnknownBandError` on an
    unrecognized `weather_level`, exactly like `fuse`."""
    pressure_fraction = _PRESSURE_FRACTION[_band_ordinal(weather_level)]
    joint = pressure_fraction * infestation_score
    if joint >= _PRINCIPLED_HIGH_FLOOR:
        combined = "High"
    elif joint >= _PRINCIPLED_MODERATE_FLOOR:
        combined = "Moderate"
    else:
        combined = "Low"
    return FusedVerdict(
        nn_verdict=nn_verdict(infestation_score),
        weather_dominant_level=weather_level,
        combined_verdict=combined,
        risk_assessment_level=_VERDICT_TO_RISK_LEVEL[combined],
        fusion_rule_version=FUSION_RULE_PRINCIPLED,
    )


_FUSION_RULES: Dict[str, Callable[[float, str], FusedVerdict]] = {
    FUSION_RULE_VERSION: fuse,
    FUSION_RULE_PRINCIPLED: fuse_principled,
}


class UnknownFusionRuleError(ValueError):
    """`fusion_rule_version` names a rule not in `_FUSION_RULES`."""

    def __init__(self, version: str) -> None:
        super().__init__(
            f"{version!r} is not a registered fusion_rule_version "
            f"(known: {sorted(_FUSION_RULES)})"
        )


def select_fusion_rule(version: str) -> Callable[[float, str], FusedVerdict]:
    """Dispatch a `fusion_rule_version` string to its rule function -- the
    single place `service.py` resolves which rule to call, so a session's provenance-
    recorded `fusion_rule_version` (the `FusedVerdict` field) always
    matches the rule that actually ran."""
    try:
        return _FUSION_RULES[version]
    except KeyError:
        raise UnknownFusionRuleError(version) from None


def dominant_band(levels: List[str]) -> str:
    """Reduce a list of per-day bands (the fuzzy engine's `daily_scores`,
    weather-only) to the single worst (highest-ordinal) band -- the raw
    weather-side input to `fuse` for the fuzzy path, mirroring PDM's own
    dominant-level-across-window convention
    (the PDM service's `weather_dominant`,
    referenced by the evaluation-vector fixtures as
    `weather_dominant_source`)."""
    if not levels:
        raise ValueError("cannot compute a dominant band over an empty list")
    return max(levels, key=_band_ordinal)
