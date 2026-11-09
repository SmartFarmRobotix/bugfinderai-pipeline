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
Versioned content map: authoritative fused-band -> conclusion/recommendation
text, with EPPO pest identity substituted from the orchestrator's own
in-code catalog. Deterministic markers
prefix every generated sentence so presence/absence is assertable by
substring search in the composed payload or the rendered PDF text, never judged.

The markers below are the fixed sentence prefixes of the report; tests assert them verbatim.

No growth stage in report content: nothing in this module reads
`RiskInput.growth_stage` or any BBCH/stage vocabulary -- crop/pest identity
here is EPPO code + common/scientific name only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from fuzzy_threats import (
    FuzzyThreatNotFound,
    ThreatDefinitionRecord,
    get_threat_record_by_eppo,
    get_threat_record_for_crop,
)

CONTENT_MAP_VERSION = "v1"

CONCLUSION_MARKER = "Conclusion: "
RECOMMENDATION_MARKER = "Recommended action: "
STOCK_MODEL_NOTICE_MARKER = (
    "Note: detections in this report were produced by a generic "
    "integration model not designated for agricultural claims"
)
FUSED_VERDICT_AUTHORITATIVE_MARKER = "Weather-rule assessment: "
FUSED_VERDICT_COMPARATIVE_MARKER = "Fuzzy assessment: "
FUSED_VERDICT_EDGE_MARKER = "Edge assessment (score band): "

DEFAULT_SITE_LABEL = "Field site"

_BAND_ACTION = {
    "Low": "Continue routine monitoring; no immediate intervention indicated",
    "Medium": "Increase monitoring frequency and prepare intervention options",
    "High": "Immediate intervention recommended",
}

_CROP_DISPLAY_NAME = {
    "brassica-oleracea": "Cabbage / Brassica crops (Brassica oleracea)",
    "brassica-oleracea-sabellica": "Kale (Brassica oleracea var. sabellica)",
    "cucumis-sativus": "Cucumber (Cucumis sativus)",
}


@dataclass(frozen=True)
class PestIdentity:
    pest_name: str
    scientific_name: str
    eppo_code: str
    eppo_url: str


def crop_display_name(crop: str) -> str:
    return _CROP_DISPLAY_NAME.get(crop, crop)


def resolve_threat_record(crop: str, pest: Optional[str] = None) -> ThreatDefinitionRecord:
    """The ONE resolution of a session's catalog record, used by the report
    content and by the fuzzy path so both read the same record: by the
    (crop, EPPO code) pair when the session carries a pest, by crop alone
    otherwise. Raises `FuzzyThreatNotFound` when there is no such record
    never another pest's record for the crop."""
    if pest:
        return get_threat_record_by_eppo(crop, pest)
    return get_threat_record_for_crop(crop)


def resolve_pest_identity(crop: str, pest: Optional[str] = None) -> Optional[PestIdentity]:
    """The orchestrator's own catalog entry for `crop` (and `pest`, when the
    session carries one), or `None` when the catalog has no match -- callers
    must degrade gracefully (generic subject wording), never fabricate an
    identity."""
    try:
        record = resolve_threat_record(crop, pest)
    except FuzzyThreatNotFound:
        return None
    definition = record.definition
    return PestIdentity(
        pest_name=definition.common_name,
        scientific_name=definition.scientific_name,
        eppo_code=record.eppo_code,
        eppo_url=f"https://gd.eppo.int/taxon/{record.eppo_code}",
    )


def compose_pest_crop_context_unavailable(crop: str, pest: str) -> str:
    """The pest-and-crop section's one sentence for a (crop, pest) pair the
    catalog holds no record for: it names both, and prints no other pest."""
    return (
        f"No catalogue record exists for {pest} on {crop_display_name(crop)}; "
        "the pest and crop context is unavailable for this session."
    )


def _subject(identity: Optional[PestIdentity], crop: str) -> str:
    if identity is None:
        return f"the detected threat on {crop_display_name(crop)}"
    return f"{identity.pest_name} ({identity.eppo_code}) on {crop_display_name(crop)}"


def compose_conclusion(band: str, identity: Optional[PestIdentity], crop: str) -> str:
    """The conclusions-section text for a session with a completed
    authoritative assessment."""
    return f"{CONCLUSION_MARKER}session risk assessment for {_subject(identity, crop)} is {band.upper()}."


def compose_conclusion_unavailable(crop: str) -> str:
    """The authoritative (PDM) path did not complete for this
    session -- an honest degraded conclusion, never a fabricated band."""
    return (
        f"{CONCLUSION_MARKER}authoritative (PDM-path) "
        f"assessment unavailable for this session on {crop_display_name(crop)}; "
        "see the labeled fused verdicts for each path's individual outcome."
    )


def compose_recommendation(band: str, identity: Optional[PestIdentity], crop: str) -> str:
    """The actionable-recommendation text."""
    action = _BAND_ACTION.get(band, "Review session details; risk band not recognized")
    return f"{RECOMMENDATION_MARKER}{action} for {_subject(identity, crop)}."


_STOCK_MODEL_NOTICE_TEMPLATE = (
    "{marker} ({detector_impl_id} version {detector_version}, "
    "designation_status={designation_status}). Agricultural conclusions and "
    "recommendations are suppressed for this report."
)


def compose_stock_model_notice(detector_impl_id: str, detector_version: str, designation_status: str) -> str:
    """The integration-only notice -- deterministically detectable
    (its own marker) so "suppressed by mode" is never confused with "missing
    by defect"."""
    return _STOCK_MODEL_NOTICE_TEMPLATE.format(
        marker=STOCK_MODEL_NOTICE_MARKER,
        detector_impl_id=detector_impl_id,
        detector_version=detector_version,
        designation_status=designation_status,
    )


def compose_fused_verdict_sentence(marker: str, level: Optional[str], status: str) -> str:
    """One path's fused verdict AS a
    sentence -- the marker IS the sentence's fixed prefix. Engine
    identity/version/the fusion-rule identifier never ride here (moved to
    `compose_engine_provenance_line`): this sentence carries no engine name,
    version string, or the token "fusion"."""
    if status != "present" or level is None:
        return f"{marker}n/a (status={status})"
    return f"{marker}{level}"


def compose_engine_provenance_line(
    role: str,
    engine_identity: str,
    engine_version: str,
    fusion_rule_version: Optional[str],
) -> str:
    """One engine-provenance line per
    present path, in the fixed shape `<role> path: <engine> <version>,
    fusion <rule>` -- `role` is exactly "authoritative" or "comparative".
    `fusion_rule_version=None` omits the fusion clause entirely (edge's own
    single line: nothing is fused on edge)."""
    line = f"{role} path: {engine_identity} {engine_version}"
    if fusion_rule_version is not None:
        line = f"{line}, fusion {fusion_rule_version}"
    return line


_BAND_WORD = {
    "low": "Low",
    "moderate": "Medium",
    "medium": "Medium",
    "high": "High",
}
_BAND_RANK = {"Low": 0, "Medium": 1, "High": 2}

_DEGRADED_REASON_WORDS = {
    "position_unavailable": "position unavailable",
    "weather_unavailable": "weather unavailable",
    "weather_not_applicable": "weather not applicable",
    "no_pest_models_configured": "no pest models configured",
    "no_pest_model_for_crop": "no weather rules for this crop",
}


def _band_word(band: Any) -> Optional[str]:
    return _BAND_WORD.get(str(band).lower()) if band is not None else None


_PLAIN_VARIABLES = {
    "temperature_2m": ("temperature", " °C"),
    "relative_humidity_2m": ("relative humidity", " %"),
    "precipitation": ("precipitation", " mm"),
    "soil_temperature_0_to_7cm": ("soil temperature (0 to 7 cm)", " °C"),
    "soil_temperature_7_to_28cm": ("soil temperature (7 to 28 cm)", " °C"),
    "soil_temperature_28_to_100cm": ("soil temperature (28 to 100 cm)", " °C"),
    "soil_temperature_100_to_255cm": ("soil temperature (100 to 255 cm)", " °C"),
}
_PLAIN_OPERATORS = {
    ">=": "at least",
    "<=": "at most",
    ">": "above",
    "<": "below",
    "==": "equal to",
    "=": "equal to",
    "!=": "different from",
}
_CONDITION_PART = re.compile(r"^\s*(\w+)\s*(>=|<=|==|!=|>|<|=)\s*(\S+)\s*$")


def _plain_number(text: str) -> str:
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:g}"


def _plain_condition(part: str) -> str:
    match = _CONDITION_PART.match(part)
    if match is None:
        return part.strip()
    variable, symbol, value = match.groups()
    words, unit = _PLAIN_VARIABLES.get(variable, (variable.replace("_", " "), ""))
    return f"{words} {_PLAIN_OPERATORS[symbol]} {_plain_number(value)}{unit}"


def plain_rule_conditions(conditions: str) -> str:
    """The PDM's condition text (`temperature_2m >= 20  AND  relative_humidity_2m >= 70`) in plain words (`temperature at least 20 °C
    and relative humidity at least 70 %`). A part that is not of the
    `variable operator value` shape is kept as it came, never dropped."""
    return " and ".join(_plain_condition(part) for part in str(conditions).split("  AND  "))


def _format_range(lo: float, hi: float) -> str:
    """A range collapses to one
    value when its bounds are equal (the one-day case)."""
    if lo == hi:
        return f"{lo:.1f}"
    return f"{lo:.1f}–{hi:.1f}"


def compose_fuzzy_details(explanation: Mapping[str, Any]) -> str:
    """The ONE
    fuzzy paragraph for the bottom of the report, composed from what the
    engine and its public sub-models return for the session's window
    never re-implementing the engine's own defuzzification. `explanation`
    is `fuzzy_client._compute_fuzzy_explanation`'s dict."""
    days = explanation["days"]
    n = len(days)
    temps = [d["temperature_avg_C"] for d in days]
    humidities = [d["humidity_max_pct"] for d in days]
    rain_total = sum(d["rain_mm"] for d in days)
    matched_days = [d for d in days if d["matched_rule_levels"]]
    k = len(matched_days)
    levels_met = sorted({level.capitalize() for d in days for level in d["matched_rule_levels"]})
    levels_text = ", ".join(levels_met) if levels_met else "none"
    streak = int(round(explanation["humidity_streak_days"]))
    min_streak = explanation["min_streak_days"]
    wet_hours = explanation["wettest_day_wet_hours"]
    min_wet_high = explanation["min_wetness_hours_high"]
    rule_count = explanation["rule_count"]
    k_text = "none" if k == 0 else str(k)
    dominant_band = str(explanation["dominant_band"]).capitalize()
    if n == 1:
        scores_text = f"The model's daily score is {explanation['score_low']:.1f} of 100"
    else:
        scores_text = (
            "The model's daily scores ranged "
            f"{_format_range(explanation['score_low'], explanation['score_high'])} of 100"
        )

    return (
        f"Fuzzy model (Mamdani) over the same window: daily average temperature "
        f"{_format_range(min(temps), max(temps))} °C, daily maximum relative humidity "
        f"{_format_range(min(humidities), max(humidities))} %, {rain_total:.1f} mm of rain in "
        f"total. Humid days in a row (relative humidity of at least 80 %): {streak} of the "
        f"{min_streak} the model requires. Leaf wetness: {wet_hours:.1f} hours on the wettest "
        f"day, {min_wet_high:.1f} required for High. {k_text} of the {n} days had values within "
        f"the ranges of at least one of the model's {rule_count} rules ({levels_text}). "
        f"{scores_text}; the resulting band is {dominant_band}."
    )


def compose_fuzzy_not_evaluated(reason: str) -> str:
    """The fuzzy sub-block's
    sentence for a session whose fuzzy path did not reach a result."""
    return f"The fuzzy model was not evaluated: {reason}."


def _weather_rule_sentences(provenance: Mapping[str, Any], weather_band: Optional[str]) -> List[str]:
    sentences: List[str] = []
    for model in provenance.get("rules_evaluated") or []:
        held = [
            f"'{plain_rule_conditions(rule['conditions'])}' held for {rule['hours_triggered']} of "
            f"{rule['hours_total']} hours ({_band_word(rule.get('level')) or rule.get('level')} risk)"
            for rule in model.get("rules") or []
        ]
        band = weather_band or _band_word(model.get("dominant_level"))
        if not held or band is None:
            continue
        sentences.append(
            f"Weather rules for {model.get('model')} over the session window: "
            f"{'; '.join(held)}. These rules put the weather risk at {band}."
        )
    return sentences


def _observed_weather_sentence(provenance: Mapping[str, Any]) -> Optional[str]:
    observed = provenance.get("observed_weather") or {}
    parts: List[str] = []
    t_avg = observed.get("temperature_2m_avg_C")
    t_max = observed.get("temperature_2m_max_C")
    rh_avg = observed.get("relative_humidity_2m_avg_pct")
    precip = observed.get("precipitation_mm")
    if t_avg is not None:
        temperature = f"an average temperature of {t_avg:.1f} °C"
        if t_max is not None:
            temperature += f" (maximum {t_max:.1f} °C)"
        parts.append(temperature)
    if rh_avg is not None:
        parts.append(f"an average relative humidity of {rh_avg:.1f} %")
    if not parts:
        return None
    sentence = f"The window had {' and '.join(parts)}"
    if precip is not None:
        sentence += f", with {precip:.2f} mm of precipitation"
    return sentence + "."


SCORE_ONLY_DETAILS = "Based on the detections in this session; no weather data for this assessment."


def compose_risk_details_short(provenance: Mapping[str, Any], level: Optional[str] = None) -> str:
    """The ONE short sentence for the Risk Assessment block near the top of
    the report: "Weather conditions alone: {weather band} risk. Combined
    with the detections: {level}." for a weather-rule assessment, the
    score-only sentence for a degraded one. `level` is the authoritative
    (fused) level the verdict rows carry; when not given it is the higher of
    the two bands. Returns "" when the facts for it are absent from
    `provenance`."""
    if provenance.get("degraded_reason"):
        return SCORE_ONLY_DETAILS
    weather_band = _band_word(provenance.get("weather_rule_level_raw"))
    detection_band = _band_word(provenance.get("nn_band"))
    if weather_band is None or detection_band is None:
        return ""
    if level is None:
        level = max(weather_band, detection_band, key=_BAND_RANK.__getitem__)
    return f"Weather conditions alone: {weather_band} risk. Combined with the detections: {level}."


def compose_risk_details_full(
    provenance: Mapping[str, Any],
    infestation_score: Optional[float],
    level: Optional[str] = None,
) -> str:
    """The full explanation of the risk assessment, for the bottom of the
    report, composed from the assessment's own facts (the PDM provenance)
    instead of the assessing service's generic sentence: the rules of each
    pest model in plain words with the hours they held, the observed
    weather, the detection band with the score described as the detector's
    confidence on the most affected images, and the combination. One
    paragraph; a clause whose facts are absent from `provenance` is
    omitted, never filled in. A score-only or degraded assessment
    (`degraded_reason` present) says so and names the reason, and claims no
    weather fact.

    The closing combination sentence needs both bands; `level` is the
    authoritative level the report shows, and when not given it is the
    higher of the two bands. Returns "" when there is nothing to say."""
    score_text = f"{infestation_score:.2f}" if infestation_score is not None else None

    reason = provenance.get("degraded_reason")
    if reason:
        words = _DEGRADED_REASON_WORDS.get(reason, str(reason).replace("_", " "))
        text = f"Score-only assessment: the weather was not evaluated ({words})."
        if score_text is not None:
            text += f" The level comes from the infestation score ({score_text}) alone."
        return text

    weather_band = _band_word(provenance.get("weather_rule_level_raw"))
    detection_band = _band_word(provenance.get("nn_band"))

    sentences = _weather_rule_sentences(provenance, weather_band)
    observed = _observed_weather_sentence(provenance)
    if observed:
        sentences.append(observed)
    if detection_band is not None and score_text is not None:
        sentences.append(
            f"The detections put the session in the {detection_band} band; the detector's "
            f"confidence on the most affected images is {score_text}."
        )
    if weather_band is not None and detection_band is not None:
        if level is None:
            level = max(weather_band, detection_band, key=_BAND_RANK.__getitem__)
        sentences.append(f"The risk level is the higher of the two: {level}.")
    return " ".join(sentences)
