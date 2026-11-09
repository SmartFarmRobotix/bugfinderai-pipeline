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
In-process fuzzy engine invocation: the Weather service history fetch (via GK) ->
`WeatherDataFrame` adapter -> `FuzzyMamdaniRisk` -> the dominant band across
the window -- the fuzzy path's raw environmental pressure, ready for the
shared fusion step.

Vision-blind by construction: `risk_input.infestation_score` is never read
by anything in this module -- only `crop`/`growth_stage`-adjacent fields
(`crop`) and the weather-window/position fields reach the engine. The
caller (service.py) is the only place the score enters, via `fusion.fuse`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agstack_pnd.foundation.types import ThreatDefinition, WeatherDataFrame
from agstack_pnd.models.agronomic.humidity_streak import HumidityStreak
from agstack_pnd.models.agronomic.leaf_wetness_rh import LeafWetnessRH
from agstack_pnd.models.disease.fuzzy_mamdani import FuzzyMamdaniRisk

from config import Settings
from contract import RiskInput
from fusion import dominant_band
from fuzzy_threats import FuzzyThreatNotFound
from report_content import resolve_threat_record
from weather_adapter import ws_history_to_weather_dataframe
from ws_client import fetch_hourly_history

class FuzzyAssessmentError(RuntimeError):
    """Any failure in the fuzzy path's Weather-service fetch, adapter or engine. This is the
    failure-isolation boundary: callers catch this (or any
    other exception from `assess_via_fuzzy_engine`) and record the fuzzy
    path as failed/absent without disturbing the PDM path or the session
    record."""


@dataclass
class WeatherWindowSummary:
    """The session window's weather,
    summarized from the SAME `WeatherDataFrame` the engine already
    consumed (no new network call) -- min/max temperature, average
    humidity, and the Weather service response's own source label. Computed whenever
    the fuzzy path reaches a result, so a successful `FuzzyAssessmentResult`
    always carries one."""

    temperature_min_C: float
    temperature_max_C: float
    relative_humidity_avg_pct: float
    source: str


def _compute_fuzzy_explanation(
    weather_frame: WeatherDataFrame,
    threat: ThreatDefinition,
    daily_scores: List[Dict[str, Any]],
    dominant: str,
) -> Dict[str, Any]:
    """The per-day
    aggregates, the two public sub-models, the threat's own thresholds, and
    a per-day rule range check, all computed from the SAME frame and threat
    record the engine already consumed -- zero new network calls, never
    re-implementing the engine's own defuzzification. A plain dict (not a
    dataclass) so `report_content.compose_fuzzy_details` never needs to
    import this module (which already imports `report_content`)."""
    temps = weather_frame.fields["air_temperature"]
    rh = weather_frame.fields["relative_humidity"]
    precip = weather_frame.fields["precipitation"]
    ts = weather_frame.timestamps
    unique_dates = np.unique(ts.astype("datetime64[D]"))

    lw_result = LeafWetnessRH().calculate(weather_frame)
    streak_result = HumidityStreak().calculate(weather_frame, {"rh_threshold": 80.0})

    days: List[Dict[str, Any]] = []
    for day in unique_dates:
        mask = ts.astype("datetime64[D]") == day
        temp_avg = float(np.mean(temps[mask])) if np.any(mask) else 0.0
        rh_max = float(np.max(rh[mask])) if np.any(mask) else 0.0
        rain_sum = float(np.sum(precip[mask]))
        matched_rule_levels = [
            rule.risk_level.value
            for rule in threat.fuzzy_rules
            if rule.temp_lo <= temp_avg <= rule.temp_hi
            and rule.hum_lo <= rh_max <= rule.hum_hi
            and rain_sum >= rule.rain_min
        ]
        days.append(
            {
                "date": str(day.astype("datetime64[D]").astype(object)),
                "temperature_avg_C": temp_avg,
                "humidity_max_pct": rh_max,
                "rain_mm": rain_sum,
                "matched_rule_levels": matched_rule_levels,
            }
        )

    score_values = [float(score["value"]) for score in daily_scores]

    return {
        "days": days,
        "humidity_streak_days": float(streak_result.summary.get("max_streak_days", 0.0)),
        "min_streak_days": threat.bio_params.min_streak,
        "wettest_day_wet_hours": max((s.value for s in lw_result.daily_scores), default=0.0),
        "min_wetness_hours_high": threat.bio_params.min_wetness_hours_high,
        "rule_count": len(threat.fuzzy_rules),
        "score_low": min(score_values) if score_values else 0.0,
        "score_high": max(score_values) if score_values else 0.0,
        "dominant_band": dominant,
    }


@dataclass
class FuzzyAssessmentResult:
    dominant_band: str
    daily_scores: List[Dict[str, Any]]
    engine_version: str
    threat_definition_version: str
    dossier_version: str
    weather_summary: WeatherWindowSummary
    explanation: Dict[str, Any] = field(default_factory=dict)


def assess_via_fuzzy_engine(
    settings: Settings, risk_input: RiskInput, pest: Optional[str] = None
) -> FuzzyAssessmentResult:
    """Weather-only by API: the infestation score is never passed in. The ThreatDefinition is resolved by the same function the
    report content uses (`report_content.resolve_threat_record`: the
    (crop, `pest`) pair when the session carries a pest, the crop alone
    otherwise), so both paths read one catalogue record. Raises
    `FuzzyAssessmentError` on any failure -- never fabricates a result."""
    if risk_input.latitude is None or risk_input.longitude is None:
        raise FuzzyAssessmentError(
            "session coordinates are absent; the fuzzy path requires "
            "lat/lon for the Weather service history call (mirrors PDM's own "
            "position_unavailable degraded reason)"
        )

    try:
        threat_record = resolve_threat_record(risk_input.crop, pest)

        ws_response = fetch_hourly_history(
            settings,
            latitude=risk_input.latitude,
            longitude=risk_input.longitude,
            time_start=risk_input.time_start,
            time_end=risk_input.time_end,
        )
        weather_frame = ws_history_to_weather_dataframe(ws_response)
        weather_summary = WeatherWindowSummary(
            temperature_min_C=round(float(weather_frame.fields["air_temperature"].min()), 1),
            temperature_max_C=round(float(weather_frame.fields["air_temperature"].max()), 1),
            relative_humidity_avg_pct=round(float(weather_frame.fields["relative_humidity"].mean()), 1),
            source=str(ws_response.get("source") or "unknown"),
        )
        result = FuzzyMamdaniRisk().calculate(weather_frame, threat=threat_record.definition)
    except FuzzyThreatNotFound as exc:
        raise FuzzyAssessmentError(f"no ThreatDefinition for this session: {exc}") from exc
    except FuzzyAssessmentError:
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberate: the Weather service/GK/adapter/engine
        raise FuzzyAssessmentError(f"fuzzy engine pipeline failed: {exc}") from exc

    if not result.daily_scores:
        raise FuzzyAssessmentError("fuzzy engine returned no daily scores for this window")

    dominant = dominant_band([score.risk_level.value for score in result.daily_scores])
    daily_scores = [
        {"date": str(score.date), "value": score.value, "risk_level": score.risk_level.value}
        for score in result.daily_scores
    ]

    return FuzzyAssessmentResult(
        dominant_band=dominant,
        daily_scores=daily_scores,
        engine_version=FuzzyMamdaniRisk.metadata.version,
        threat_definition_version=threat_record.version,
        dossier_version=threat_record.dossier_version,
        weather_summary=weather_summary,
        explanation=_compute_fuzzy_explanation(weather_frame, threat_record.definition, daily_scores, dominant),
    )


def assess_outlook_hours_via_fuzzy_engine(
    crop: str, observations: List[Dict[str, Any]], pest: Optional[str] = None
) -> Dict[str, str]:
    """
    The fuzzy level of each outlook day, from the hourly weather of those
    days. `observations` are Weather-service hourly rows (`{"timestamp",
    "values": {temperature_2m, relative_humidity_2m, precipitation}}`), all
    complete; they go through the same adapter as the session window into the
    engine, so its streak and wetness sub-models see real hours, and the
    engine returns one score per calendar day. Weather-only and vision-blind
    like `assess_via_fuzzy_engine`; pure -- no network. Returns {date: band};
    raises `FuzzyAssessmentError` on any failure.
    """
    if not observations:
        raise FuzzyAssessmentError("no hourly observation to evaluate")

    try:
        threat_record = resolve_threat_record(crop, pest)
        frame = ws_history_to_weather_dataframe({"data": observations})
        result = FuzzyMamdaniRisk().calculate(frame, threat=threat_record.definition)
    except FuzzyThreatNotFound as exc:
        raise FuzzyAssessmentError(f"no ThreatDefinition for this session: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- same isolation as the session-window path
        raise FuzzyAssessmentError(f"fuzzy engine pipeline failed: {exc}") from exc

    return {str(score.date): str(score.risk_level.value).capitalize() for score in result.daily_scores}
