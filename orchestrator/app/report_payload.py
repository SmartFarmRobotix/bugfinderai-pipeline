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
Full Reporting Service payload composition. The orchestrator is the single
writer of the Reporting envelope in every mode; the Reporting service
stays a pure renderer -- every field composed below fits its existing
`PestSessionReportRequest` shape, so the renderer needs no change.

No growth stage in report content: this module never reads
`RiskInput.growth_stage`, and the `ReportingEnvelope` it forwards has no
growth-stage field to begin with -- both facts hold structurally, not by
convention alone.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from contract import ReportingEnvelope, RiskInput
from forecast_outlook import ForecastResult, forecast_outlook_payload
from fusion import nn_verdict as fusion_nn_verdict
from models import RiskAssessmentRecord
from report_content import (
    DEFAULT_SITE_LABEL,
    FUSED_VERDICT_AUTHORITATIVE_MARKER,
    FUSED_VERDICT_COMPARATIVE_MARKER,
    FUSED_VERDICT_EDGE_MARKER,
    compose_conclusion,
    compose_conclusion_unavailable,
    compose_engine_provenance_line,
    compose_fused_verdict_sentence,
    compose_fuzzy_details,
    compose_fuzzy_not_evaluated,
    compose_pest_crop_context_unavailable,
    compose_recommendation,
    compose_risk_details_full,
    compose_risk_details_short,
    compose_stock_model_notice,
    crop_display_name,
    resolve_pest_identity,
)

DESIGNATED_STATUS = "designated"

POSITION_UNAVAILABLE = "position_unavailable"
WEATHER_UNAVAILABLE = "weather_unavailable"
WEATHER_NOT_APPLICABLE = "weather_not_applicable"
_WEATHER_UNAVAILABLE_REASON_HUMAN = {
    POSITION_UNAVAILABLE: "position unavailable",
    WEATHER_UNAVAILABLE: "weather unavailable",
    WEATHER_NOT_APPLICABLE: "weather not applicable",
}


class ReportComposeError(RuntimeError):
    """The payload cannot be honestly composed -- e.g. a successful fuzzy
    assessment carrying no weather summary."""


def _fused_sentence(marker: str, record: Optional[RiskAssessmentRecord]) -> str:
    if record is None:
        return compose_fused_verdict_sentence(marker, None, "absent")
    return compose_fused_verdict_sentence(marker, record.fused_verdict, record.status)


def _composed_risk_details(
    assessment: Optional[RiskAssessmentRecord],
    risk_input: RiskInput,
    level: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """(the short sentence for the Risk Assessment block, the full paragraph
    for the bottom of the report); each None when its facts are absent."""
    if assessment is None or not assessment.provenance_json:
        return None, None
    provenance = json.loads(assessment.provenance_json)
    short = compose_risk_details_short(provenance, level=level)
    full = compose_risk_details_full(provenance, risk_input.infestation_score, level=level)
    return short or None, full or None


def _composed_fuzzy_details(fuzzy_assessment: Optional[RiskAssessmentRecord]) -> Optional[str]:
    """The "Fuzzy model (Mamdani)"
    sub-block under "Assessment details" -- the paragraph for a completed
    fuzzy path, why-not-evaluated for a failed one, absent (no label
    printed) when no fuzzy path was attempted at all."""
    if fuzzy_assessment is None:
        return None
    if fuzzy_assessment.status == "present":
        if not fuzzy_assessment.provenance_json:
            return None
        provenance = json.loads(fuzzy_assessment.provenance_json)
        explanation = provenance.get("explanation")
        if not explanation:
            return None
        return compose_fuzzy_details(explanation)
    if fuzzy_assessment.status == "failed":
        return compose_fuzzy_not_evaluated(fuzzy_assessment.failure_reason or "unknown reason")
    return None


def reassessment_provenance_line(edge_report_id: str) -> str:
    """The provenance line of a cloud re-assessment of a synced edge session,
    naming the on-device report."""
    return (
        f"Cloud re-assessment after synchronisation of the on-device report {edge_report_id}: "
        "the weather rules, the fuzzy path and the outlook are computed from the recorded weather "
        "of the session window; the on-device report keeps its score-band assessment."
    )


def _engine_provenance_lines(
    pdm_assessment: Optional[RiskAssessmentRecord],
    fuzzy_assessment: Optional[RiskAssessmentRecord],
    is_edge: bool,
) -> Optional[List[str]]:
    """One line per PRESENT path, fixed shapes; edge's single line
    carries no fusion clause (nothing is fused on edge)."""
    lines: List[str] = []
    if pdm_assessment is not None and pdm_assessment.status == "present":
        lines.append(
            compose_engine_provenance_line(
                "authoritative",
                pdm_assessment.engine_identity or "n/a",
                pdm_assessment.engine_version or "n/a",
                None if is_edge else pdm_assessment.fusion_rule_version,
            )
        )
    if not is_edge and fuzzy_assessment is not None and fuzzy_assessment.status == "present":
        lines.append(
            compose_engine_provenance_line(
                "comparative",
                fuzzy_assessment.engine_identity or "n/a",
                fuzzy_assessment.engine_version or "n/a",
                fuzzy_assessment.fusion_rule_version,
            )
        )
    return lines or None


def _weather_summary_payload(
    risk_input: RiskInput,
    fuzzy_assessment: Optional[RiskAssessmentRecord],
    is_edge: bool,
) -> Dict[str, Any]:
    """The
    renderer's existing `weather_summary` shape when the comparative path
    actually fetched weather, else an honest reason from the existing
    degraded taxonomy. Never both (the Reporting service's WeatherSummary validator refuses
    that contradiction; this composer never even attempts it)."""
    if is_edge:
        return {"unavailable_reason": _WEATHER_UNAVAILABLE_REASON_HUMAN[WEATHER_NOT_APPLICABLE]}

    if risk_input.latitude is None or risk_input.longitude is None:
        return {"unavailable_reason": _WEATHER_UNAVAILABLE_REASON_HUMAN[POSITION_UNAVAILABLE]}

    if fuzzy_assessment is None or fuzzy_assessment.status != "present":
        return {"unavailable_reason": _WEATHER_UNAVAILABLE_REASON_HUMAN[WEATHER_UNAVAILABLE]}

    provenance: Dict[str, Any] = {}
    if fuzzy_assessment.provenance_json:
        provenance = json.loads(fuzzy_assessment.provenance_json)
    summary = provenance.get("weather_summary")
    if not summary:
        raise ReportComposeError(
            "fuzzy assessment status='present' but its provenance carries no "
            "weather_summary -- a successful weather fetch must never compose "
            "without one"
        )

    return {
        "source": summary.get("source"),
        "variables": ["temperature_2m", "relative_humidity_2m"],
        "observed": {
            "temperature_2m_max_C": summary.get("temperature_max_C"),
            "temperature_2m_min_C": summary.get("temperature_min_C"),
            "relative_humidity_2m_avg_pct": summary.get("relative_humidity_avg_pct"),
        },
    }


def compose_report_payload(
    risk_input: RiskInput,
    envelope: ReportingEnvelope,
    pdm_assessment: Optional[RiskAssessmentRecord],
    fuzzy_assessment: Optional[RiskAssessmentRecord],
    contract_version: Optional[str],
    site_label: Optional[str] = None,
    is_edge: bool = False,
    forecast: Optional[ForecastResult] = None,
    reassessment_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose the full `PestSessionReportRequest`-shaped payload dict for
    one session. `pdm_assessment` is the AUTHORITATIVE-slot path -- the PDM record for cloud/mixed, the edge score-band
    record for edge (edge_report.py's own docstring); `fuzzy_assessment` is
    the labeled comparative path -- never merged into one number, always `None` for edge. `is_edge` selects the single edge verdict sentence and the
    weather-not-applicable reason. `site_label` is
    the deployment-configured value the caller reads off `Settings`; falls
    back to `DEFAULT_SITE_LABEL` when unset (a None/empty default must
    never reach the payload). `forecast` is the session's forecast outlook
    as the caller obtained it (`forecast_builder.build_forecast_outlook`),
    or its `ForecastUnavailable` reason; None when none was attempted.
    `reassessment_of` is the id of the on-device report when this is the
    cloud re-assessment of a synced edge session: it adds the provenance line
    naming it (`reassessment_provenance_line`); None adds nothing."""
    stock_mode = envelope.provenance.designation_status != DESIGNATED_STATUS
    identity = resolve_pest_identity(risk_input.crop, envelope.pest)

    authoritative_band = (
        pdm_assessment.fused_verdict
        if pdm_assessment is not None and pdm_assessment.status == "present"
        else None
    )

    if is_edge:
        verdict_sentences = [_fused_sentence(FUSED_VERDICT_EDGE_MARKER, pdm_assessment)]
    else:
        verdict_sentences = [
            _fused_sentence(FUSED_VERDICT_AUTHORITATIVE_MARKER, pdm_assessment),
            _fused_sentence(FUSED_VERDICT_COMPARATIVE_MARKER, fuzzy_assessment),
        ]
    engine_provenance = _engine_provenance_lines(pdm_assessment, fuzzy_assessment, is_edge)
    if reassessment_of:
        engine_provenance = [*(engine_provenance or []), reassessment_provenance_line(reassessment_of)]

    pdm_details, pdm_details_full = _composed_risk_details(pdm_assessment, risk_input, authoritative_band)
    fuzzy_details = _composed_fuzzy_details(fuzzy_assessment)

    if stock_mode:
        conclusion = None
        recommendation = None
        pest_crop_context: Optional[Dict[str, Any]] = None
        pest_crop_context_unavailable_reason: Optional[str] = None
        stock_notice = compose_stock_model_notice(
            detector_impl_id=envelope.provenance.detector_impl_id,
            detector_version=envelope.provenance.detector_version,
            designation_status=envelope.provenance.designation_status,
        )
        risk_assessment_details = f"{stock_notice} {pdm_details}" if pdm_details else stock_notice
    else:
        conclusion = (
            compose_conclusion(authoritative_band, identity, risk_input.crop)
            if authoritative_band is not None
            else compose_conclusion_unavailable(risk_input.crop)
        )
        recommendation = (
            compose_recommendation(authoritative_band, identity, risk_input.crop)
            if authoritative_band is not None
            else None
        )
        pest_crop_context = (
            {
                "pest_name": identity.pest_name,
                "pest_scientific_name": identity.scientific_name,
                "pest_eppo_code": identity.eppo_code,
                "pest_eppo_url": identity.eppo_url,
                "crop_name": crop_display_name(risk_input.crop),
                "crop_scientific": risk_input.crop,
            }
            if identity is not None
            else None
        )
        pest_crop_context_unavailable_reason = (
            compose_pest_crop_context_unavailable(risk_input.crop, envelope.pest)
            if identity is None and envelope.pest
            else None
        )
        risk_assessment_details = pdm_details

    risk_reasoning: Dict[str, Any] = {
        "nn_score": risk_input.infestation_score,
        "nn_verdict": fusion_nn_verdict(risk_input.infestation_score),
        "nn_note": "\n".join(verdict_sentences),
        "weather_window": (
            (pdm_assessment.weather_window if pdm_assessment else None)
            or (fuzzy_assessment.weather_window if fuzzy_assessment else None)
        ),
        "combined_level": authoritative_band or "n/a",
        "narrative": conclusion,
        "engine_provenance": engine_provenance,
    }

    forecast_outlook = forecast_outlook_payload(None if stock_mode else forecast, recommendation)

    return {
        "contract_version": contract_version,
        "envelope": envelope.model_dump(mode="json", exclude_none=True),
        "infestation_score": risk_input.infestation_score,
        "risk_assessment": {
            "level": authoritative_band or "n/a",
            "details": risk_assessment_details,
            "assessment_details_full": pdm_details_full,
            "fuzzy_details": fuzzy_details,
        },
        "risk_index": None,
        "gdd": None,
        "weather_summary": _weather_summary_payload(risk_input, fuzzy_assessment, is_edge),
        "pest_crop_context": pest_crop_context,
        "pest_crop_context_unavailable_reason": pest_crop_context_unavailable_reason,
        "risk_reasoning": risk_reasoning,
        "forecast_outlook": forecast_outlook,
        "site_label": site_label or DEFAULT_SITE_LABEL,
    }
