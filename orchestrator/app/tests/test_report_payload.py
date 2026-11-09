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
Report content composition: conclusions and actionable recommendations; pest, crop and model
provenance; no growth stage; the integration-only stock-model mode; both risk paths' fused verdicts
presented without cross-path merging; fetched weather reaches the report and absent weather is
explained; sentence-form verdicts; the edge report keeps its single-record surface.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pypdf
import pytest

from config import Settings
from contract import DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput
from fusion import FUSION_RULE_VERSION
from models import RiskAssessmentRecord
from report_content import (
    CONCLUSION_MARKER,
    crop_display_name,
    DEFAULT_SITE_LABEL,
    FUSED_VERDICT_AUTHORITATIVE_MARKER,
    FUSED_VERDICT_COMPARATIVE_MARKER,
    FUSED_VERDICT_EDGE_MARKER,
    RECOMMENDATION_MARKER,
    STOCK_MODEL_NOTICE_MARKER,
)
from report_pdf.pest_session_pdf_core import render_pest_session_pdf
from report_payload import ReportComposeError, compose_report_payload


def _risk_input(**overrides) -> RiskInput:
    fields = dict(
        session_id="sess-1",
        infestation_score=0.8,
        crop="brassica-oleracea",
        growth_stage=GrowthStage(representation="enum", unknown=False, stage_id="BBCH-40"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    fields.update(overrides)
    return RiskInput(**fields)


def _envelope(designation_status: str = "designated", **overrides) -> ReportingEnvelope:
    fields = dict(
        session_id="sess-1",
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
    fields.update(overrides)
    return ReportingEnvelope(**fields)


def _pdm_row(
    fused_verdict="High", status="present", engine_version="0.2.0", details=None, provenance=None
) -> RiskAssessmentRecord:
    return RiskAssessmentRecord(
        session_id="sess-1",
        path="pdm",
        status=status,
        level="moderate",
        details=details,
        provenance_json=json.dumps(provenance) if provenance is not None else None,
        fused_verdict=fused_verdict if status == "present" else None,
        fusion_rule_version=FUSION_RULE_VERSION if status == "present" else None,
        engine_identity="pdm-rule-engine",
        engine_version=engine_version,
        weather_window="2026-01-01T00:00:00Z  →  2026-01-02T00:00:00Z",
        failure_reason=None if status == "present" else "forced failure (test)",
    )


def _edge_row(fused_verdict="High", status="present", engine_version="v1") -> RiskAssessmentRecord:
    """The edge score-band record shape service.py's
    `_attempt_edge_score_band_path` persists -- passed into the
    `pdm_assessment` slot for edge composition (edge_report.py's own
    docstring)."""
    return RiskAssessmentRecord(
        session_id="sess-1",
        path="edge-score-band",
        status=status,
        level=None,
        fused_verdict=fused_verdict if status == "present" else None,
        fusion_rule_version=FUSION_RULE_VERSION if status == "present" else None,
        engine_identity="edge-score-band",
        engine_version=engine_version,
        weather_window=None,
        failure_reason=None if status == "present" else "forced failure (test)",
    )


def _weather_rule_provenance(**overrides) -> dict:
    """The weather-rule path's PDM provenance as the PDM now reports it: one
    pest model of two rules over a 24-hour window, the first rule holding
    for 6 hours."""
    provenance = {
        "weather_rule_based": True,
        "degraded_reason": None,
        "weather_window": "2026-01-01T00:00:00Z  \u2192  2026-01-02T00:00:00Z",
        "weather_rule_level_raw": "moderate",
        "ruleset_version": "1.0.0",
        "pdm_app_version": "0.2.0",
        "rules_evaluated": [
            {
                "model": "Cabbage aphid",
                "dominant_level": "moderate",
                "rules": [
                    {
                        "level": "moderate",
                        "conditions": "temperature_2m >= 20  AND  relative_humidity_2m >= 70",
                        "hours_triggered": 6,
                        "hours_total": 24,
                    },
                    {
                        "level": "high",
                        "conditions": "temperature_2m >= 30",
                        "hours_triggered": 0,
                        "hours_total": 24,
                    },
                ],
            }
        ],
        "observed_weather": {
            "temperature_2m_avg_C": 21.3,
            "temperature_2m_max_C": 27.8,
            "relative_humidity_2m_avg_pct": 71.0,
            "precipitation_mm": 1.25,
        },
        "nn_band": "Moderate",
        "nn_band_boundaries": {"high": 0.70, "moderate": 0.40},
        "combination_rule": "max(weather_band, detection_band)",
    }
    provenance.update(overrides)
    return provenance


def _weather_summary_provenance(
    temperature_min_C=12.0, temperature_max_C=24.0, relative_humidity_avg_pct=55.0, source="openmeteo",
) -> str:
    """The shape service.py's `_attempt_fuzzy_path` persists into
    the fuzzy `RiskAssessmentRecord.provenance_json` -- reused here so
    fixtures match production persistence, not an invented shape."""
    return json.dumps(
        {
            "daily_scores": [{"date": "2026-01-01", "value": 40.0, "risk_level": "moderate"}],
            "threat_definition_version": "1.0.0",
            "dossier_version": "v2",
            "vision_score_seen": False,
            "weather_summary": {
                "temperature_min_C": temperature_min_C,
                "temperature_max_C": temperature_max_C,
                "relative_humidity_avg_pct": relative_humidity_avg_pct,
                "source": source,
            },
        }
    )


def _fuzzy_row(
    fused_verdict="Moderate", status="present", engine_version="2.0.0", with_weather_summary=True
) -> RiskAssessmentRecord:
    return RiskAssessmentRecord(
        session_id="sess-1",
        path="fuzzy",
        status=status,
        level="moderate",
        fused_verdict=fused_verdict if status == "present" else None,
        fusion_rule_version=FUSION_RULE_VERSION if status == "present" else None,
        engine_identity="fuzzy-mamdani-risk",
        engine_version=engine_version,
        weather_window="2026-01-01T00:00:00Z  →  2026-01-02T00:00:00Z",
        failure_reason=None if status == "present" else "forced failure (test)",
        provenance_json=_weather_summary_provenance() if (status == "present" and with_weather_summary) else None,
    )


def _walk_strings(value):
    """Yield every string value reachable from a nested dict/list structure --
    used by the assertion to scan the WHOLE composed payload, not
    just the fields this module happens to know about."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


class TestConclusionsAndRecommendationsRenderFromLiveData:
    """Recommendation renders from live data -- a designated
    model + a completed authoritative (PDM-path) assessment."""

    def test_populated_conclusion_and_recommendation_present(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.2.0",
        )

        narrative = payload["risk_reasoning"]["narrative"]
        recommendation = payload["forecast_outlook"]["recommendation"]
        assert CONCLUSION_MARKER in narrative
        assert "HIGH" in narrative
        assert RECOMMENDATION_MARKER in recommendation
        assert "BRVCBR" in recommendation

    def test_identifiers_and_provenance_present(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )

        assert payload["pest_crop_context"]["pest_eppo_code"] == "BRVCBR"
        provenance = payload["envelope"]["provenance"]
        assert provenance["detector_version"] == "1.0.0"
        assert provenance["designation_status"] == "designated"


class TestGrowthStageNeverInComposedPayload:
    """Growth stage absent from report content -- regardless of
    whether the session's risk input carried a concrete growth stage or the
    unknown marker."""

    def test_no_growth_stage_field_label_or_value_with_concrete_stage(self):
        payload = compose_report_payload(
            risk_input=_risk_input(
                growth_stage=GrowthStage(representation="enum", unknown=False, stage_id="BBCH-40")
            ),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )
        strings = list(_walk_strings(payload))
        assert not any("growth" in s.lower() for s in strings)
        assert not any("bbch" in s.lower() for s in strings)
        assert "growth_stage" not in payload["envelope"]

    def test_no_growth_stage_field_label_or_value_with_unknown_stage(self):
        payload = compose_report_payload(
            risk_input=_risk_input(growth_stage=GrowthStage.unknown_for("enum")),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )
        strings = list(_walk_strings(payload))
        assert not any("growth" in s.lower() for s in strings)


class TestIntegrationOnlyStockModelSuppression:
    """Non-designated provenance -> visible generic-model notice,
    zero agricultural conclusion/recommendation content."""

    def test_stock_provenance_renders_notice_and_zero_agricultural_markers(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="stub"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.2.0",
        )

        assert payload["risk_assessment"]["details"] is not None
        assert STOCK_MODEL_NOTICE_MARKER in payload["risk_assessment"]["details"]
        assert "lookup-detector" in payload["risk_assessment"]["details"]

        assert payload["pest_crop_context"] is None
        assert payload["forecast_outlook"] is None
        assert payload["risk_reasoning"]["narrative"] is None

        strings = list(_walk_strings(payload))
        assert not any(CONCLUSION_MARKER in s for s in strings)
        assert not any(RECOMMENDATION_MARKER in s for s in strings)

    def test_pdm_weather_rule_technical_text_survives_alongside_the_notice(self):
        """The weather-rule facts are technical computation facts, not agricultural advice, so
        suppression keeps them. Since the text is composed from the PDM provenance, not PDM's own generic
        sentence."""
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="stub"),
            pdm_assessment=_pdm_row(provenance=_weather_rule_provenance()),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )
        details = payload["risk_assessment"]["details"]
        assert STOCK_MODEL_NOTICE_MARKER in details
        assert "Weather conditions alone: Medium risk." in details
        assert "held for 6 of 24 hours" in payload["risk_assessment"]["assessment_details_full"]

    def test_integration_only_designation_also_suppresses(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="integration-only"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )
        assert STOCK_MODEL_NOTICE_MARKER in payload["risk_assessment"]["details"]

    def test_notice_is_not_silent_omission(self):
        """Suppression is not silent omission -- the notice
        is deterministically detectable, not merely an absence."""
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="stub"),
            pdm_assessment=None,
            fuzzy_assessment=None,
            contract_version="1.2.0",
        )
        assert STOCK_MODEL_NOTICE_MARKER in payload["risk_assessment"]["details"]

    def test_fused_verdict_lines_still_present_in_stock_mode(self):
        """The dual fused-verdict labels are risk numbers, not
        agricultural advice -- suppression does not apply to them."""
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="stub"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.2.0",
        )
        note = payload["risk_reasoning"]["nn_note"]
        assert FUSED_VERDICT_AUTHORITATIVE_MARKER in note
        assert FUSED_VERDICT_COMPARATIVE_MARKER in note


class TestBothPathsFusedVerdictsLabeledNoMerge:
    """Both paths' fused verdicts, labeled (authoritative PDM-path +
    comparative fuzzy-path), each with engine identity/version and the
    fusion-rule version -- never merged into one unlabeled value."""

    def test_two_labeled_fused_verdicts_with_engine_and_fusion_version(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High", engine_version="0.2.0"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate", engine_version="2.0.0"),
            contract_version="1.2.0",
        )
        note = payload["risk_reasoning"]["nn_note"]
        lines = note.split("\n")
        authoritative_line = next(l for l in lines if FUSED_VERDICT_AUTHORITATIVE_MARKER in l)
        comparative_line = next(l for l in lines if FUSED_VERDICT_COMPARATIVE_MARKER in l)

        assert "High" in authoritative_line
        assert "Moderate" in comparative_line
        assert authoritative_line != comparative_line
        assert "High" not in comparative_line
        assert "Moderate" not in authoritative_line

        for line in (authoritative_line, comparative_line):
            assert "pdm-rule-engine" not in line
            assert "fuzzy-mamdani-risk" not in line
            assert "fusion" not in line.lower()

        provenance = payload["risk_reasoning"]["engine_provenance"]
        auth_prov = next(l for l in provenance if l.startswith("authoritative path:"))
        comp_prov = next(l for l in provenance if l.startswith("comparative path:"))
        assert auth_prov == f"authoritative path: pdm-rule-engine 0.2.0, fusion {FUSION_RULE_VERSION}"
        assert comp_prov == f"comparative path: fuzzy-mamdani-risk 2.0.0, fusion {FUSION_RULE_VERSION}"

    def test_authoritative_verdict_alone_drives_risk_assessment_level(self):
        """The PDM-path fused verdict is authoritative (drives
        conclusions/recommendations/risk_assessment.level); the fuzzy path
        never overrides it even when the two diverge."""
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Low"),
            contract_version="1.2.0",
        )
        assert payload["risk_assessment"]["level"] == "High"
        assert payload["risk_reasoning"]["combined_level"] == "High"

    def test_absent_path_reports_status_not_a_fabricated_verdict(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(status="failed"),
            contract_version="1.2.0",
        )
        note = payload["risk_reasoning"]["nn_note"]
        comparative_line = next(l for l in note.split("\n") if FUSED_VERDICT_COMPARATIVE_MARKER in l)
        assert "n/a" in comparative_line
        assert "status=failed" in comparative_line

    def test_authoritative_path_absent_yields_honest_degraded_conclusion(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=None,
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.2.0",
        )
        assert payload["risk_assessment"]["level"] == "n/a"
        assert CONCLUSION_MARKER in payload["risk_reasoning"]["narrative"]
        assert payload["forecast_outlook"] is None


class TestVerdictSentencesEqualPinnedLiterals:
    """These constants are pinned literals; an assertion mirroring a wrong constant cannot pass,
    because both are judged
    against the literals here."""

    def test_constants_equal_the_pinned_literals(self):
        assert FUSED_VERDICT_AUTHORITATIVE_MARKER == "Weather-rule assessment: "
        assert FUSED_VERDICT_COMPARATIVE_MARKER == "Fuzzy assessment: "
        assert FUSED_VERDICT_EDGE_MARKER == "Edge assessment (score band): "
        assert CONCLUSION_MARKER == "Conclusion: "
        assert RECOMMENDATION_MARKER == "Recommended action: "
        assert STOCK_MODEL_NOTICE_MARKER == (
            "Note: detections in this report were produced by a generic "
            "integration model not designated for agricultural claims"
        )
        assert DEFAULT_SITE_LABEL == "Field site"


class TestCloudMixedComposeBothSentencesEdgeComposesOne:
    """Cloud/mixed reports carry both verdict sentences; edge reports carry exactly one."""

    def test_cloud_mixed_report_carries_both_sentences(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="Low"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Low"),
            contract_version="1.2.0",
        )
        note = payload["risk_reasoning"]["nn_note"]
        assert f"{FUSED_VERDICT_AUTHORITATIVE_MARKER}Low" in note
        assert f"{FUSED_VERDICT_COMPARATIVE_MARKER}Low" in note
        assert FUSED_VERDICT_EDGE_MARKER not in note

    def test_edge_report_carries_exactly_one_sentence_with_edge_surface(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_edge_row(fused_verdict="High"),
            fuzzy_assessment=None,
            contract_version="1.2.0",
            is_edge=True,
        )
        note = payload["risk_reasoning"]["nn_note"]
        lines = note.split("\n")
        assert len(lines) == 1
        assert lines[0] == f"{FUSED_VERDICT_EDGE_MARKER}High"
        assert FUSED_VERDICT_AUTHORITATIVE_MARKER not in note
        assert FUSED_VERDICT_COMPARATIVE_MARKER not in note

    def test_edge_engine_provenance_has_no_fusion_clause(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_edge_row(fused_verdict="High", engine_version="v1"),
            fuzzy_assessment=None,
            contract_version="1.2.0",
            is_edge=True,
        )
        provenance = payload["risk_reasoning"]["engine_provenance"]
        assert provenance == ["authoritative path: edge-score-band v1"]
        assert "fusion" not in provenance[0]


_REASSESSMENT_LINE = (
    "Cloud re-assessment after synchronisation of the on-device report edge-report-uuid-1: "
    "the weather rules, the fuzzy path and the outlook are computed from the recorded weather "
    "of the session window; the on-device report keeps its score-band assessment."
)


class TestCloudReassessmentProvenanceLine:
    """A cloud
    re-assessment of a synced edge session ends its engine-provenance lines
    with one pinned line naming the on-device report; without it, no line."""

    def _compose(self, **kwargs):
        return compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High", engine_version="0.2.0"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate", engine_version="2.0.0"),
            contract_version="1.2.0",
            **kwargs,
        )

    def test_the_line_is_appended_last_with_its_exact_text(self):
        provenance = self._compose(reassessment_of="edge-report-uuid-1")["risk_reasoning"]["engine_provenance"]

        assert provenance[-1] == _REASSESSMENT_LINE
        assert len(provenance) == 3
        assert provenance[0].startswith("authoritative path:")
        assert provenance[1].startswith("comparative path:")

    def test_no_line_by_default(self):
        provenance = self._compose()["risk_reasoning"]["engine_provenance"]

        assert len(provenance) == 2
        assert not any("re-assessment" in line for line in provenance)

    def test_the_line_stands_alone_when_no_path_has_a_record(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=None,
            fuzzy_assessment=None,
            contract_version="1.2.0",
            reassessment_of="edge-report-uuid-1",
        )

        assert payload["risk_reasoning"]["engine_provenance"] == [_REASSESSMENT_LINE]


class TestVerdictSentencesCarryNoEngineIdentifiers:
    def test_sentences_contain_no_engine_name_version_or_fusion_token(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High", engine_version="0.2.0"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate", engine_version="2.0.0"),
            contract_version="1.2.0",
        )
        note = payload["risk_reasoning"]["nn_note"]
        assert "pdm-rule-engine" not in note
        assert "fuzzy-mamdani-risk" not in note
        assert "0.2.0" not in note
        assert "2.0.0" not in note
        assert "fusion" not in note.lower()


def _real_settings(monkeypatch, **env) -> Settings:
    """`Settings` (config.py) is the real config object
    `service.py`/`edge_report.py` read `report_site_label` off of
    (`getattr(settings, "report_site_label", None)`) before passing it to
    the composer -- a test that hands `site_label=` straight to
    `compose_report_payload` never exercises that config path at all.
    Edge profile keeps the settings-construction requirements minimal (no
    Gatekeeper credential pair needed)."""
    monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
    monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", "a-real-unique-edge-secret")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


class TestSiteLabelConfigIdentity:
    """Passed through unchanged -- a
    None/empty default cannot pass; a configured value survives
    byte-for-byte (the passthrough property that makes a geocoding call
    structurally pointless). Routed through the real `Settings` object
    (set and unset), not a literal handed directly to the composer."""

    def test_unset_setting_emits_the_default_label(self, monkeypatch):
        monkeypatch.delenv("ORCH_REPORT_SITE_LABEL", raising=False)
        settings = _real_settings(monkeypatch)
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
            site_label=getattr(settings, "report_site_label", None),
        )
        assert payload["site_label"] == DEFAULT_SITE_LABEL
        assert payload["site_label"]

    def test_empty_string_setting_also_falls_back_to_default(self, monkeypatch):
        settings = _real_settings(monkeypatch, ORCH_REPORT_SITE_LABEL="")
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
            site_label=getattr(settings, "report_site_label", None),
        )
        assert payload["site_label"] == DEFAULT_SITE_LABEL

    def test_configured_value_passes_through_unchanged(self, monkeypatch):
        settings = _real_settings(monkeypatch, ORCH_REPORT_SITE_LABEL="Example region")
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
            site_label=getattr(settings, "report_site_label", None),
        )
        assert payload["site_label"] == "Example region"


class TestWeatherSummaryFetchedReachesTheReport:
    """Fetched weather reaches the report."""

    def test_populated_summary_carries_figures_and_source(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.2.0",
        )
        weather = payload["weather_summary"]
        assert weather["source"] == "openmeteo"
        assert weather["observed"]["temperature_2m_max_C"] == 24.0
        assert weather["observed"]["temperature_2m_min_C"] == 12.0
        assert weather["observed"]["relative_humidity_2m_avg_pct"] == 55.0
        assert "unavailable_reason" not in weather


class TestWeatherSummaryAbsentIsExplained:
    """Absent weather is explained -- each degraded reason from the existing taxonomy, in its
    human form."""

    def test_no_coordinates_yields_position_unavailable(self):
        payload = compose_report_payload(
            risk_input=_risk_input(latitude=None, longitude=None),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=None,
            contract_version="1.2.0",
        )
        assert payload["weather_summary"] == {"unavailable_reason": "position unavailable"}

    def test_fuzzy_path_failed_yields_weather_unavailable(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(status="failed"),
            contract_version="1.2.0",
        )
        assert payload["weather_summary"] == {"unavailable_reason": "weather unavailable"}

    def test_edge_path_yields_weather_not_applicable(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_edge_row(),
            fuzzy_assessment=None,
            contract_version="1.2.0",
            is_edge=True,
        )
        assert payload["weather_summary"] == {"unavailable_reason": "weather not applicable"}


class TestSuccessfulFuzzyResultMissingSummaryIsRefused:
    """A successful fuzzy result missing its summary is refused."""

    def test_present_status_without_weather_summary_raises(self):
        with pytest.raises(ReportComposeError):
            compose_report_payload(
                risk_input=_risk_input(),
                envelope=_envelope(designation_status="designated"),
                pdm_assessment=_pdm_row(),
                fuzzy_assessment=_fuzzy_row(with_weather_summary=False),
                contract_version="1.2.0",
            )


class TestFuzzyDetailsInThePayload:
    """`risk_assessment.fuzzy_details` is the fuzzy path's own composed paragraph (a completed path), the
    not-evaluated sentence (a failed path), or absent (no fuzzy path at
    all, edge sessions)."""

    _EXPLANATION = {
        "days": [
            {
                "date": "2026-01-01",
                "temperature_avg_C": 21.3,
                "humidity_max_pct": 71.0,
                "rain_mm": 0.0,
                "matched_rule_levels": ["moderate"],
            }
        ],
        "humidity_streak_days": 0.0,
        "min_streak_days": 2,
        "wettest_day_wet_hours": 4.0,
        "min_wetness_hours_high": 24.0,
        "rule_count": 3,
        "score_low": 40.0,
        "score_high": 40.0,
        "dominant_band": "moderate",
    }

    def _fuzzy_row_with_explanation(self):
        return RiskAssessmentRecord(
            session_id="sess-1",
            path="fuzzy",
            status="present",
            level="moderate",
            fused_verdict="Moderate",
            fusion_rule_version=FUSION_RULE_VERSION,
            engine_identity="fuzzy-mamdani-risk",
            engine_version="2.0.0",
            weather_window="2026-01-01T00:00:00Z  →  2026-01-02T00:00:00Z",
            provenance_json=json.dumps(
                {
                    "daily_scores": [{"date": "2026-01-01", "value": 40.0, "risk_level": "moderate"}],
                    "threat_definition_version": "1.0.0",
                    "dossier_version": "v2",
                    "vision_score_seen": False,
                    "weather_summary": {
                        "temperature_min_C": 12.0, "temperature_max_C": 24.0,
                        "relative_humidity_avg_pct": 55.0, "source": "openmeteo",
                    },
                    "explanation": self._EXPLANATION,
                }
            ),
        )

    def test_a_completed_fuzzy_path_composes_the_paragraph(self):
        from report_content import compose_fuzzy_details

        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=self._fuzzy_row_with_explanation(),
            contract_version="1.2.0",
        )

        assert payload["risk_assessment"]["fuzzy_details"] == compose_fuzzy_details(self._EXPLANATION)

    def test_a_failed_fuzzy_path_says_why(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(status="failed"),
            contract_version="1.2.0",
        )

        assert payload["risk_assessment"]["fuzzy_details"] == (
            "The fuzzy model was not evaluated: forced failure (test)."
        )

    def test_an_edge_session_carries_no_fuzzy_details(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_edge_row(),
            fuzzy_assessment=None,
            contract_version="1.2.0",
            is_edge=True,
        )

        assert payload["risk_assessment"]["fuzzy_details"] is None


class TestKaleSessionRendersItsCropAndPest:
    """The crop `brassica-oleracea-sabellica`
    carries the Brevicoryne brassicae identity and its own display name."""

    KALE = "brassica-oleracea-sabellica"
    KALE_NAME = "Kale (Brassica oleracea var. sabellica)"

    def _payload(self, crop):
        return compose_report_payload(
            risk_input=_risk_input(crop=crop),
            envelope=_envelope(designation_status="designated"),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.5.0",
        )

    def _rendered_text(self, crop):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "session.pdf")
            render_pest_session_pdf(self._payload(crop), lambda variant: None, output_path)
            reader = pypdf.PdfReader(output_path)
            return "".join("".join((page.extract_text() or "").split()) for page in reader.pages)

    def test_display_name_is_the_kale_string_and_cabbage_is_unchanged(self):
        assert crop_display_name(self.KALE) == self.KALE_NAME
        assert crop_display_name("brassica-oleracea") == "Cabbage / Brassica crops (Brassica oleracea)"

    def test_payload_carries_the_pest_context_and_the_display_name(self):
        context = self._payload(self.KALE)["pest_crop_context"]

        assert context["pest_eppo_code"] == "BRVCBR"
        assert context["pest_scientific_name"] == "Brevicoryne brassicae"
        assert context["crop_name"] == self.KALE_NAME
        assert context["crop_scientific"] == self.KALE

    def test_recommendation_names_the_pest_on_kale(self):
        recommendation = self._payload(self.KALE)["forecast_outlook"]["recommendation"]

        assert "BRVCBR" in recommendation
        assert self.KALE_NAME in recommendation

    def test_the_rendered_report_names_kale_the_pest_and_its_code(self):
        text = self._rendered_text(self.KALE)

        for literal in (self.KALE_NAME, "Brevicoryne brassicae", "BRVCBR"):
            assert "".join(literal.split()) in text, literal

    def test_the_host_crop_line_carries_no_raw_crop_code(self):
        """The display name only:
        crop_scientific (the raw crop code) never rides on the line."""
        text = self._rendered_text(self.KALE)

        assert "".join(f"Host Crop {self.KALE_NAME}".split()) in text
        assert "".join(f"({self.KALE})".split()) not in text

    def test_a_cabbage_session_is_unchanged(self):
        context = self._payload("brassica-oleracea")["pest_crop_context"]

        assert context["pest_eppo_code"] == "BRVCBR"
        assert context["pest_scientific_name"] == "Brevicoryne brassicae"
        assert context["crop_name"] == "Cabbage / Brassica crops (Brassica oleracea)"


GENERIC_PDM_SENTENCE = "Highest risk level observed in the hourly window"

FULL_WEATHER_RULE_PARAGRAPH = (
    "Weather rules for Cabbage aphid over the session window: 'temperature at least "
    "20 \u00b0C and relative humidity at least 70 %' held for 6 of 24 hours (Medium risk); "
    "'temperature at least 30 \u00b0C' held for 0 of 24 hours (High risk). These rules put "
    "the weather risk at Medium. The window had an average temperature of 21.3 \u00b0C "
    "(maximum 27.8 \u00b0C) and an average relative humidity of 71.0 %, with 1.25 mm of "
    "precipitation. The detections put the session in the Medium band; the detector's "
    "confidence on the most affected images is 0.45. The risk level is the higher of the "
    "two: Medium."
)
SHORT_WEATHER_RULE_SENTENCE = "Weather conditions alone: Medium risk. Combined with the detections: Medium."
FULL_DEGRADED_PARAGRAPH = (
    "Score-only assessment: the weather was not evaluated (weather unavailable). "
    "The level comes from the infestation score (0.62) alone."
)
SCORE_ONLY_SENTENCE = "Based on the detections in this session; no weather data for this assessment."


def _layout_lines(payload) -> list:
    """`pdftotext -layout` lines of the rendered report, whitespace-collapsed."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.pdf"
        render_pest_session_pdf(payload, lambda variant: None, str(out))
        text = subprocess.run(
            ["pdftotext", "-layout", str(out), "-"], capture_output=True, text=True, check=True
        ).stdout
    return [" ".join(line.split()) for line in text.splitlines()]


class TestRiskDetailsAreComposedFromTheAssessmentFacts:
    """The risk-assessment details are composed from the assessment's facts, in two levels: one short
    sentence for the Risk Assessment block, the full
    paragraph for the bottom of the report."""

    def _both(self, provenance, score=0.45, fused_verdict="Medium", **kwargs):
        payload = compose_report_payload(
            risk_input=_risk_input(infestation_score=score),
            envelope=_envelope(),
            pdm_assessment=_pdm_row(
                fused_verdict=fused_verdict,
                details="Weather-rule-based risk assessment. " + GENERIC_PDM_SENTENCE + ".",
                provenance=provenance,
            ),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.5.0",
            **kwargs,
        )
        return payload["risk_assessment"]["details"], payload["risk_assessment"]["assessment_details_full"]

    def test_weather_rule_details_are_one_short_sentence_and_a_full_paragraph(self):
        short, full = self._both(_weather_rule_provenance())

        assert short == SHORT_WEATHER_RULE_SENTENCE
        assert full == FULL_WEATHER_RULE_PARAGRAPH
        for text in (short, full):
            assert GENERIC_PDM_SENTENCE not in text
            assert "temperature_2m" not in text and "relative_humidity_2m" not in text

    def test_the_full_paragraph_opens_with_the_pinned_weather_rules_phrase(self):
        _, full = self._both(_weather_rule_provenance())

        assert full.startswith("Weather rules for Cabbage aphid over the session window:")
        assert "Score-only assessment" not in full
        assert "Score-only/degraded risk assessment" not in full

    def test_the_level_is_the_authoritative_level_the_report_shows(self):
        short, full = self._both(
            _weather_rule_provenance(nn_band="High"), score=0.85, fused_verdict="High"
        )
        assert short == "Weather conditions alone: Medium risk. Combined with the detections: High."
        assert "put the session in the High band; the detector's confidence on the most affected images is 0.85" in full
        assert full.endswith("The risk level is the higher of the two: High.")

    def test_a_clause_whose_facts_are_absent_is_omitted(self):
        provenance = _weather_rule_provenance()
        del provenance["observed_weather"]["precipitation_mm"]
        del provenance["observed_weather"]["temperature_2m_max_C"]
        _, full = self._both(provenance)
        assert "average temperature of 21.3 \u00b0C and an average relative humidity of 71.0 %." in full
        assert "precipitation" not in full
        assert "maximum" not in full

        for key in ("observed_weather", "rules_evaluated"):
            del provenance[key]
        _, full = self._both(provenance)
        assert full.startswith("The detections put the session in the Medium band")
        assert "temperature" not in full

    def test_details_without_rule_facts_do_not_claim_the_combination(self):
        assert self._both({"weather_rule_based": True, "degraded_reason": None}) == (None, None)

    def test_a_degraded_assessment_says_score_only_and_names_the_reason(self):
        short, full = self._both(
            {
                "weather_rule_based": False,
                "degraded_reason": "weather_unavailable",
                "weather_window": None,
                "weather_rule_level_raw": None,
            },
            score=0.62,
            fused_verdict="Medium",
        )
        assert short == SCORE_ONLY_SENTENCE
        assert full == FULL_DEGRADED_PARAGRAPH
        assert GENERIC_PDM_SENTENCE not in full

    def test_the_edge_record_is_composed_as_score_only(self):
        payload = compose_report_payload(
            risk_input=_risk_input(infestation_score=0.62),
            envelope=_envelope(),
            pdm_assessment=_edge_row(fused_verdict="Medium"),
            fuzzy_assessment=None,
            contract_version="1.5.0",
            is_edge=True,
        )
        assert payload["risk_assessment"]["details"] is None
        assert payload["risk_assessment"]["assessment_details_full"] is None

        edge = _edge_row(fused_verdict="Medium")
        edge.provenance_json = json.dumps(
            {"weather_rule_based": False, "degraded_reason": "weather_not_applicable", "edge_local": True}
        )
        payload = compose_report_payload(
            risk_input=_risk_input(infestation_score=0.62),
            envelope=_envelope(),
            pdm_assessment=edge,
            fuzzy_assessment=None,
            contract_version="1.5.0",
            is_edge=True,
        )
        assert payload["risk_assessment"]["details"] == SCORE_ONLY_SENTENCE
        assert payload["risk_assessment"]["assessment_details_full"] == (
            "Score-only assessment: the weather was not evaluated (weather not applicable). "
            "The level comes from the infestation score (0.62) alone."
        )

    def _rendered(self, provenance, score=0.45, fused_verdict="Medium"):
        payload = compose_report_payload(
            risk_input=_risk_input(infestation_score=score),
            envelope=_envelope(),
            pdm_assessment=_pdm_row(fused_verdict=fused_verdict, details=GENERIC_PDM_SENTENCE, provenance=provenance),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.5.0",
        )
        return _layout_lines(payload)

    def test_the_short_line_sits_in_the_risk_assessment_block_and_the_paragraph_after_the_image(self):
        lines = self._rendered(_weather_rule_provenance())
        text = "\n".join(lines)

        risk_at = lines.index("Risk Assessment")
        context_at = lines.index("Pest & Crop Context")
        short_at = lines.index(f"Details {SHORT_WEATHER_RULE_SENTENCE}")
        assert risk_at < short_at < context_at

        image_at = lines.index("Session image")
        details_heading_at = lines.index("Assessment details")
        inference_at = lines.index("Inference Details")
        assert image_at < inference_at < details_heading_at
        paragraph = " ".join(lines[details_heading_at + 1:])
        assert FULL_WEATHER_RULE_PARAGRAPH in paragraph
        assert text.count("held for 6 of 24 hours") == 1
        assert not [line for line in lines[:image_at] if "held for 6 of 24 hours" in line]

    def test_no_raw_identifier_and_no_generic_sentence_reach_the_report(self):
        lines = self._rendered(_weather_rule_provenance())
        whole = " ".join(lines)
        bottom = " ".join(lines[lines.index("Assessment details"):])

        assert "temperature_2m" not in whole
        assert "relative_humidity_2m" not in whole
        assert GENERIC_PDM_SENTENCE not in whole
        assert "21.3" in bottom and "71.0" in bottom

    def test_the_short_line_ends_with_the_level_both_verdict_rows_carry(self):
        payload = compose_report_payload(
            risk_input=_risk_input(infestation_score=0.85),
            envelope=_envelope(),
            pdm_assessment=_pdm_row(
                fused_verdict="High",
                provenance=_weather_rule_provenance(weather_rule_level_raw="low", nn_band="High"),
            ),
            fuzzy_assessment=_fuzzy_row(fused_verdict="High"),
            contract_version="1.5.0",
        )
        lines = _layout_lines(payload)

        sentence = "Weather conditions alone: Low risk. Combined with the detections: High."
        assert f"Details {sentence}" in lines
        assert "Weather-rule assessment: High" in lines
        assert "Fuzzy assessment: High" in lines
        for former in ("Authoritative assess" + "ment (", "Comparative assess" + "ment ("):
            assert former not in " ".join(lines)

    def test_weather_rule_text_holds_on_the_rendered_report(self):
        text = " ".join(self._rendered(_weather_rule_provenance()))

        assert "Weather rules for" in text
        assert "Score-only assessment" not in text
        assert "Score-only/degraded risk assessment" not in text
        assert "Weather Window" in text or "weather window" in text

    def test_a_degraded_report_says_score_only_on_top_and_names_the_reason_below(self):
        lines = self._rendered(
            {"weather_rule_based": False, "degraded_reason": "weather_unavailable"},
            score=0.62,
        )
        text = " ".join(lines)

        assert f"Details {SCORE_ONLY_SENTENCE}" in lines
        heading_at = lines.index("Assessment details")
        assert lines.index("Session image") < lines.index("Inference Details") < heading_at
        assert FULL_DEGRADED_PARAGRAPH in " ".join(lines[heading_at + 1:])
        for fact in ("temperature at least", "held for", "21.3", "71.0", "Weather rules for"):
            assert fact not in text


class TestPlainWordConditions:
    """The PDM's condition text rendered in plain words."""

    @pytest.mark.parametrize(
        "conditions, plain",
        [
            ("temperature_2m >= 20", "temperature at least 20 \u00b0C"),
            ("relative_humidity_2m >= 70", "relative humidity at least 70 %"),
            ("temperature_2m <= 5", "temperature at most 5 \u00b0C"),
            ("relative_humidity_2m > 90", "relative humidity above 90 %"),
            ("temperature_2m < 0", "temperature below 0 \u00b0C"),
            ("temperature_2m >= 20.0", "temperature at least 20 \u00b0C"),
            ("temperature_2m >= 17.5", "temperature at least 17.5 \u00b0C"),
            ("precipitation > 2", "precipitation above 2 mm"),
            (
                "temperature_2m >= 20  AND  relative_humidity_2m >= 70",
                "temperature at least 20 \u00b0C and relative humidity at least 70 %",
            ),
        ],
    )
    def test_conditions_read_in_plain_words(self, conditions, plain):
        from report_content import plain_rule_conditions

        assert plain_rule_conditions(conditions) == plain

    def test_a_variable_without_a_mapping_keeps_its_name_spaced_and_gets_no_unit(self):
        from report_content import plain_rule_conditions

        assert plain_rule_conditions("wind_speed_10m > 10") == "wind speed 10m above 10"

    def test_a_part_that_is_not_a_condition_is_kept_not_dropped(self):
        from report_content import plain_rule_conditions

        assert plain_rule_conditions("something else entirely") == "something else entirely"


class TestPestResolvesByThePair:
    """A session that carries a pest
    resolves its catalogue record by (crop, EPPO code); without one, by
    crop alone as before; an unknown pair renders unavailable by name."""

    @staticmethod
    def _spies(monkeypatch):
        import report_content

        calls = {"pair": [], "crop": []}
        real_pair, real_crop = report_content.get_threat_record_by_eppo, report_content.get_threat_record_for_crop

        def _pair(crop, eppo_code):
            calls["pair"].append((crop, eppo_code))
            return real_pair(crop, eppo_code)

        def _crop(crop):
            calls["crop"].append(crop)
            return real_crop(crop)

        monkeypatch.setattr(report_content, "get_threat_record_by_eppo", _pair)
        monkeypatch.setattr(report_content, "get_threat_record_for_crop", _crop)
        return calls

    def _payload(self, crop="brassica-oleracea", pest=None):
        return compose_report_payload(
            risk_input=_risk_input(crop=crop),
            envelope=_envelope(designation_status="designated", pest=pest),
            pdm_assessment=_pdm_row(fused_verdict="High"),
            fuzzy_assessment=_fuzzy_row(fused_verdict="Moderate"),
            contract_version="1.5.0",
        )

    def _rendered_text(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "session.pdf")
            render_pest_session_pdf(payload, lambda variant: None, output_path)
            reader = pypdf.PdfReader(output_path)
            return " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)

    def test_a_session_with_a_pest_resolves_by_the_pair(self, monkeypatch):
        calls = self._spies(monkeypatch)
        payload = self._payload(pest="BRVCBR")

        assert calls["pair"] == [("brassica-oleracea", "BRVCBR")]
        assert calls["crop"] == []
        context = payload["pest_crop_context"]
        assert context["pest_scientific_name"] == "Brevicoryne brassicae"
        assert context["pest_eppo_code"] == "BRVCBR"
        assert payload["pest_crop_context_unavailable_reason"] is None
        text = self._rendered_text(payload)
        assert "Brevicoryne brassicae" in text and "BRVCBR" in text

    def test_a_session_without_a_pest_resolves_by_crop_as_before(self, monkeypatch):
        calls = self._spies(monkeypatch)
        payload = self._payload(pest=None)

        assert calls["pair"] == []
        assert calls["crop"] == ["brassica-oleracea"]
        assert payload["pest_crop_context"]["pest_eppo_code"] == "BRVCBR"
        assert payload["pest_crop_context_unavailable_reason"] is None

    def test_cucumber_context_uses_the_display_name_and_selected_pest(self):
        default = self._payload(crop="cucumis-sativus")
        aphigo = self._payload(crop="cucumis-sativus", pest="APHIGO")
        for payload, pest in ((default, "MYZUPE"), (aphigo, "APHIGO")):
            context = payload["pest_crop_context"]
            assert context["pest_eppo_code"] == pest
            assert context["crop_name"] == "Cucumber (Cucumis sativus)"
            assert "cucumis-sativus" not in self._rendered_text(payload)

    def test_an_unknown_pair_renders_unavailable_by_name(self, monkeypatch):
        calls = self._spies(monkeypatch)
        payload = self._payload(pest="MYZUPE")

        assert calls["pair"] == [("brassica-oleracea", "MYZUPE")]
        assert payload["pest_crop_context"] is None
        reason = payload["pest_crop_context_unavailable_reason"]
        assert "MYZUPE" in reason
        assert "Cabbage / Brassica crops (Brassica oleracea)" in reason
        assert "Brevicoryne" not in json.dumps(payload)
        text = self._rendered_text(payload)
        assert reason in text
        assert "Brevicoryne" not in text and "BRVCBR" not in text

    def test_a_known_pest_on_the_wrong_crop_is_an_unknown_pair(self):
        payload = self._payload(crop="cucumis-sativus", pest="BRVCBR")
        assert payload["pest_crop_context"] is None
        assert "BRVCBR" in payload["pest_crop_context_unavailable_reason"]

    def test_the_pair_lookup_is_case_insensitive_on_the_crop_and_the_code(self):
        from fuzzy_threats import get_threat_record_by_eppo

        assert get_threat_record_by_eppo("Brassica-Oleracea", "brvcbr").eppo_code == "BRVCBR"

    def test_a_kale_session_with_a_pest_resolves_the_kale_record(self):
        payload = self._payload(crop="brassica-oleracea-sabellica", pest="BRVCBR")
        assert payload["pest_crop_context"]["crop_scientific"] == "brassica-oleracea-sabellica"

    def test_a_kale_session_carrying_the_legacy_code_renders_the_verified_code(self):
        """Sessions that carry the legacy code BREBRA are normalized to BRVCBR in the report."""
        payload = self._payload(crop="brassica-oleracea-sabellica", pest="BREBRA")
        context = payload["pest_crop_context"]

        assert context["pest_eppo_code"] == "BRVCBR"
        assert context["pest_eppo_url"].endswith("/BRVCBR")
        text = self._rendered_text(payload)
        assert "BRVCBR" in text
        assert "BREBRA" not in text

    def test_no_context_and_no_reason_in_stock_mode(self):
        payload = compose_report_payload(
            risk_input=_risk_input(),
            envelope=_envelope(designation_status="stub", pest="MYZUPE"),
            pdm_assessment=_pdm_row(),
            fuzzy_assessment=_fuzzy_row(),
            contract_version="1.5.0",
        )
        assert payload["pest_crop_context"] is None
        assert payload["pest_crop_context_unavailable_reason"] is None

    def test_the_default_identity_call_is_still_the_crop_only_signature(self):
        from report_content import resolve_pest_identity

        assert resolve_pest_identity("brassica-oleracea").eppo_code == "BRVCBR"
        assert resolve_pest_identity("unknown-crop") is None
        assert resolve_pest_identity("brassica-oleracea", "MYZUPE") is None
