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
Failure isolation between the two risk paths.

When one risk engine fails or times out for a session, the other engine's assessment and the
report still complete, and the failed assessment is recorded as failed/absent with the reason. Forced-failure tests each way, plus the
both-fail case (the session record itself must survive even that).
"""
from __future__ import annotations

from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fusion import FUSION_RULE_VERSION
from fuzzy_client import FuzzyAssessmentError, FuzzyAssessmentResult, WeatherWindowSummary
from models import RiskAssessmentRecord, SessionRecord, SessionState
from pdm_client import PdmAssessmentResult, PdmConnectionError
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-fake-1")


class _FakeSettings:
    max_outbound_retries = 3


def _submission(session_id: str = "sess-iso") -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=0.8,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    envelope = ReportingEnvelope(
        session_id=session_id,
        best_image_id="img-1",
        image_count=1,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status="stub",
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
        ),
    )
    return SessionSubmission(
        contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope
    )


def _pdm_succeeds(settings, risk_input):
    return PdmAssessmentResult(
        level="High",
        details="Weather-rule-based risk assessment.",
        provenance={
            "weather_rule_based": True,
            "weather_rule_level_raw": "moderate",
            "ruleset_version": "1.0.0",
            "pdm_app_version": "0.2.0",
        },
    )


def _pdm_fails(settings, risk_input):
    raise PdmConnectionError("PDM down (forced failure test)")


def _fuzzy_succeeds(settings, risk_input):
    return FuzzyAssessmentResult(
        dominant_band="high",
        daily_scores=[{"date": "2026-01-01", "value": 70.0, "risk_level": "high"}],
        engine_version="2.0.0",
        threat_definition_version="1.0.0",
        dossier_version="v2",
        weather_summary=WeatherWindowSummary(
            temperature_min_C=10.0,
            temperature_max_C=20.0,
            relative_humidity_avg_pct=50.0,
            source="openmeteo",
        ),
    )


def _fuzzy_fails(settings, risk_input):
    raise FuzzyAssessmentError("Weather service unreachable (forced failure test)")


class TestPdmFailureLeavesFuzzyAndSessionIntact:
    def test_pdm_forced_down_fuzzy_still_completes(self, db):
        intake_submission(db, _submission())

        result = advance_session(
            db,
            "sess-iso",
            settings=_FakeSettings(),
            pdm_call=_pdm_fails,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        assert db.query(SessionRecord).filter_by(session_id="sess-iso").count() == 1
        assert result.state == SessionState.REPORTED.value

        pdm_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="pdm").one()
        assert pdm_row.status == "failed"
        assert pdm_row.failure_reason is not None
        assert pdm_row.fused_verdict is None

        fuzzy_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="fuzzy").one()
        assert fuzzy_row.status == "present"
        assert fuzzy_row.level == "high"
        assert fuzzy_row.fused_verdict == "High"
        assert fuzzy_row.fusion_rule_version == FUSION_RULE_VERSION


class TestFuzzyFailureLeavesPdmAndSessionIntact:
    def test_fuzzy_forced_down_pdm_still_completes(self, db):
        intake_submission(db, _submission())

        result = advance_session(
            db,
            "sess-iso",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_fails,
            report_call=_report_succeeds,
        )

        assert db.query(SessionRecord).filter_by(session_id="sess-iso").count() == 1
        assert result.state == SessionState.REPORTED.value

        pdm_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="pdm").one()
        assert pdm_row.status == "present"
        assert pdm_row.level == "moderate"
        assert pdm_row.fused_verdict == "High"
        assert pdm_row.fusion_rule_version == FUSION_RULE_VERSION
        assert pdm_row.engine_version == "0.2.0"

        fuzzy_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="fuzzy").one()
        assert fuzzy_row.status == "failed"
        assert fuzzy_row.failure_reason is not None
        assert fuzzy_row.fused_verdict is None


class TestBothEnginesFailingStillParksSessionExplicitly:
    def test_both_forced_down_session_still_reaches_assessed_with_both_absent(self, db):
        intake_submission(db, _submission())

        result = advance_session(
            db,
            "sess-iso",
            settings=_FakeSettings(),
            pdm_call=_pdm_fails,
            fuzzy_call=_fuzzy_fails,
            report_call=_report_succeeds,
        )

        assert db.query(SessionRecord).filter_by(session_id="sess-iso").count() == 1
        assert result.state == SessionState.REPORTED.value

        assert (
            db.query(RiskAssessmentRecord)
            .filter_by(session_id="sess-iso", path="pdm", status="failed")
            .count()
            == 1
        )
        assert (
            db.query(RiskAssessmentRecord)
            .filter_by(session_id="sess-iso", path="fuzzy", status="failed")
            .count()
            == 1
        )


class TestPdmRawBandNotDoubleCountedWithScore:
    """The stored `level` on the PDM path's
    `RiskAssessmentRecord` must equal PDM's RAW weather-only band, never
    PDM's own already-score-combined `level` — the case those two values
    actually diverge is the one that catches a regression back to fusing
    `level` a second time."""

    def test_stored_raw_band_equals_pdms_raw_field_not_its_fused_level(self, db):
        def _pdm_score_dominates(settings, risk_input):
            return PdmAssessmentResult(
                level="High",
                details="Weather-rule-based risk assessment.",
                provenance={
                    "weather_rule_based": True,
                    "weather_rule_level_raw": "low",
                    "ruleset_version": "1.0.0",
                    "pdm_app_version": "0.2.0",
                },
            )

        intake_submission(db, _submission())
        advance_session(
            db,
            "sess-iso",
            settings=_FakeSettings(),
            pdm_call=_pdm_score_dominates,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        pdm_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="pdm").one()
        assert pdm_row.level == "low"
        assert pdm_row.level != "High"
        assert pdm_row.fused_verdict == "High"


class TestPdmDegradedPathHasNoRawBandButStillFuses:
    """PDM's score-only/degraded response (no weather-rule engine ran) has
    no raw weather-side band to report — the orchestrator must not crash or
    fabricate one; the fused verdict falls back to the NN-only band."""

    def test_degraded_pdm_response_stores_no_raw_band_and_self_fuses(self, db):
        def _pdm_degraded(settings, risk_input):
            return PdmAssessmentResult(
                level="High",
                details="Score-only/degraded risk assessment (weather unavailable).",
                provenance={
                    "weather_rule_based": False,
                    "degraded_reason": "weather_unavailable",
                    "weather_rule_level_raw": None,
                    "ruleset_version": "1.0.0",
                    "pdm_app_version": "0.2.0",
                },
            )

        intake_submission(db, _submission())
        advance_session(
            db,
            "sess-iso",
            settings=_FakeSettings(),
            pdm_call=_pdm_degraded,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        pdm_row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-iso", path="pdm").one()
        assert pdm_row.status == "present"
        assert pdm_row.level is None
        assert pdm_row.fused_verdict == "High"
        assert pdm_row.fusion_rule_version == FUSION_RULE_VERSION
