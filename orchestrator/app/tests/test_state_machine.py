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
Restart and retry behaviour: a session restarted mid-workflow resumes or fails explicitly (no lost or
double-completed sessions); bounded retries end in an explicit failure.

Both the PDM path and the fuzzy path are wired, with
fusion applied to both — a successfully-processed session
reaches "assessed" once BOTH paths have concluded, then continues on its own through "reporting" into the terminal
"reported" state — see service.py's module docstring.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import gatekeeper
from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fuzzy_client import FuzzyAssessmentError, FuzzyAssessmentResult, WeatherWindowSummary
from models import AttemptLog, RiskAssessmentRecord, SessionRecord, SessionState
from pdm_client import PdmAssessmentResult, PdmConnectionError, assess_via_pdm
from reporting_client import ReportPushResult
from service import (
    advance_session,
    intake_submission,
    process_session_step,
)


def _submission(session_id: str = "sess-1") -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=0.5,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    envelope = ReportingEnvelope(
        session_id=session_id,
        best_image_id="img-1",
        image_count=3,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
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


class _FakeSettings:
    max_outbound_retries = 3


class _FakeSettingsWithGatekeeper(_FakeSettings):
    """Adds the fields `assess_via_pdm`/`get_gatekeeper_token` need, so the
    gatekeeper-auth-failure test below drives the real call path (not a
    stand-in that raises the exception directly) — see
    TestGatekeeperAuthFailureEntersBoundedRetryPath."""

    gatekeeper_base_url = "http://gatekeeper:8000"
    gatekeeper_username = "orchestrator"
    gatekeeper_password = "orchestrator-secret"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 10
    pdm_service_name = "pdm"
    pdm_risk_assessment_path = "/api/v1/risk-assessment/"
    pdm_call_timeout_seconds = 15


def _always_fails(settings, risk_input):
    raise PdmConnectionError("PDM unreachable (test)")


def _always_succeeds(settings, risk_input):
    return PdmAssessmentResult(
        level="High",
        details="test",
        provenance={
            "weather_rule_based": True,
            "weather_rule_level_raw": "high",
            "ruleset_version": "1.0.0",
            "pdm_app_version": "0.2.0",
        },
    )


def _fuzzy_always_succeeds(settings, risk_input):
    return FuzzyAssessmentResult(
        dominant_band="moderate",
        daily_scores=[{"date": "2026-01-01", "value": 40.0, "risk_level": "moderate"}],
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


def _fuzzy_always_fails(settings, risk_input):
    raise FuzzyAssessmentError("fuzzy engine unreachable (test)")


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-fake-1")


class TestRestartMidWorkflowResumes:
    def test_session_stuck_in_assessing_resumes_after_restart(self, db, new_db_connection):
        intake_submission(db, _submission())
        session = db.query(SessionRecord).filter_by(session_id="sess-1").one()

        process_session_step(db, session, settings=_FakeSettings())
        assert session.state == SessionState.ASSESSING.value

        restarted_db = new_db_connection()
        resumed_session = restarted_db.query(SessionRecord).filter_by(session_id="sess-1").one()
        assert resumed_session.state == SessionState.ASSESSING.value

        result = advance_session(
            restarted_db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_succeeds,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.park_reason is None
        assert restarted_db.query(RiskAssessmentRecord).filter_by(session_id="sess-1").count() == 2

    def test_already_assessed_session_is_not_reprocessed_on_resume(self, db, new_db_connection):
        intake_submission(db, _submission())
        advance_session(
            db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_succeeds,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert db.query(SessionRecord).filter_by(session_id="sess-1").one().state == SessionState.REPORTED.value
        assert db.query(RiskAssessmentRecord).filter_by(session_id="sess-1").count() == 2

        restarted_db = new_db_connection()
        result = advance_session(
            restarted_db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_succeeds,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert restarted_db.query(RiskAssessmentRecord).filter_by(session_id="sess-1").count() == 2

    def test_restart_mid_retry_budget_resumes_attempt_count_not_reset(self, db, new_db_connection):
        """Covers the transition the other two restart tests don't: a crash
        between individual retry attempts (not before the first, not after
        the retry budget is already exhausted) — the resumed session must
        continue the attempt count, not restart it from zero or lose the
        prior failed attempt's AttemptLog row. Filters to call_type
        "pdm_assess" since the fuzzy path now also logs its own attempt
        once PDM concludes — this test is specifically about PDM's
        bookkeeping."""
        intake_submission(db, _submission())
        session = db.query(SessionRecord).filter_by(session_id="sess-1").one()

        process_session_step(db, session, settings=_FakeSettings())
        process_session_step(db, session, settings=_FakeSettings(), pdm_call=_always_fails)
        assert session.state == SessionState.ASSESSING.value
        assert session.pdm_attempt_count == 1

        restarted_db = new_db_connection()
        resumed_session = restarted_db.query(SessionRecord).filter_by(session_id="sess-1").one()
        assert resumed_session.pdm_attempt_count == 1
        assert restarted_db.query(AttemptLog).filter_by(session_id="sess-1", call_type="pdm_assess").count() == 1

        result = advance_session(
            restarted_db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_succeeds,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.pdm_attempt_count == 2
        attempts = (
            restarted_db.query(AttemptLog)
            .filter_by(session_id="sess-1", call_type="pdm_assess")
            .order_by(AttemptLog.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in attempts] == [1, 2]
        assert [a.status for a in attempts] == ["failed", "succeeded"]


class TestBoundedPdmRetriesThenPathRecordedFailed:
    """PDM's own bounded-retry bookkeeping, landing on the PDM
    path's own RiskAssessmentRecord rather than on the session."""

    def test_persistent_pdm_failure_records_path_failed_session_still_reports(self, db):
        intake_submission(db, _submission())
        settings = _FakeSettings()

        result = advance_session(
            db,
            "sess-1",
            settings=settings,
            pdm_call=_always_fails,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.pdm_attempt_count == settings.max_outbound_retries

        pdm_attempts = db.query(AttemptLog).filter_by(session_id="sess-1", call_type="pdm_assess").all()
        assert len(pdm_attempts) == settings.max_outbound_retries
        assert all(a.status == "failed" for a in pdm_attempts)

        failed_assessment = db.query(RiskAssessmentRecord).filter_by(session_id="sess-1", path="pdm").one()
        assert failed_assessment.status == "failed"
        assert failed_assessment.fused_verdict is None

        fuzzy_assessment = db.query(RiskAssessmentRecord).filter_by(session_id="sess-1", path="fuzzy").one()
        assert fuzzy_assessment.status == "present"
        assert fuzzy_assessment.fused_verdict is not None

    def test_failed_pdm_path_carries_attempt_history_visible_by_query(self, db):
        intake_submission(db, _submission())
        advance_session(
            db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_fails,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        pdm_attempts = (
            db.query(AttemptLog)
            .filter_by(session_id="sess-1", call_type="pdm_assess")
            .order_by(AttemptLog.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in pdm_attempts] == [1, 2, 3]

    def test_no_session_disappears_none_double_completed(self, db):
        """Sanity check across outcomes: whatever advance_session does,
        exactly one session_record row exists throughout, never zero —
        even when BOTH paths fail."""
        intake_submission(db, _submission())
        assert db.query(SessionRecord).count() == 1

        advance_session(
            db,
            "sess-1",
            settings=_FakeSettings(),
            pdm_call=_always_fails,
            fuzzy_call=_fuzzy_always_fails,
            report_call=_report_succeeds,
        )
        assert db.query(SessionRecord).count() == 1


class TestGatekeeperAuthFailureEntersBoundedRetryPath:
    """A GateKeeper login failure while the
    session is ASSESSING is a distinct exception type (GatekeeperAuthError)
    from PdmCallError, raised before assess_via_pdm ever reaches PDM. Before
    the fix, process_session_step only caught PdmCallError, so this
    exception propagated uncaught and the session was left parked in
    ASSESSING with pdm_attempt_count still 0 and no AttemptLog row — never
    recorded, never retried, never failed. Drives the REAL assess_via_pdm/
    get_gatekeeper_token call path (only the HTTP boundary mocked), not a
    stand-in pdm_call, so this reproduces the case exactly:
    GateKeeper itself rejecting the login, not PDM. The fuzzy path is
    stubbed (this case is about the PDM path only)."""

    def test_gatekeeper_auth_failure_counts_attempts_and_still_reports(self, db, monkeypatch):
        def _fake_post(url, headers=None, json=None, timeout=None):
            return MagicMock(status_code=401, text="invalid credentials")

        monkeypatch.setattr(gatekeeper.requests, "post", _fake_post)

        intake_submission(db, _submission())
        settings = _FakeSettingsWithGatekeeper()

        result = advance_session(
            db,
            "sess-1",
            settings=settings,
            pdm_call=assess_via_pdm,
            fuzzy_call=_fuzzy_always_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.pdm_attempt_count == settings.max_outbound_retries

        attempts = db.query(AttemptLog).filter_by(session_id="sess-1", call_type="pdm_assess").all()
        assert len(attempts) == settings.max_outbound_retries
        assert all(a.status == "failed" for a in attempts)
        assert all("GateKeeper" in (a.error or "") for a in attempts)

        failed_assessment = db.query(RiskAssessmentRecord).filter_by(session_id="sess-1", path="pdm").one()
        assert failed_assessment.status == "failed"
