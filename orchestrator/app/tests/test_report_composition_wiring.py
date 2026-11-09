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
Report composition wiring: `service.py`'s ASSESSED -> REPORTING -> REPORTED/FAILED
transitions, the ReportIndexRecord it populates on success, and the bounded
retry budget on the report-push outbound call.
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
from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary
from models import AttemptLog, ReportIndexRecord, RiskAssessmentRecord, SessionRecord, SessionState
from pdm_client import PdmAssessmentResult
from reporting_client import ReportingConnectionError, ReportPushResult
from service import advance_session, intake_submission


class _FakeSettings:
    max_outbound_retries = 3


def _submission(session_id: str = "sess-report") -> SessionSubmission:
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
        image_count=3,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status="designated",
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


def _fuzzy_succeeds(settings, risk_input):
    return FuzzyAssessmentResult(
        dominant_band="moderate",
        daily_scores=[{"date": "2026-01-01", "value": 40.0, "risk_level": "moderate"}],
        engine_version="2.0.0",
        threat_definition_version="1.0.0",
        dossier_version="v2",
        weather_summary=WeatherWindowSummary(
            temperature_min_C=11.0,
            temperature_max_C=22.0,
            relative_humidity_avg_pct=58.0,
            source="openmeteo",
        ),
    )


class TestSuccessfulReportPushReachesReportedWithIndexEntry:
    def test_session_reaches_reported_with_report_index_row(self, db):
        captured_payload = {}

        def _report_succeeds(settings, payload):
            captured_payload.update(payload)
            return ReportPushResult(report_id="report-uuid-1")

        intake_submission(db, _submission())
        result = advance_session(
            db,
            "sess-report",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.park_reason is None
        assert result.report_attempt_count == 1

        index_row = db.query(ReportIndexRecord).filter_by(session_id="sess-report").one()
        assert index_row.report_id == "report-uuid-1"
        assert index_row.origin == "cloud-generated"
        assert index_row.state == "reported"
        assert index_row.content_hash is not None

        assert captured_payload["envelope"]["session_id"] == "sess-report"
        assert captured_payload["risk_assessment"]["level"] == "High"

    def test_persisted_fuzzy_weather_summary_carries_through_to_composition(self, db):
        """Service-seam case: the
        fuzzy handler's PERSISTED RiskAssessmentRecord (round-tripped
        through provenance_json, not the in-memory FuzzyAssessmentResult)
        is what composition reads -- a summary dropped at that seam must
        not silently pass."""
        captured_payload = {}

        def _report_succeeds(settings, payload):
            captured_payload.update(payload)
            return ReportPushResult(report_id="report-uuid-weather-seam")

        intake_submission(db, _submission("sess-weather-seam"))
        advance_session(
            db,
            "sess-weather-seam",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        weather = captured_payload["weather_summary"]
        assert "unavailable_reason" not in weather
        assert weather["source"] == "openmeteo"
        assert weather["observed"]["temperature_2m_min_C"] == 11.0
        assert weather["observed"]["temperature_2m_max_C"] == 22.0
        assert weather["observed"]["relative_humidity_2m_avg_pct"] == 58.0

    def test_second_advance_call_does_not_duplicate_report_index_row(self, db):
        def _report_succeeds(settings, payload):
            return ReportPushResult(report_id="report-uuid-1")

        intake_submission(db, _submission())
        advance_session(
            db,
            "sess-report",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )
        advance_session(
            db,
            "sess-report",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_succeeds,
        )

        assert db.query(ReportIndexRecord).filter_by(session_id="sess-report").count() == 1


class TestBoundedReportRetriesThenSessionFailed:
    def test_persistent_report_push_failure_moves_session_to_failed(self, db):
        def _report_always_fails(settings, payload):
            raise ReportingConnectionError("ReportingService unreachable (test)")

        intake_submission(db, _submission())
        settings = _FakeSettings()
        result = advance_session(
            db,
            "sess-report",
            settings=settings,
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_always_fails,
        )

        assert result.state == SessionState.FAILED.value
        assert result.failure_reason is not None
        assert result.report_attempt_count == settings.max_outbound_retries

        report_attempts = db.query(AttemptLog).filter_by(session_id="sess-report", call_type="report_push").all()
        assert len(report_attempts) == settings.max_outbound_retries
        assert all(a.status == "failed" for a in report_attempts)

        assert db.query(ReportIndexRecord).filter_by(session_id="sess-report").count() == 0
        assert db.query(RiskAssessmentRecord).filter_by(session_id="sess-report").count() == 2

    def test_transient_failure_then_success_still_reports(self, db):
        attempts = {"count": 0}

        def _report_fails_once_then_succeeds(settings, payload):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ReportingConnectionError("transient (test)")
            return ReportPushResult(report_id="report-uuid-2")

        intake_submission(db, _submission())
        result = advance_session(
            db,
            "sess-report",
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_fails_once_then_succeeds,
        )

        assert result.state == SessionState.REPORTED.value
        assert result.report_attempt_count == 2


class TestIntakeResponseCarriesReportId:
    """Intake is synchronous end-to-end, so a caller (the session feeder's push client) gets the report_id back in the
    SAME response, without a separate query call."""

    def test_report_id_present_when_state_is_reported(self, db, monkeypatch):
        import main as main_module

        def _fake_advance_session(db_, session_id, settings=None, **kwargs):
            from service import advance_session as real_advance_session

            return real_advance_session(
                db_,
                session_id,
                settings=settings,
                pdm_call=_pdm_succeeds,
                fuzzy_call=_fuzzy_succeeds,
                report_call=lambda settings, payload: ReportPushResult(report_id="report-uuid-http-1"),
            )

        monkeypatch.setattr(main_module, "advance_session", _fake_advance_session)

        response = main_module._receive_session_submission(_submission("sess-report-http"), db)

        assert response.state == SessionState.REPORTED.value
        assert response.report_id == "report-uuid-http-1"

    def test_report_id_none_when_not_reported(self, db, monkeypatch):
        import main as main_module

        def _fake_advance_session(db_, session_id, settings=None, **kwargs):
            from service import advance_session as real_advance_session

            return real_advance_session(
                db_,
                session_id,
                settings=settings,
                pdm_call=_pdm_succeeds,
                fuzzy_call=_fuzzy_succeeds,
                report_call=lambda settings, payload: (_ for _ in ()).throw(
                    __import__("reporting_client").ReportingConnectionError("down (test)")
                ),
            )

        monkeypatch.setattr(main_module, "advance_session", _fake_advance_session)

        response = main_module._receive_session_submission(_submission("sess-report-http-2"), db)

        assert response.state == SessionState.FAILED.value
        assert response.report_id is None

    def test_report_id_present_on_duplicate_submission_of_already_reported_session(self, db, monkeypatch):
        """A duplicate submission of an already-completed session returns the same report_id a
        fresh success would, not None."""
        import main as main_module

        def _fake_advance_session(db_, session_id, settings=None, **kwargs):
            from service import advance_session as real_advance_session

            return real_advance_session(
                db_,
                session_id,
                settings=settings,
                pdm_call=_pdm_succeeds,
                fuzzy_call=_fuzzy_succeeds,
                report_call=lambda settings, payload: ReportPushResult(report_id="report-uuid-dup-1"),
            )

        monkeypatch.setattr(main_module, "advance_session", _fake_advance_session)

        submission = _submission("sess-report-http-dup")
        first = main_module._receive_session_submission(submission, db)
        assert first.state == SessionState.REPORTED.value
        assert first.report_id == "report-uuid-dup-1"

        second = main_module._receive_session_submission(submission, db)
        assert second.status == "duplicate"
        assert second.state == SessionState.REPORTED.value
        assert second.report_id == "report-uuid-dup-1"


class TestThePestReachesTheFuzzyCall:
    """Service seam: a session whose
    envelope carries a pest hands it to the fuzzy call by keyword; one
    without a pest calls it exactly as before (two arguments)."""

    def _run(self, db, session_id, pest):
        seen = {}

        def _fuzzy(settings, risk_input, **kwargs):
            seen["kwargs"] = kwargs
            return _fuzzy_succeeds(settings, risk_input)

        submission = _submission(session_id)
        if pest is not None:
            submission.reporting_envelope.pest = pest
        intake_submission(db, submission)
        advance_session(
            db,
            session_id,
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy,
            report_call=lambda settings, payload: ReportPushResult(report_id="r"),
        )
        return seen["kwargs"]

    def test_a_pest_is_passed_by_keyword(self, db):
        assert self._run(db, "sess-pest", "BRVCBR") == {"pest": "BRVCBR"}

    def test_no_pest_means_no_extra_argument(self, db):
        assert self._run(db, "sess-nopest", None) == {}


_REASSESSMENT_LINE = (
    "Cloud re-assessment after synchronisation of the on-device report edge-report-uuid-9: "
    "the weather rules, the fuzzy path and the outlook are computed from the recorded weather "
    "of the session window; the on-device report keeps its score-band assessment."
)


class TestEdgeReassessmentIsLabelledThroughThePipeline:
    """An intake marked as
    an edge re-assessment runs the unchanged pipeline (PDM, fuzzy and report
    fakes) and labels what it writes; a plain intake is unchanged."""

    def _intake(self, db, monkeypatch, session_id, origin, edge_report_id):
        import main as main_module

        captured_payload = {}

        def _report_succeeds(settings, payload):
            captured_payload.update(payload)
            return ReportPushResult(report_id="report-uuid-3")

        def _advance_with_fakes(db_, sid, settings=None, **kwargs):
            from service import advance_session as real_advance_session

            return real_advance_session(
                db_,
                sid,
                settings=_FakeSettings(),
                pdm_call=_pdm_succeeds,
                fuzzy_call=_fuzzy_succeeds,
                report_call=_report_succeeds,
                forecast_call=lambda *args, **kwargs: None,
            )

        monkeypatch.setattr(main_module, "advance_session", _advance_with_fakes)
        response = main_module._receive_session_submission(
            _submission(session_id), db, origin=origin, edge_report_id=edge_report_id
        )
        return response, captured_payload

    def test_an_edge_reassessment_labels_the_records_the_index_and_the_provenance(self, db, monkeypatch):
        response, payload = self._intake(db, monkeypatch, "sess-reassess", "edge-sync", "edge-report-uuid-9")

        assert (response.status, response.state, response.report_id) == ("accepted", "reported", "report-uuid-3")
        record = db.query(SessionRecord).filter_by(session_id="sess-reassess").one()
        assert (record.origin, record.edge_report_id) == ("edge-sync", "edge-report-uuid-9")
        modes = {row.path: row.assessment_mode for row in db.query(RiskAssessmentRecord).filter_by(session_id="sess-reassess")}
        assert modes == {"pdm": "retroactive", "fuzzy": "retroactive"}
        index_row = db.query(ReportIndexRecord).filter_by(session_id="sess-reassess").one()
        assert (index_row.origin, index_row.state, index_row.report_id) == ("edge-reassessed", "reported", "report-uuid-3")
        assert payload["risk_reasoning"]["engine_provenance"][-1] == _REASSESSMENT_LINE

    def test_a_plain_submission_stays_live_and_cloud_generated(self, db, monkeypatch):
        response, payload = self._intake(db, monkeypatch, "sess-plain-wired", None, None)

        assert (response.status, response.state) == ("accepted", "reported")
        record = db.query(SessionRecord).filter_by(session_id="sess-plain-wired").one()
        assert (record.origin, record.edge_report_id) == (None, None)
        modes = {row.path: row.assessment_mode for row in db.query(RiskAssessmentRecord).filter_by(session_id="sess-plain-wired")}
        assert modes == {"pdm": "live", "fuzzy": "live"}
        index_row = db.query(ReportIndexRecord).filter_by(session_id="sess-plain-wired").one()
        assert index_row.origin == "cloud-generated"
        assert not any("re-assessment" in line for line in payload["risk_reasoning"]["engine_provenance"])

    def test_a_failed_engine_path_is_labelled_too(self, db, monkeypatch):
        from pdm_client import PdmCallError

        def _pdm_down(settings, risk_input):
            raise PdmCallError("PDM unreachable")

        def _fuzzy_down(settings, risk_input, **kwargs):
            raise RuntimeError("fuzzy unavailable")

        intake_submission(db, _submission("sess-failed-paths"), origin="edge-sync", edge_report_id="edge-report-uuid-9")
        advance_session(
            db,
            "sess-failed-paths",
            settings=_FakeSettings(),
            pdm_call=_pdm_down,
            fuzzy_call=_fuzzy_down,
            report_call=lambda settings, payload: ReportPushResult(report_id="r"),
            forecast_call=lambda *args, **kwargs: None,
        )

        rows = db.query(RiskAssessmentRecord).filter_by(session_id="sess-failed-paths").all()
        assert {row.path: (row.status, row.assessment_mode) for row in rows} == {
            "pdm": ("failed", "retroactive"),
            "fuzzy": ("failed", "retroactive"),
        }
