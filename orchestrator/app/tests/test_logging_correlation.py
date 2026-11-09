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
A session's full history can be rebuilt from its logs by session_id: intake, every state transition, each outbound call, and the
terminal state are all present as structured entries carrying session_id
and the propagated request/correlation id.
"""
from __future__ import annotations

import contextvars
import logging

import pytest

from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary
from logging_utils import CorrelationLogFilter, current_request_id, request_id_var, session_id_var
from pdm_client import PdmAssessmentResult
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


class _FakeSettings:
    max_outbound_retries = 3


def _submission(session_id: str) -> SessionSubmission:
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
        details="ok",
        provenance={"weather_rule_level_raw": "moderate", "pdm_app_version": "0.2.0"},
    )


def _fuzzy_succeeds(settings, risk_input):
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


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-uuid-log-1")


class TestCorrelationLogFilterInjectsContextvars:
    def test_filter_reads_current_contextvar_values(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", None, None)
        token_req = request_id_var.set("corr-abc-123")
        token_sess = session_id_var.set("sess-filter-1")
        try:
            assert CorrelationLogFilter().filter(record) is True
            assert record.request_id == "corr-abc-123"
            assert record.session_id == "sess-filter-1"
        finally:
            request_id_var.reset(token_req)
            session_id_var.reset(token_sess)

    def test_default_values_when_nothing_set(self):
        def _check() -> None:
            record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", None, None)
            assert CorrelationLogFilter().filter(record) is True
            assert record.request_id == "-"
            assert record.session_id == "-"

        contextvars.Context().run(_check)


class TestCurrentRequestIdOmitsUnsetSentinel:
    def test_none_when_unset(self):
        assert current_request_id() is None

    def test_returns_the_set_value(self):
        token = request_id_var.set("corr-xyz")
        try:
            assert current_request_id() == "corr-xyz"
        finally:
            request_id_var.reset(token)


class TestSessionHistoryReconstructableFromLogs:
    """Attaches CorrelationLogFilter to caplog's own handler (a legitimate
    per-test wiring, not production code) so every captured record is
    tagged exactly like it would be against the real root handler
    `logging_utils.configure_logging` wires at process startup — then
    proves the WHOLE run (intake through the terminal state) is one
    session_id-tagged, reconstructable timeline."""

    @pytest.fixture(autouse=True)
    def _wire_filter(self, db, caplog):
        logging.getLogger("orchestrator").disabled = False
        logging.getLogger("orchestrator.service").disabled = False
        caplog.handler.addFilter(CorrelationLogFilter())
        caplog.set_level(logging.INFO, logger="orchestrator")

    def test_full_run_every_record_carries_session_id_and_request_id(self, db, caplog):
        session_id = "sess-log-1"
        request_token = request_id_var.set("corr-log-run-1")
        try:
            intake_submission(db, _submission(session_id))
            result = advance_session(
                db,
                session_id,
                settings=_FakeSettings(),
                pdm_call=_pdm_succeeds,
                fuzzy_call=_fuzzy_succeeds,
                report_call=_report_succeeds,
            )
        finally:
            request_id_var.reset(request_token)

        assert result.state == "reported"

        orchestrator_records = [r for r in caplog.records if r.name.startswith("orchestrator")]
        assert orchestrator_records, "expected at least one orchestrator.* log record"

        for record in orchestrator_records:
            assert record.session_id == session_id
            assert record.request_id == "corr-log-run-1"

        messages = " | ".join(r.getMessage() for r in orchestrator_records)
        assert "intake" in messages
        assert "received -> assessing" in messages
        assert "assessing -> assessed" in messages
        assert "assessed -> reporting" in messages
        assert "reporting -> reported" in messages
        assert "pdm_assess" in messages
        assert "fuzzy_assess" in messages
        assert "report_push" in messages

    def test_bounded_report_failure_history_ends_in_failed(self, db, caplog):
        def _report_always_fails(settings, payload):
            from reporting_client import ReportingConnectionError

            raise ReportingConnectionError("unreachable (test)")

        session_id = "sess-log-2"
        intake_submission(db, _submission(session_id))
        result = advance_session(
            db,
            session_id,
            settings=_FakeSettings(),
            pdm_call=_pdm_succeeds,
            fuzzy_call=_fuzzy_succeeds,
            report_call=_report_always_fails,
        )

        assert result.state == "failed"
        messages = " | ".join(r.getMessage() for r in caplog.records if r.name.startswith("orchestrator"))
        assert "reporting -> failed" in messages
