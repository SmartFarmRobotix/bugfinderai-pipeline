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
Evaluation-shaped queries return complete data
without DB access — every assertion below goes through `queries.py`'s
functions only, never a raw `db.query(...)` of its own.
"""
from __future__ import annotations

import datetime

import pytest

from contract import (
    CONTRACT_VERSION,
    ConversionChainProvenance,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary
from models import CutoverBoundary
from pdm_client import PdmAssessmentResult
from queries import SessionNotFound, get_paired_assessments, get_report_state, get_session_detail, list_sessions
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


class _FakeSettings:
    max_outbound_retries = 3


def _submission(session_id: str, execution_target: str = "cloud") -> SessionSubmission:
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
    conversion_chain = (
        ConversionChainProvenance(
            execution_target=execution_target,
            pt_version="1.0",
            pt_source="local",
            pt_license="MIT",
            onnx_artifact="model.onnx",
            onnx_sha256="a" * 64,
            engine_artifact="model.engine",
            engine_sha256="b" * 64,
        )
        if execution_target in ("cloud", "jetson")
        else None
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
            conversion_chain=conversion_chain,
        ),
    )
    return SessionSubmission(
        contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope
    )


def _pdm_succeeds(settings, risk_input):
    return PdmAssessmentResult(
        level="High",
        details="ok",
        provenance={
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
            temperature_min_C=10.0,
            temperature_max_C=20.0,
            relative_humidity_avg_pct=50.0,
            source="openmeteo",
        ),
    )


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-uuid-q1")


def _process(db, session_id: str, execution_target: str = "cloud"):
    intake_submission(db, _submission(session_id, execution_target=execution_target))
    return advance_session(
        db,
        session_id,
        settings=_FakeSettings(),
        pdm_call=_pdm_succeeds,
        fuzzy_call=_fuzzy_succeeds,
        report_call=_report_succeeds,
    )


class TestGetSessionDetail:
    def test_complete_detail_for_a_reported_session(self, db):
        _process(db, "sess-q1", execution_target="cloud")

        detail = get_session_detail(db, "sess-q1")

        assert detail.session_id == "sess-q1"
        assert detail.state == "reported"
        assert detail.scenario == "cloud"
        assert set(detail.assessments.keys()) == {"pdm", "fuzzy"}
        assert detail.assessments["pdm"].fused_verdict == "High"
        assert detail.assessments["fuzzy"].fused_verdict is not None
        assert detail.report is not None
        assert detail.report.report_id == "report-uuid-q1"
        assert detail.report.state == "reported"

    def test_scenario_derives_mixed_from_jetson_execution_target(self, db):
        _process(db, "sess-q2", execution_target="jetson")
        detail = get_session_detail(db, "sess-q2")
        assert detail.scenario == "mixed"

    def test_scenario_unknown_when_no_conversion_chain(self, db):
        _process(db, "sess-q3", execution_target="none")
        detail = get_session_detail(db, "sess-q3")
        assert detail.scenario == "unknown"

    def test_unknown_session_raises_session_not_found(self, db):
        with pytest.raises(SessionNotFound):
            get_session_detail(db, "does-not-exist")

    def test_scenario_is_edge_for_an_edge_session_not_execution_target(self, db, tmp_path):
        """An edge
        session's scenario must read "edge", not fall through to the
        cloud/mixed conversion-chain heuristic -- mixed also executes on
        Jetson hardware (execution_target="jetson"), so execution_target
        alone cannot distinguish edge from mixed; the edge-score-band
        assessment record (only ever written under DeploymentProfile.EDGE)
        is what makes this authoritative rather than re-derived."""
        from config import DeploymentProfile

        class _EdgeSettings:
            deployment_profile = DeploymentProfile.EDGE
            max_outbound_retries = 3
            fusion_rule_version = "v1-max-of-bands"
            edge_inference_call_timeout_seconds = 5
            edge_inference_base_url = "http://127.0.0.1:1"
            edge_inference_api_key = "test-key"

            def __init__(self, report_dir):
                self.edge_report_dir = report_dir
                self.sync_queue_dir = report_dir + "-sync-queue"

        session_id = "sess-edge-scenario"
        intake_submission(db, _submission(session_id, execution_target="jetson"))
        advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        detail = get_session_detail(db, session_id)
        assert detail.scenario == "edge"


class TestSessionDetailImagesPending:
    """`status_call`
    is the injection point `get_session_detail` calls only for a
    cloud-generated report with an id -- these cells stub it directly,
    the HTTP boundary (`fetch_report_status` itself) is covered in
    test_query_endpoints_http.py."""

    def test_status_call_true_then_false_with_same_report_id(self, db):
        _process(db, "sess-pending-1", execution_target="cloud")

        pending = get_session_detail(db, "sess-pending-1", status_call=lambda settings, session_id: True)
        assert pending.report.images_pending is True
        assert pending.report.report_id == "report-uuid-q1"

        complete = get_session_detail(db, "sess-pending-1", status_call=lambda settings, session_id: False)
        assert complete.report.images_pending is False
        assert complete.report.report_id == "report-uuid-q1"

    def test_status_call_raising_leaves_detail_intact_with_field_none(self, db):
        _process(db, "sess-pending-2", execution_target="cloud")

        def _raises(settings, session_id):
            raise RuntimeError("status route boom")

        detail = get_session_detail(db, "sess-pending-2", status_call=_raises)
        assert detail.report.images_pending is None
        assert detail.report.report_id == "report-uuid-q1"
        assert detail.state == "reported"
        assert set(detail.assessments.keys()) == {"pdm", "fuzzy"}

    def test_status_call_returning_none_leaves_field_none(self, db):
        _process(db, "sess-pending-3", execution_target="cloud")

        detail = get_session_detail(db, "sess-pending-3", status_call=lambda settings, session_id: None)
        assert detail.report.images_pending is None
        assert detail.report.report_id == "report-uuid-q1"
        assert detail.state == "reported"
        assert set(detail.assessments.keys()) == {"pdm", "fuzzy"}

    def test_edge_origin_report_is_not_probed(self, db, tmp_path):
        from config import DeploymentProfile

        class _EdgeSettings:
            deployment_profile = DeploymentProfile.EDGE
            max_outbound_retries = 3
            fusion_rule_version = "v1-max-of-bands"
            edge_inference_call_timeout_seconds = 5
            edge_inference_base_url = "http://127.0.0.1:1"
            edge_inference_api_key = "test-key"

            def __init__(self, report_dir):
                self.edge_report_dir = report_dir
                self.sync_queue_dir = report_dir + "-sync-queue"

        session_id = "sess-pending-edge"
        intake_submission(db, _submission(session_id, execution_target="jetson"))
        advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        called = []

        def _stub(settings, sid):
            called.append(sid)
            return True

        detail = get_session_detail(db, session_id, status_call=_stub)
        assert detail.report.origin == "edge-local"
        assert detail.report.images_pending is None
        assert called == []


class TestListSessionsByRangeAndScenario:
    def test_scenario_filter(self, db):
        _process(db, "sess-cloud-1", execution_target="cloud")
        _process(db, "sess-mixed-1", execution_target="jetson")

        cloud_only = list_sessions(db, scenario="cloud")
        assert {d.session_id for d in cloud_only} == {"sess-cloud-1"}

        mixed_only = list_sessions(db, scenario="mixed")
        assert {d.session_id for d in mixed_only} == {"sess-mixed-1"}

    def test_range_filter_excludes_out_of_window_sessions(self, db):
        _process(db, "sess-range-1")

        future_start = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        assert list_sessions(db, start=future_start) == []

        past_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
        assert any(d.session_id == "sess-range-1" for d in list_sessions(db, start=past_start))

    def test_pre_cutover_sessions_excluded_by_default(self, db):
        _process(db, "sess-legacy")
        boundary_row = db.query(CutoverBoundary).one()
        boundary_row.boundary_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        db.add(boundary_row)
        db.commit()

        assert list_sessions(db) == []
        assert any(d.session_id == "sess-legacy" for d in list_sessions(db, exclude_pre_cutover=False))


class TestPairedAssessments:
    def test_both_paths_present_keyed_by_path(self, db):
        _process(db, "sess-pair-1")
        paired = get_paired_assessments(db, "sess-pair-1")
        assert set(paired.keys()) == {"pdm", "fuzzy"}
        assert paired["pdm"].level == "moderate"

    def test_unknown_session_raises(self, db):
        with pytest.raises(SessionNotFound):
            get_paired_assessments(db, "does-not-exist")


class TestReportState:
    def test_report_state_none_before_reporting(self, db):
        intake_submission(db, _submission("sess-pending"))
        assert get_report_state(db, "sess-pending") is None

    def test_report_state_present_after_reporting(self, db):
        _process(db, "sess-reported-1")
        report = get_report_state(db, "sess-reported-1")
        assert report is not None
        assert report.report_id == "report-uuid-q1"

    def test_unknown_session_raises(self, db):
        with pytest.raises(SessionNotFound):
            get_report_state(db, "does-not-exist")
