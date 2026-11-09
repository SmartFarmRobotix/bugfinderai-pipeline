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
A risk result with score-only edge provenance for a sample session: the edge profile's score-band
risk mapping, wired into the orchestrator
state machine with no PDM/fuzzy/Gatekeeper call of any kind.

These tests drive `process_session_step` one call at a time (RECEIVED ->
ASSESSING, then ASSESSING -> ASSESSED) rather than `advance_session`, so
they stop exactly at the score-band assessment and never reach into report
generation. ASSESSED is a transient checkpoint under `advance_session` (mirrors cloud/
mixed), so asserting on it specifically requires the step-by-step driver.
"""

from __future__ import annotations

import json

from config import DeploymentProfile
from contract import CONTRACT_VERSION, DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput, SessionSubmission
from models import AttemptLog, RiskAssessmentRecord, SessionRecord, SessionState
from service import intake_submission, process_session_step


class _EdgeSettings:
    deployment_profile = DeploymentProfile.EDGE
    max_outbound_retries = 3
    fusion_rule_version = "v1-max-of-bands"


def _submission(session_id: str = "edge-sess-1", infestation_score: float = 0.82) -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=infestation_score,
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
        image_count=2,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
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
    return SessionSubmission(contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope)


def _advance_to_assessed(db, session_id: str, settings=None) -> SessionRecord:
    settings = settings or _EdgeSettings()
    session = db.query(SessionRecord).filter_by(session_id=session_id).one()
    session = process_session_step(db, session, settings=settings)
    assert session.state == SessionState.ASSESSING.value
    session = process_session_step(db, session, settings=settings)
    assert session.state == SessionState.ASSESSED.value
    return session


class TestEdgeScoreBandAssessment:
    def test_high_score_yields_high_band_with_score_only_edge_provenance(self, db):
        intake_submission(db, _submission(session_id="edge-sess-high", infestation_score=0.85))

        result = _advance_to_assessed(db, "edge-sess-high")

        assert result.park_reason is not None

        record = db.query(RiskAssessmentRecord).filter_by(
            session_id="edge-sess-high", path="edge-score-band"
        ).one()
        assert record.status == "present"
        assert record.fused_verdict == "High"
        assert record.engine_identity == "edge-score-band"
        assert record.level is None

        provenance = json.loads(record.provenance_json)
        assert provenance["weather_rule_based"] is False
        assert provenance["degraded_reason"] == "weather_not_applicable"
        assert provenance["edge_local"] is True

    def test_low_score_yields_low_band(self, db):
        intake_submission(db, _submission(session_id="edge-sess-low", infestation_score=0.1))

        _advance_to_assessed(db, "edge-sess-low")

        record = db.query(RiskAssessmentRecord).filter_by(
            session_id="edge-sess-low", path="edge-score-band"
        ).one()
        assert record.fused_verdict == "Low"

    def test_moderate_score_yields_medium_band(self, db):
        intake_submission(db, _submission(session_id="edge-sess-mid", infestation_score=0.5))

        _advance_to_assessed(db, "edge-sess-mid")

        record = db.query(RiskAssessmentRecord).filter_by(
            session_id="edge-sess-mid", path="edge-score-band"
        ).one()
        assert record.fused_verdict == "Medium"

    def test_single_attempt_no_retry_bookkeeping_needed(self, db):
        """Unlike the PDM path (bounded retries), the score-band path is a
        pure computation -- one recorded attempt, always succeeded."""
        intake_submission(db, _submission(session_id="edge-sess-attempt"))
        _advance_to_assessed(db, "edge-sess-attempt")

        attempts = db.query(AttemptLog).filter_by(
            session_id="edge-sess-attempt", call_type="edge_score_band_assess"
        ).all()
        assert len(attempts) == 1
        assert attempts[0].status == "succeeded"

    def test_repeated_step_at_assessed_is_idempotent(self, db):
        intake_submission(db, _submission(session_id="edge-sess-idem"))
        session = _advance_to_assessed(db, "edge-sess-idem")

        count = db.query(RiskAssessmentRecord).filter_by(
            session_id="edge-sess-idem", path="edge-score-band"
        ).count()
        assert count == 1
        assert session.state == SessionState.ASSESSED.value

    def test_restart_mid_workflow_resumes_at_assessing(self, db, new_db_connection):
        intake_submission(db, _submission(session_id="edge-sess-restart"))
        session = db.query(SessionRecord).filter_by(session_id="edge-sess-restart").one()

        process_session_step(db, session, settings=_EdgeSettings())
        assert session.state == SessionState.ASSESSING.value

        restarted_db = new_db_connection()
        resumed_session = restarted_db.query(SessionRecord).filter_by(session_id="edge-sess-restart").one()
        result = process_session_step(restarted_db, resumed_session, settings=_EdgeSettings())

        assert result.state == SessionState.ASSESSED.value
        assert restarted_db.query(RiskAssessmentRecord).filter_by(
            session_id="edge-sess-restart", path="edge-score-band"
        ).count() == 1
