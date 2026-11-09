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

"""Duplicate submission: the second delivery is acknowledged, one record, no reprocessing."""
from __future__ import annotations

import pytest

from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from models import IdempotencyKey, SessionRecord, SessionState
from service import (
    ContractVersionRejected,
    IdempotencyConflict,
    content_hash_for,
    intake_submission,
)


def _submission(session_id: str = "sess-1", infestation_score: float = 0.5) -> SessionSubmission:
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


class TestIntakeCreatesRecordOnce:
    def test_first_submission_creates_session_and_key(self, db):
        outcome = intake_submission(db, _submission())

        assert outcome.duplicate is False
        assert outcome.state == SessionState.RECEIVED.value
        assert db.query(SessionRecord).count() == 1
        assert db.query(IdempotencyKey).count() == 1

    def test_idempotency_key_persisted_before_processing(self, db):
        """The key exists the instant intake_submission returns — before any
        state-machine advancement is ever attempted."""
        outcome = intake_submission(db, _submission())
        key = db.query(IdempotencyKey).filter_by(session_id=outcome.session_id).one()
        record = db.query(SessionRecord).filter_by(session_id=outcome.session_id).one()
        assert key.content_hash == content_hash_for(_submission())
        assert record.state == SessionState.RECEIVED.value


class TestDuplicateSubmission:
    def test_same_submission_twice_is_acknowledged_without_reprocessing(self, db):
        submission = _submission()

        first = intake_submission(db, submission)
        second = intake_submission(db, submission)

        assert second.duplicate is True
        assert second.session_id == first.session_id
        assert db.query(SessionRecord).count() == 1
        assert db.query(IdempotencyKey).count() == 1

    def test_duplicate_response_identifies_existing_state(self, db):
        submission = _submission()
        intake_submission(db, submission)

        record = db.query(SessionRecord).filter_by(session_id="sess-1").one()
        record.state = SessionState.ASSESSED.value
        db.commit()

        second = intake_submission(db, submission)
        assert second.state == SessionState.ASSESSED.value

    def test_same_session_id_different_content_is_a_conflict(self, db):
        intake_submission(db, _submission(infestation_score=0.2))
        with pytest.raises(IdempotencyConflict):
            intake_submission(db, _submission(infestation_score=0.9))


class TestContractVersionGate:
    def test_unsupported_major_version_rejected(self, db):
        submission = _submission()
        submission = submission.model_copy(update={"contract_version": "99.0.0"})
        with pytest.raises(ContractVersionRejected):
            intake_submission(db, submission)
        assert db.query(SessionRecord).count() == 0


class TestOriginAndEdgeReportIdAreStored:
    """The intake records
    the re-assessment marker on the new session record and keeps it when the
    same submission arrives again."""

    def test_a_plain_submission_stores_nulls(self, db):
        intake_submission(db, _submission("sess-plain"))

        record = db.query(SessionRecord).filter_by(session_id="sess-plain").one()
        assert (record.origin, record.edge_report_id) == (None, None)

    def test_the_parameters_are_stored_on_the_new_record(self, db):
        intake_submission(db, _submission("sess-edge"), origin="edge-sync", edge_report_id="edge-uuid-1")

        record = db.query(SessionRecord).filter_by(session_id="sess-edge").one()
        assert (record.origin, record.edge_report_id) == ("edge-sync", "edge-uuid-1")

    def test_a_duplicate_keeps_the_stored_values(self, db):
        intake_submission(db, _submission("sess-dup"), origin="edge-sync", edge_report_id="edge-uuid-1")

        outcome = intake_submission(db, _submission("sess-dup"), origin="edge-sync", edge_report_id="edge-uuid-2")
        plain_again = intake_submission(db, _submission("sess-dup"))

        assert outcome.duplicate is True and plain_again.duplicate is True
        record = db.query(SessionRecord).filter_by(session_id="sess-dup").one()
        assert (record.origin, record.edge_report_id) == ("edge-sync", "edge-uuid-1")

    def test_a_duplicate_of_a_plain_session_does_not_gain_an_origin(self, db):
        intake_submission(db, _submission("sess-dup2"))

        intake_submission(db, _submission("sess-dup2"), origin="edge-sync", edge_report_id="edge-uuid-1")

        record = db.query(SessionRecord).filter_by(session_id="sess-dup2").one()
        assert (record.origin, record.edge_report_id) == (None, None)
