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
The intake-state endpoint
is the single answer to "is this session id clean?" across every
intake-side table, idempotency key included -- never just the
session-detail record. The key case:
a session id can be "held" purely by a surviving `idempotency_key` row
while `session_record` (and therefore the session-detail GET) has none.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import config as config_module
import gatekeeper
import intake_auth
import main
from db.session import get_db
from models import AttemptLog, IdempotencyKey, ReportIndexRecord, RiskAssessmentRecord, SessionRecord, SessionState
from queries import SessionNotFound, get_intake_state, get_session_detail


def _seed_idempotency_key_only(db, session_id: str) -> None:
    db.add(IdempotencyKey(session_id=session_id, content_hash="deadbeef"))
    db.commit()


def _seed_fully_populated_session(db, session_id: str) -> None:
    db.add(IdempotencyKey(session_id=session_id, content_hash="deadbeef"))
    db.add(
        SessionRecord(
            session_id=session_id,
            state=SessionState.REPORTED.value,
            risk_input_json="{}",
            reporting_envelope_json="{}",
        )
    )
    db.commit()
    db.add(RiskAssessmentRecord(session_id=session_id, path="pdm", status="present", level="moderate"))
    db.add(
        ReportIndexRecord(
            session_id=session_id, report_id="report-1", content_hash="report-hash", origin="cloud-generated"
        )
    )
    db.add(AttemptLog(session_id=session_id, call_type="pdm_assess", attempt_number=1, status="succeeded"))
    db.commit()


class TestGetIntakeStateFunction:
    def test_idempotency_key_alone_is_held_while_session_detail_still_404s(self, db):
        _seed_idempotency_key_only(db, "lonely-key-session")

        result = get_intake_state(db, "lonely-key-session")
        assert result["held"] is True
        assert result["state"] == {
            "idempotency_key": True,
            "session_record": False,
            "risk_assessment_record": False,
            "report_index_record": False,
            "attempt_log": False,
        }

        with pytest.raises(SessionNotFound):
            get_session_detail(db, "lonely-key-session")

    def test_unknown_session_id_is_not_held(self, db):
        result = get_intake_state(db, "never-seen-session")
        assert result["held"] is False
        assert all(value is False for value in result["state"].values())

    def test_fully_populated_session_reports_every_table_held(self, db):
        _seed_fully_populated_session(db, "full-session")

        result = get_intake_state(db, "full-session")
        assert result["held"] is True
        assert result["state"] == {
            "idempotency_key": True,
            "session_record": True,
            "risk_assessment_record": True,
            "report_index_record": True,
            "attempt_log": True,
        }


@pytest.fixture()
def client(db, monkeypatch):
    config_module.get_settings.cache_clear()
    monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "cloud")
    monkeypatch.setenv("ORCH_GATEKEEPER_USERNAME", "orchestrator")
    monkeypatch.setenv("ORCH_GATEKEEPER_PASSWORD", "orchestrator-secret")
    config_module.get_settings.cache_clear()

    def _override_get_db():
        yield db

    main.app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()
        config_module.get_settings.cache_clear()


def _authed(monkeypatch):
    monkeypatch.setattr(gatekeeper.requests, "post", lambda *a, **k: MagicMock(status_code=200))
    monkeypatch.setattr(intake_auth, "_validate_gatekeeper_token", lambda settings, token: None)


class TestIntakeStateEndpointHTTP:
    def test_requires_auth(self, client):
        response = client.get("/api/v1/sessions/some-session/intake-state")
        assert response.status_code == 401

    def test_held_true_while_session_detail_endpoint_still_404s(self, client, monkeypatch, db):
        _authed(monkeypatch)
        _seed_idempotency_key_only(db, "lonely-key-session")

        intake_state = client.get(
            "/api/v1/sessions/lonely-key-session/intake-state", headers={"Authorization": "Bearer test-token"}
        )
        assert intake_state.status_code == 200
        body = intake_state.json()
        assert body["held"] is True
        assert body["state"]["idempotency_key"] is True
        assert body["state"]["session_record"] is False

        session_detail = client.get(
            "/api/v1/sessions/lonely-key-session", headers={"Authorization": "Bearer test-token"}
        )
        assert session_detail.status_code == 404

    def test_fresh_session_id_reports_held_false(self, client, monkeypatch):
        _authed(monkeypatch)
        response = client.get(
            "/api/v1/sessions/never-seen-session/intake-state", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["held"] is False
        assert body["session_id"] == "never-seen-session"
