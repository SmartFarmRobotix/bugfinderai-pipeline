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

"""`run_housekeeping` was
reachable only through a manual call to `/internal/housekeeping/run` --
no production path invoked it. This test drives the real
session-submission HTTP endpoint (the only call in the test) and asserts
a pre-existing sync-confirmed session's report + sync-queue entry are
purged as an automatic side effect of that single call, under the
85%-used fixture band, with
zero manual call to `run_housekeeping` or `/internal/housekeeping/run`
anywhere in this test.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import config as config_module
import housekeeping
import main
from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from db.session import get_db
from models import ReportIndexRecord


def _submission(session_id: str) -> SessionSubmission:
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


@pytest.fixture()
def client(db, monkeypatch):
    config_module.get_settings.cache_clear()

    def _override_get_db():
        yield db

    main.app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr(main, "SessionLocal", sessionmaker(bind=db.get_bind()))
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()
        config_module.get_settings.cache_clear()


class TestAutomaticHousekeepingTriggerFiresWithoutManualCall:
    def test_session_submission_triggers_automatic_housekeeping_85pct_band(
        self, client, db, monkeypatch, tmp_path
    ):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        report_dir.mkdir()
        queue_dir.mkdir()

        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", "a-real-unique-edge-secret")
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)
        monkeypatch.setenv("ORCH_EDGE_REPORT_DIR", str(report_dir))
        monkeypatch.setenv("ORCH_SYNC_QUEUE_DIR", str(queue_dir))
        monkeypatch.setenv("ORCH_EDGE_INFERENCE_BASE_URL", "http://127.0.0.1:1")
        monkeypatch.setenv("EDGE_DISK_BUDGET_PCT", "80")
        monkeypatch.setenv("EDGE_DISK_HEADROOM_MIN_PCT", "10")
        config_module.get_settings.cache_clear()

        (report_dir / "already-synced.pdf").write_bytes(b"%PDF-1.4 fake")
        (queue_dir / "already-synced.manifest.json").write_text('{"state": "synced"}')
        db.add(
            ReportIndexRecord(
                session_id="already-synced",
                report_id=None,
                content_hash="hash",
                origin="edge-local",
                state="synced",
            )
        )
        db.commit()

        monkeypatch.setattr(housekeeping, "used_pct", lambda *a, **k: 85.0)

        response = client.post(
            "/api/v1/sessions/",
            json=_submission(session_id="sess-triggers-hk").model_dump(mode="json"),
            headers={"X-Edge-Shared-Secret": "a-real-unique-edge-secret"},
        )
        assert response.status_code == 200

        assert not (report_dir / "already-synced.pdf").exists()
        assert not (queue_dir / "already-synced.manifest.json").exists()
