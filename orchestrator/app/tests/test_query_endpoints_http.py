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

"""The query APIs are reachable over HTTP (not just as Python
functions) and sit behind the same auth as intake — a caller with no
bearer token gets 401, never session data.

Also: the session
detail's `report.images_pending`, both through the live endpoint (the
Reporting-service status call mocked at the HTTP boundary) and directly
against `reporting_client.fetch_report_status`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import config as config_module
import gatekeeper
import main
import reporting_client
from contract import (
    CONTRACT_VERSION,
    ConversionChainProvenance,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from db.session import get_db
from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary
from models import IdempotencyKey, SessionRecord
from pdm_client import PdmAssessmentResult
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


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


class TestQueryEndpointsRequireAuth:
    def test_get_session_without_token_is_rejected(self, client):
        response = client.get("/api/v1/sessions/does-not-exist")
        assert response.status_code == 401

    def test_list_sessions_without_token_is_rejected(self, client):
        response = client.get("/api/v1/sessions/")
        assert response.status_code == 401


class TestQueryEndpointsWithValidToken:
    def test_get_unknown_session_is_404(self, client, monkeypatch):
        monkeypatch.setattr(
            gatekeeper.requests,
            "post",
            lambda *a, **k: MagicMock(status_code=200),
        )
        import intake_auth

        monkeypatch.setattr(intake_auth, "_validate_gatekeeper_token", lambda settings, token: None)

        response = client.get(
            "/api/v1/sessions/does-not-exist", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 404

    def test_list_sessions_returns_empty_list_when_store_is_empty(self, client, monkeypatch):
        import intake_auth

        monkeypatch.setattr(intake_auth, "_validate_gatekeeper_token", lambda settings, token: None)

        response = client.get("/api/v1/sessions/", headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200
        assert response.json() == []


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
            conversion_chain=ConversionChainProvenance(
                execution_target="cloud",
                pt_version="1.0",
                pt_source="local",
                pt_license="MIT",
                onnx_artifact="model.onnx",
                onnx_sha256="a" * 64,
                engine_artifact="model.engine",
                engine_sha256="b" * 64,
            ),
        ),
    )
    return SessionSubmission(
        contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope
    )


def _seed_cloud_reported_session(db, session_id: str) -> None:
    """A session reaching "reported" with a cloud-generated report id --
    the precondition `report.images_pending` is only ever set for."""

    def _pdm_succeeds(settings, risk_input):
        return PdmAssessmentResult(level="High", details="ok", provenance={})

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
        return ReportPushResult(report_id="report-uuid-http-1")

    intake_submission(db, _submission(session_id))
    advance_session(
        db,
        session_id,
        settings=_FakeSettings(),
        pdm_call=_pdm_succeeds,
        fuzzy_call=_fuzzy_succeeds,
        report_call=_report_succeeds,
    )


class TestSessionDetailImagesPendingOverHTTP:
    def test_session_detail_carries_images_pending_from_a_stubbed_status_call(self, db, client, monkeypatch):
        import intake_auth

        _seed_cloud_reported_session(db, "sess-http-pending-1")
        monkeypatch.setattr(intake_auth, "_validate_gatekeeper_token", lambda settings, token: None)
        monkeypatch.setattr(
            gatekeeper.requests,
            "post",
            lambda *a, **k: MagicMock(status_code=200, json=lambda: {"access": "test-token"}),
        )

        def _fake_get(url, headers=None, timeout=None):
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "session_id": "sess-http-pending-1",
                "report_uuid": "report-uuid-http-1",
                "images_pending": True,
            }
            response.raise_for_status = lambda: None
            return response

        monkeypatch.setattr(reporting_client.requests, "get", _fake_get)

        response = client.get(
            "/api/v1/sessions/sess-http-pending-1", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200
        report = response.json()["report"]
        assert report["report_id"] == "report-uuid-http-1"
        assert report["images_pending"] is True


class TestFetchReportStatusHTTPBoundary:
    """reporting_client.fetch_report_status with only the HTTP boundary
    mocked, mirroring test_reporting_client.py's `push_report` pattern."""

    class _Settings:
        gatekeeper_base_url = "http://gatekeeper:8000"
        gatekeeper_username = "orchestrator"
        gatekeeper_password = "orchestrator-secret"
        gatekeeper_token_endpoint = "/api/login/"
        gatekeeper_timeout_seconds = 10
        rs_service_name = "reporting"
        rs_report_status_path = "/api/v1/openagri-report/pest-session-report/by-session/{session_id}/status"
        rs_call_timeout_seconds = 15

    def _mock_login(self, monkeypatch):
        login_response = MagicMock(status_code=200)
        login_response.json.return_value = {"access": "test-token"}
        monkeypatch.setattr(gatekeeper.requests, "post", lambda *a, **k: login_response)

    def test_200_with_images_pending_true_returns_true(self, monkeypatch):
        self._mock_login(monkeypatch)
        response = MagicMock(status_code=200)
        response.json.return_value = {"images_pending": True}
        response.raise_for_status = lambda: None
        monkeypatch.setattr(reporting_client.requests, "get", lambda *a, **k: response)

        result = reporting_client.fetch_report_status(self._Settings(), "sess-1")
        assert result is True

    def test_503_returns_none(self, monkeypatch):
        self._mock_login(monkeypatch)

        def _fake_get(*a, **k):
            response = MagicMock(status_code=503, text="unavailable")

            def _raise():
                raise reporting_client.requests.HTTPError("503 unavailable")

            response.raise_for_status = _raise
            return response

        monkeypatch.setattr(reporting_client.requests, "get", _fake_get)

        result = reporting_client.fetch_report_status(self._Settings(), "sess-1")
        assert result is None

    def test_302_with_images_pending_body_returns_none(self, monkeypatch):
        self._mock_login(monkeypatch)
        response = MagicMock(status_code=302)
        response.json.return_value = {"images_pending": True}
        response.raise_for_status = lambda: None
        monkeypatch.setattr(reporting_client.requests, "get", lambda *a, **k: response)

        result = reporting_client.fetch_report_status(self._Settings(), "sess-1")
        assert result is None

    def test_body_without_the_field_returns_none(self, monkeypatch):
        self._mock_login(monkeypatch)
        response = MagicMock(status_code=200)
        response.json.return_value = {"session_id": "sess-1", "report_uuid": "report-uuid-1"}
        response.raise_for_status = lambda: None
        monkeypatch.setattr(reporting_client.requests, "get", lambda *a, **k: response)

        result = reporting_client.fetch_report_status(self._Settings(), "sess-1")
        assert result is None


class TestIntakeOriginParameters:
    """The intake accepts
    `origin=edge-sync` with an `edge_report_id`, refuses every other origin
    and `edge-sync` without an id with 422, and stores nothing on a refusal."""

    _AUTH = {"Authorization": "Bearer test-token"}

    @pytest.fixture()
    def authed_client(self, client, monkeypatch):
        import intake_auth

        monkeypatch.setattr(intake_auth, "_validate_gatekeeper_token", lambda settings, token: None)
        monkeypatch.setattr(
            main, "advance_session", lambda db, session_id, **kwargs: db.query(SessionRecord).one()
        )
        return client

    def _post(self, client, session_id, query=""):
        return client.post(
            f"/api/v1/sessions/{query}",
            json=_submission(session_id).model_dump(mode="json"),
            headers=self._AUTH,
        )

    def test_edge_sync_with_an_id_is_accepted_and_stored(self, authed_client, db):
        response = self._post(authed_client, "sess-http-edge", "?origin=edge-sync&edge_report_id=edge-uuid-1")

        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
        record = db.query(SessionRecord).filter_by(session_id="sess-http-edge").one()
        assert (record.origin, record.edge_report_id) == ("edge-sync", "edge-uuid-1")

    def test_a_duplicate_keeps_the_stored_values(self, authed_client, db):
        self._post(authed_client, "sess-http-dup", "?origin=edge-sync&edge_report_id=edge-uuid-1")

        response = self._post(authed_client, "sess-http-dup", "?origin=edge-sync&edge_report_id=edge-uuid-1")

        assert response.json()["status"] == "duplicate"
        record = db.query(SessionRecord).filter_by(session_id="sess-http-dup").one()
        assert (record.origin, record.edge_report_id) == ("edge-sync", "edge-uuid-1")

    def test_a_plain_submission_is_unchanged(self, authed_client, db):
        response = self._post(authed_client, "sess-http-plain")

        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
        record = db.query(SessionRecord).filter_by(session_id="sess-http-plain").one()
        assert (record.origin, record.edge_report_id) == (None, None)

    @pytest.mark.parametrize(
        "query, named",
        [
            ("?origin=other", "other"),
            ("?origin=other&edge_report_id=edge-uuid-1", "other"),
            ("?origin=", "origin"),
            ("?origin=Edge-Sync&edge_report_id=edge-uuid-1", "Edge-Sync"),
            ("?origin=edge-sync", "edge_report_id"),
            ("?origin=edge-sync&edge_report_id=", "edge_report_id"),
            ("?origin=edge-sync&edge_report_id=%20%20", "edge_report_id"),
        ],
    )
    def test_an_unknown_origin_or_edge_sync_without_an_id_is_refused_and_nothing_stored(
        self, authed_client, db, query, named
    ):
        response = self._post(authed_client, "sess-http-refused", query)

        assert response.status_code == 422
        assert named in response.json()["detail"]
        assert db.query(SessionRecord).count() == 0
        assert db.query(IdempotencyKey).count() == 0
