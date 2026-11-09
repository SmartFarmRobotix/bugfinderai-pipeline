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
Intake authentication:
  cloud/mixed intake requires a Gatekeeper-validated token, rejected
      before any session processing begins (no session record created).
  edge intake authenticates on the local shared secret alone, with
      zero Gatekeeper calls whether the secret is right or wrong.
  default-credential rejection: edge orchestrator refuses to start
      with a placeholder shared-secret value.
  an edge session completes (intake, persistence, score-band assessment, local report)
      with Gatekeeper unreachable and zero Gatekeeper calls.
"""
from __future__ import annotations

import http.server
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import config as config_module
import main
from config import DeploymentProfile, Settings
from contract import CONTRACT_VERSION, DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput, SessionSubmission
from db.session import get_db
from models import ReportIndexRecord, RiskAssessmentRecord, SessionRecord, SessionState
from service import advance_session, intake_submission


def _submission(session_id: str = "sess-auth-1") -> SessionSubmission:
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
    """A TestClient wired to the same per-test SQLite `db` fixture every
    other test module uses, with the process-wide settings cache cleared so
    each test's env-var monkeypatches actually take effect."""
    config_module.get_settings.cache_clear()

    def _override_get_db():
        yield db

    main.app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()
        config_module.get_settings.cache_clear()


def _cloud_settings(monkeypatch, **overrides):
    monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "cloud")
    monkeypatch.setenv("ORCH_GATEKEEPER_USERNAME", "orchestrator")
    monkeypatch.setenv("ORCH_GATEKEEPER_PASSWORD", "orchestrator-secret")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    config_module.get_settings.cache_clear()


def _edge_settings(monkeypatch, secret: str = "a-real-unique-edge-secret", **overrides):
    monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
    monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", secret)
    monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
    monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    config_module.get_settings.cache_clear()


class _GatekeeperNetworkStandInHandler(http.server.BaseHTTPRequestHandler):
    """A real, listening HTTP server standing in for Gatekeeper — used only
    by the network-level check below. Mocking `requests.post` proves a given
    Python callable was never invoked; it can't prove no *network* attempt
    was made by a code path the test didn't think to patch. A socket that
    genuinely nothing ever connects to is that stronger evidence: every
    accepted connection is recorded in `self.server.hit_paths`."""

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.server.hit_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def gatekeeper_network_standin():
    """Binds a real localhost socket and yields (base_url, hit_paths). Point
    ORCH_GATEKEEPER_BASE_URL at base_url and, unlike every other test in
    this module, do NOT monkeypatch requests.post — any real outbound call
    the code under test makes lands here and is recorded."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _GatekeeperNetworkStandInHandler)
    server.hit_paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server.hit_paths
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestCloudIntakeRequiresGatekeeperToken:
    """Unauthenticated cloud/mixed intake is rejected before session
    processing begins."""

    def test_missing_bearer_token_rejected_no_session_created(self, client, db, monkeypatch):
        _cloud_settings(monkeypatch)

        response = client.post("/api/v1/sessions/", json=_submission().model_dump(mode="json"))

        assert response.status_code == 401
        assert db.query(SessionRecord).count() == 0

    def test_gatekeeper_rejecting_token_is_rejected_no_session_created(self, client, db, monkeypatch):
        _cloud_settings(monkeypatch)
        fake_response = MagicMock(status_code=401)
        monkeypatch.setattr(
            "intake_auth.requests.post", MagicMock(return_value=fake_response)
        )

        response = client.post(
            "/api/v1/sessions/",
            json=_submission().model_dump(mode="json"),
            headers={"Authorization": "Bearer not-a-real-token"},
        )

        assert response.status_code == 401
        assert db.query(SessionRecord).count() == 0

    def test_gatekeeper_accepting_token_admits_the_request(self, client, db, monkeypatch):
        _cloud_settings(monkeypatch)

        validate_calls = []

        def _dispatch(url, *args, **kwargs):
            if "validate_token" in url:
                validate_calls.append((url, args, kwargs))
                return MagicMock(status_code=200)
            if url.endswith("/api/login/"):
                return MagicMock(status_code=200, json=lambda: {"access": "pdm-token"})
            return MagicMock(
                status_code=200,
                json=lambda: {"risk_assessment": {"level": "Low", "details": "test", "provenance": {}}},
            )

        monkeypatch.setattr("requests.post", _dispatch)

        response = client.post(
            "/api/v1/sessions/",
            json=_submission().model_dump(mode="json"),
            headers={"Authorization": "Bearer a-valid-looking-token"},
        )

        assert response.status_code == 200
        assert db.query(SessionRecord).count() == 1
        assert len(validate_calls) == 1
        assert "validate_token" in validate_calls[0][0]


class TestEdgeIntakeSharedSecret:
    """Edge intake authenticates on the local shared secret alone —
    correct secret admits with Gatekeeper unreachable/absent; missing or
    wrong secret is rejected regardless of Gatekeeper's reachability."""

    def test_correct_secret_admits_with_gatekeeper_absent(self, client, db, monkeypatch):
        _edge_settings(monkeypatch, secret="a-real-unique-edge-secret")
        mock_post = MagicMock(side_effect=AssertionError("must never call out to Gatekeeper"))
        monkeypatch.setattr("intake_auth.requests.post", mock_post)

        response = client.post(
            "/api/v1/sessions/",
            json=_submission(session_id="sess-edge-ok").model_dump(mode="json"),
            headers={"X-Edge-Shared-Secret": "a-real-unique-edge-secret"},
        )

        assert response.status_code == 200
        assert db.query(SessionRecord).count() == 1
        mock_post.assert_not_called()

    def test_missing_secret_rejected_no_session_created(self, client, db, monkeypatch):
        _edge_settings(monkeypatch, secret="a-real-unique-edge-secret")

        response = client.post("/api/v1/sessions/", json=_submission(session_id="sess-edge-missing").model_dump(mode="json"))

        assert response.status_code == 401
        assert db.query(SessionRecord).count() == 0

    def test_incorrect_secret_rejected_regardless_of_gatekeeper_reachability(self, client, db, monkeypatch):
        _edge_settings(monkeypatch, secret="a-real-unique-edge-secret")
        mock_post = MagicMock(side_effect=AssertionError("must never call out to Gatekeeper"))
        monkeypatch.setattr("intake_auth.requests.post", mock_post)

        response = client.post(
            "/api/v1/sessions/",
            json=_submission(session_id="sess-edge-wrong").model_dump(mode="json"),
            headers={"X-Edge-Shared-Secret": "not-the-secret"},
        )

        assert response.status_code == 401
        assert db.query(SessionRecord).count() == 0
        mock_post.assert_not_called()


class TestDefaultCredentialRejection:
    """Edge orchestrator refuses to start with a
    placeholder shared-secret value. (GK's admin/admin login rejection is
    OpenAgri-GateKeeper's own test suite, a separate repo.)"""

    @pytest.mark.parametrize(
        "placeholder", ["change-me", "some-key", "dev_jwt_signing_key", "smoketest-jwt-key-change-in-production"]
    )
    def test_placeholder_edge_secret_refused_at_startup(self, monkeypatch, placeholder):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", placeholder)
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_missing_edge_secret_refused_at_startup(self, monkeypatch):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.delenv("ORCH_EDGE_SHARED_SECRET", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_cloud_profile_still_requires_gatekeeper_credentials(self, monkeypatch):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "cloud")
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestFullEdgeSessionNoGatekeeperTraffic:
    """An edge session completes (intake, persistence, score-band assessment, local report) with
    Gatekeeper unreachable and zero Gatekeeper calls.
    The edge-local score-band assessment and report generation run, so this asserts a completed edge
    session (REPORTED, with a local PDF on disk).
    """

    def test_full_edge_session_zero_gatekeeper_network_calls(
        self, client, db, monkeypatch, gatekeeper_network_standin, tmp_path
    ):
        gk_base_url, hit_paths = gatekeeper_network_standin
        _edge_settings(
            monkeypatch,
            secret="a-real-unique-edge-secret",
            ORCH_GATEKEEPER_BASE_URL=gk_base_url,
            ORCH_EDGE_REPORT_DIR=str(tmp_path),
            ORCH_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
            ORCH_EDGE_INFERENCE_BASE_URL="http://127.0.0.1:1",
        )

        response = client.post(
            "/api/v1/sessions/",
            json=_submission(session_id="sess-edge-full-net").model_dump(mode="json"),
            headers={"X-Edge-Shared-Secret": "a-real-unique-edge-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "accepted"
        assert body["state"] == SessionState.REPORTED.value

        record = db.query(SessionRecord).filter_by(session_id="sess-edge-full-net").one()
        assert record.state == SessionState.REPORTED.value
        assert record.park_reason is None

        assessment = db.query(RiskAssessmentRecord).filter_by(
            session_id="sess-edge-full-net", path="edge-score-band"
        ).one()
        assert assessment.status == "present"
        assert assessment.fused_verdict is not None

        report_index = db.query(ReportIndexRecord).filter_by(session_id="sess-edge-full-net").one()
        assert report_index.origin == "edge-local"
        assert (tmp_path / "sess-edge-full-net.pdf").is_file()

        assert hit_paths == []

    def test_service_layer_edge_session_completes_without_gatekeeper_call(self, db, monkeypatch, tmp_path):
        """Same guarantee driven directly through service.py (not HTTP),
        matching test_state_machine.py's style, with a settings stand-in
        that (unlike the FastAPI-layer test) proves advance_session itself
        never reaches for a pdm_call function on the edge profile."""
        _edge_settings(
            monkeypatch,
            secret="a-real-unique-edge-secret",
            ORCH_EDGE_REPORT_DIR=str(tmp_path),
            ORCH_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
            ORCH_EDGE_INFERENCE_BASE_URL="http://127.0.0.1:1",
        )
        intake_submission(db, _submission(session_id="sess-edge-service"))

        def _pdm_call_must_not_be_invoked(settings, risk_input):
            raise AssertionError("edge profile must never attempt the PDM/Gatekeeper call")

        result = advance_session(
            db, "sess-edge-service", settings=config_module.get_settings(), pdm_call=_pdm_call_must_not_be_invoked
        )

        assert result.state == SessionState.REPORTED.value
        assert result.park_reason is None
        assert (tmp_path / "sess-edge-service.pdf").is_file()
