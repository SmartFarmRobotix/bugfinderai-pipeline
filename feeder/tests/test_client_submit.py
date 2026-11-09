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

"""SessionFeederClient submit and push against a stub inference service."""

from __future__ import annotations

from app.client import SessionFeederClient
from app.config import DeploymentProfile
from app.store import load_replay_session
from tests.conftest import make_settings
from tests.stub_inf import StubINF


def test_upload_session_posts_replayed_session_and_receives_session_id(fixtures_dir):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF(next_session_id="created-session-42")
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD)

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    assert result.session_id == "sess-min"
    upload_requests = stub.requests_to("/api/v1/sessions")
    assert len(upload_requests) == 1
    assert upload_requests[0].files.keys() == {"a.jpg"}


def test_push_session_ported_and_parses_pdm_response(fixtures_dir):
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD)

    with SessionFeederClient(settings, transport=stub.transport) as client:
        push = client.push_session("created-session-42")

    assert push.success
    assert push.pushed is True
    assert push.report_id == "report-001"
    assert push.pdm_session_id == "pdm-session-001"


import json  # noqa: E402
import shutil  # noqa: E402


def _session_with_pest(fixtures_dir, tmp_path, pest):
    """The minimal conforming session copied under `tmp_path` with `pest` in
    its manifest, loaded through the store."""
    session_dir = tmp_path / "sessions" / "sess-min"
    shutil.copytree(fixtures_dir / "minimal_conforming" / "sessions" / "sess-min", session_dir)
    manifest = json.loads((session_dir / "manifest.json").read_text())
    manifest["pest"] = pest
    (session_dir / "manifest.json").write_text(json.dumps(manifest))
    return load_replay_session(tmp_path, "sess-min")


def _submit(session, profile, tmp_path):
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=profile, REPLAY_DATA_ROOT=str(tmp_path))
    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)
    assert result.success
    return stub, settings


def test_a_manifest_pest_reaches_the_multipart_form_next_to_crop(fixtures_dir, tmp_path):
    session = _session_with_pest(fixtures_dir, tmp_path, "BRVCBR")
    stub, _ = _submit(session, DeploymentProfile.CLOUD, tmp_path)

    (request,) = stub.requests_to("/api/v1/sessions")
    assert request.form_fields["pest"] == "BRVCBR"
    assert request.form_fields["crop"] == "brassica-oleracea"
    keys = list(request.form_fields)
    assert keys.index("pest") == keys.index("crop") + 1


def test_a_manifest_pest_reaches_the_path_registration_payload_next_to_crop(fixtures_dir, tmp_path):
    session = _session_with_pest(fixtures_dir, tmp_path, "BRVCBR")
    stub, _ = _submit(session, DeploymentProfile.EDGE, tmp_path)

    (request,) = stub.requests_to("/api/v1/sessions/path")
    assert request.json_body["pest"] == "BRVCBR"
    keys = list(request.json_body)
    assert keys.index("pest") == keys.index("crop") + 1


def test_a_manifest_without_a_pest_submits_exactly_as_before_in_the_upload_mode(fixtures_dir):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD)
    with SessionFeederClient(settings, transport=stub.transport) as client:
        client.submit_session(session)

    (request,) = stub.requests_to("/api/v1/sessions")
    assert request.form_fields == {
        "session_id": "sess-min",
        "time_start": "2026-01-01T08:00:00+00:00",
        "time_end": "2026-01-01T08:00:00+00:00",
        "crop": "brassica-oleracea",
        "latitude": "10.0",
        "longitude": "20.0",
        "parcel_id": "parcel-001",
    }
    assert session.pest is None


def test_a_manifest_without_a_pest_submits_exactly_as_before_in_the_path_mode(fixtures_dir):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.EDGE)
    with SessionFeederClient(settings, transport=stub.transport) as client:
        client.submit_session(session)

    (request,) = stub.requests_to("/api/v1/sessions/path")
    assert request.json_body == {
        "session_id": "sess-min",
        "path": str(settings.visible_path_for(session.session_dir)),
        "parcel_id": "parcel-001",
        "time_start": "2026-01-01T08:00:00+00:00",
        "time_end": "2026-01-01T08:00:00+00:00",
        "crop": "brassica-oleracea",
        "growth_stage_representation": None,
        "growth_stage_bbch_min": None,
        "growth_stage_bbch_max": None,
        "growth_stage_stage_id": None,
        "latitude": 10.0,
        "longitude": 20.0,
    }
