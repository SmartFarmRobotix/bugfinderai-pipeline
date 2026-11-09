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
 per-image position/provenance passthrough.
"""

from __future__ import annotations

from dataclasses import replace
import json

from app.client import SessionFeederClient
from app.config import DeploymentProfile
from app.store import load_replay_session
from tests.conftest import make_settings
from tests.stub_inf import StubINF


def test_present_position_is_forwarded_with_replay_provenance_http_upload(fixtures_dir):
    session = load_replay_session(fixtures_dir / "optional_fields", "sess-opt")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD)

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions")
    metadata = json.loads(req.form_fields["metadata"])
    positions = metadata["per_image_positions"]

    assert positions["a.jpg"] == {"latitude": 10.5, "longitude": 20.5, "provenance": "replay"}
    assert "b.png" not in positions


def test_absent_position_produces_no_metadata_field_when_no_image_has_one(fixtures_dir):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD)

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions")
    assert "metadata" not in req.form_fields


def test_session_metadata_coordinates_apply_when_no_environment_override(fixtures_dir):
    session = replace(
        load_replay_session(fixtures_dir / "minimal_conforming", "sess-min"), latitude=10.5, longitude=20.5
    )
    stub = StubINF()
    settings = make_settings(
        REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
        REPLAY_DEFAULT_LATITUDE=None,
        REPLAY_DEFAULT_LONGITUDE=None,
    )

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions")
    assert req.form_fields["latitude"] == "10.5"
    assert req.form_fields["longitude"] == "20.5"


def test_environment_coordinates_override_session_metadata(fixtures_dir):
    session = replace(
        load_replay_session(fixtures_dir / "minimal_conforming", "sess-min"), latitude=10.5, longitude=20.5
    )
    stub = StubINF()
    settings = make_settings(
        REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
        REPLAY_DEFAULT_LATITUDE=11.5,
        REPLAY_DEFAULT_LONGITUDE=21.5,
    )

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions")
    assert req.form_fields["latitude"] == "11.5"
    assert req.form_fields["longitude"] == "21.5"


def test_absent_session_and_environment_coordinates_are_omitted(fixtures_dir):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(
        REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
        REPLAY_DEFAULT_LATITUDE=None,
        REPLAY_DEFAULT_LONGITUDE=None,
    )

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions")
    assert "latitude" not in req.form_fields
    assert "longitude" not in req.form_fields


def test_present_position_is_forwarded_with_replay_provenance_path_mode(fixtures_dir, tmp_path):
    session = load_replay_session(fixtures_dir / "optional_fields", "sess-opt")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.MIXED, REPLAY_DATA_ROOT=str(tmp_path))

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    [req] = stub.requests_to("/api/v1/sessions/path")
    positions = req.json_body["metadata"]["per_image_positions"]
    assert positions["a.jpg"] == {"latitude": 10.5, "longitude": 20.5, "provenance": "replay"}
    assert "b.png" not in positions
