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

"""Deployment-profile-driven ingestion-mode dispatch."""

from __future__ import annotations

import pytest

from app.client import SessionFeederClient
from app.config import DeploymentProfile, IngestionMode
from app.store import load_replay_session
from tests.conftest import make_settings
from tests.stub_inf import StubINF


@pytest.mark.parametrize("profile", [DeploymentProfile.MIXED, DeploymentProfile.EDGE])
def test_mixed_and_edge_use_path_registration_and_send_no_bytes(fixtures_dir, tmp_path, profile):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=profile, REPLAY_DATA_ROOT=str(tmp_path))

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    assert result.ingestion_mode == IngestionMode.PATH.value
    assert len(stub.requests_to("/api/v1/sessions/path")) == 1
    assert stub.requests_to("/api/v1/sessions") == []


@pytest.mark.parametrize("profile", [DeploymentProfile.CLOUD, DeploymentProfile.X86_CI])
def test_cloud_and_x86_ci_use_http_upload_and_make_no_path_call(fixtures_dir, tmp_path, profile):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=profile, REPLAY_DATA_ROOT=str(tmp_path))

    with SessionFeederClient(settings, transport=stub.transport) as client:
        result = client.submit_session(session)

    assert result.success
    assert result.ingestion_mode == IngestionMode.HTTP_UPLOAD.value
    assert len(stub.requests_to("/api/v1/sessions")) == 1
    assert stub.requests_to("/api/v1/sessions/path") == []


def test_mode_is_fixed_per_deployment_not_runtime_negotiated(fixtures_dir, tmp_path):
    session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
    stub = StubINF()
    settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.MIXED, REPLAY_DATA_ROOT=str(tmp_path))

    with SessionFeederClient(settings, transport=stub.transport) as client:
        first = client.submit_session(session)
        second = client.submit_session(session)

    assert first.ingestion_mode == second.ingestion_mode == IngestionMode.PATH.value
    assert len(stub.requests_to("/api/v1/sessions/path")) == 2
