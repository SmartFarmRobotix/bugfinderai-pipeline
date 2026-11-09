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
x86/Jetson parity: the same component code replays a conforming session folder end to end,
exactly the code path the Jetson deployment uses.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.client import SessionFeederClient
from app.config import DeploymentProfile
from app.selection import select_session_id
from app.store import load_replay_session
from tests.conftest import make_settings
from tests.stub_inf import StubINF

FIXTURE_DATA_ROOT = Path(__file__).parent / "fixtures" / "handwritten_conforming"


def test_x86_fixture_replays_end_to_end_against_stub_inf(tmp_path):
    data_root = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA_ROOT, data_root)
    settings = make_settings(
        REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.X86_CI,
        REPLAY_DATA_ROOT=str(data_root),
        REPLAY_MARKER_FILENAME=f".replayed-test-{tmp_path.name}.json",
    )
    session_id = select_session_id(settings.data_root, settings.marker_filename)
    session = load_replay_session(settings.data_root, session_id, settings.manifest_filename)

    assert session.session_id == "sess-001"
    assert len(session.images) == 2
    assert session.skipped_unsupported == []

    stub = StubINF()
    with SessionFeederClient(settings, transport=stub.transport) as client:
        assert client.healthcheck() is True
        result = client.submit_session(session)
        assert result.success
        assert result.session_id == "sess-001"

        push = client.push_session(result.session_id)
        assert push.success and push.pushed

    marker_path = settings.data_root / settings.marker_filename
    marker_path.unlink(missing_ok=True)
