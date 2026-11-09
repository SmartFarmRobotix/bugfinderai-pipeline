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
Unit tests for this component's
outbound byte-accounting module.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app import accounting
from app.config import DeploymentProfile
from tests.conftest import make_settings


def test_http_upload_mode_crosses_boundary_and_is_counted():
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD, REPLAY_DATA_ROOT=tmp
        )
        accountant = accounting.ByteAccountant(settings, "sess-1")
        accountant.record(endpoint="e", direction="up", kind="session_upload", byte_count=500_000)
        record = accountant.record(endpoint="e", direction="down", kind="session_upload", byte_count=200)

    assert record["scenario"] == "cloud"
    assert record["bytes_up"] == 500_000
    assert record["bytes_down"] == 200
    assert record["transfers"][0]["crosses_boundary"] is True


def test_path_mode_is_local_and_not_counted():
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.MIXED, REPLAY_DATA_ROOT=tmp
        )
        accountant = accounting.ByteAccountant(settings, "sess-2")
        record = accountant.record(
            endpoint="e", direction="up", kind="session_registration_local", byte_count=900_000
        )

    assert record["scenario"] == "mixed"
    assert record["bytes_up"] == 0
    assert record["transfers"][0]["bytes"] == 900_000
    assert record["transfers"][0]["crosses_boundary"] is False


def test_edge_profile_is_labelled_edge_not_mixed():
    """Regression guard: edge and mixed share PATH ingestion (both use the
    same edge-local inference service hop) but must not share a scenario label -- edge and
    mixed are distinct scenarios in the comparison."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.EDGE, REPLAY_DATA_ROOT=tmp
        )
        accountant = accounting.ByteAccountant(settings, "sess-edge")
        record = accountant.record(
            endpoint="e", direction="up", kind="session_registration_local", byte_count=900_000
        )

    assert record["scenario"] == "edge"
    assert record["bytes_up"] == 0
    assert record["transfers"][0]["bytes"] == 900_000
    assert record["transfers"][0]["crosses_boundary"] is False


def test_x86_ci_profile_is_labelled_cloud():
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.X86_CI, REPLAY_DATA_ROOT=tmp
        )
        accountant = accounting.ByteAccountant(settings, "sess-ci")
        record = accountant.record(endpoint="e", direction="up", kind="session_upload", byte_count=500)

    assert record["scenario"] == "cloud"


def test_repeated_record_calls_accumulate_rather_than_overwrite():
    """Separate calls
    (e.g. submit then push) must add up, not clobber each other."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD, REPLAY_DATA_ROOT=tmp
        )
        accounting.ByteAccountant(settings, "sess-3").record(
            endpoint="e1", direction="up", kind="session_upload", byte_count=100
        )
        record = accounting.ByteAccountant(settings, "sess-3").record(
            endpoint="e2", direction="up", kind="push_trigger", byte_count=50
        )

    assert record["bytes_up"] == 150
    assert len(record["transfers"]) == 2


def test_finalize_mirrors_to_accounting_file():
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "mirror.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=tmp,
            REPLAY_ACCOUNTING_FILE=str(mirror),
        )
        accountant = accounting.ByteAccountant(settings, "sess-4")
        accountant.record(endpoint="e", direction="up", kind="session_upload", byte_count=42)
        loaded = accountant.finalize()

        assert mirror.exists()
        assert loaded["bytes_up"] == 42


def test_multipart_bytes_sums_form_fields_and_file_content():
    form_data = {"session_id": "abc"}
    file_parts = [("files", ("a.png", b"1234567890", "image/png"))]
    total = accounting.multipart_bytes(form_data, file_parts)
    assert total == accounting.bytes_of(form_data) + 10
