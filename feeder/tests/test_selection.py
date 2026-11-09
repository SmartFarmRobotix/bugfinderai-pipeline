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

"""Deterministic replay-session selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.selection import (
    NoSessionsAvailableError,
    SessionNotFoundError,
    record_replayed,
    select_session_id,
)


def _make_session(data_root: Path, session_id: str) -> None:
    images_dir = data_root / "sessions" / session_id / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "a.jpg").write_bytes(b"data")
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "crop": "brassica-oleracea",
        "session_start": "2026-01-01T08:00:00+00:00",
        "session_end": "2026-01-01T08:00:00+00:00",
        "images": [{"filename": "a.jpg"}],
    }
    (data_root / "sessions" / session_id / "manifest.json").write_text(json.dumps(manifest))


def test_no_sessions_available_raises(tmp_path):
    with pytest.raises(NoSessionsAvailableError):
        select_session_id(tmp_path)


def test_repeated_runs_are_reproducible(tmp_path):
    _make_session(tmp_path, "session-a")
    _make_session(tmp_path, "session-b")

    first = select_session_id(tmp_path)
    second = select_session_id(tmp_path)
    assert first == second == "session-a"


def test_explicit_session_id_overrides_default_selection(tmp_path):
    _make_session(tmp_path, "session-a")
    _make_session(tmp_path, "session-b")

    selected = select_session_id(tmp_path, explicit_session_id="session-b")
    assert selected == "session-b"


def test_explicit_session_id_not_in_store_raises(tmp_path):
    _make_session(tmp_path, "session-a")
    with pytest.raises(SessionNotFoundError):
        select_session_id(tmp_path, explicit_session_id="does-not-exist")


def test_selection_progresses_after_recording_replay(tmp_path):
    _make_session(tmp_path, "session-a")
    _make_session(tmp_path, "session-b")

    first = select_session_id(tmp_path)
    assert first == "session-a"
    record_replayed(tmp_path, ".replayed.json", first)

    second = select_session_id(tmp_path)
    assert second == "session-b"


def test_selection_wraps_around_once_everything_replayed(tmp_path):
    _make_session(tmp_path, "session-a")
    _make_session(tmp_path, "session-b")

    for _ in range(2):
        picked = select_session_id(tmp_path)
        record_replayed(tmp_path, ".replayed.json", picked)

    third = select_session_id(tmp_path)
    assert third == "session-a"
