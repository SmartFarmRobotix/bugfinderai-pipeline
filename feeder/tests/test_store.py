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
Session-directory data-root containment in
load_replay_session, distinct from app/manifest.py's per-image-filename
containment (test_manifest_contract.py). A session_id is store-controlled
(app/selection.py only ever hands back a name list_store_session_ids
produced), but the store scan follows symlinks (Path.iterdir + is_dir), so
a session_id could still be a symlink planted to point outside data_root.
"""

from __future__ import annotations

import json

import pytest

from app.store import SessionDirectoryEscapesDataRootError, load_replay_session


def _write_conforming_session(session_dir, session_id: str) -> None:
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "images" / "a.jpg").write_bytes(b"data")
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "crop": "brassica-oleracea",
                "session_start": "2026-01-01T00:00:00Z",
                "session_end": "2026-01-01T01:00:00Z",
                "images": [{"filename": "a.jpg"}],
            }
        )
    )


def test_load_replay_session_rejects_outside_symlinked_session_dir(tmp_path):
    data_root = tmp_path / "root"
    (data_root / "sessions").mkdir(parents=True)

    outside = tmp_path / "outside-session"
    _write_conforming_session(outside, "escaped")

    (data_root / "sessions" / "escaped").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionDirectoryEscapesDataRootError):
        load_replay_session(data_root, "escaped")


def test_load_replay_session_accepts_session_dir_within_data_root(tmp_path):
    data_root = tmp_path / "root"
    session_dir = data_root / "sessions" / "sess-ok"
    _write_conforming_session(session_dir, "sess-ok")

    session = load_replay_session(data_root, "sess-ok")
    assert session.session_id == "sess-ok"
    assert [img.filename for img in session.images] == ["a.jpg"]


def test_load_replay_session_accepts_symlink_contained_within_data_root(tmp_path):
    data_root = tmp_path / "root"
    (data_root / "sessions").mkdir(parents=True)

    real = data_root / "real-session"
    _write_conforming_session(real, "aliased")

    (data_root / "sessions" / "aliased").symlink_to(real, target_is_directory=True)

    session = load_replay_session(data_root, "aliased")
    assert session.session_id == "aliased"
