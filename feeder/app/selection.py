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
Session feeder — deterministic replay-session selection.

Default policy: oldest-not-yet-replayed, tracked by a small local JSON
"replayed" marker file so repeated runs progress through the store rather
than re-replaying the same session forever; once every stored session has
been replayed, selection wraps around (cyclic, oldest-first again).
Override: an explicit session id selects a specific stored session
regardless of the marker state (targeted testing / fixed acceptance
fixtures).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.store import list_store_session_ids

logger = logging.getLogger("feeder.selection")


class NoSessionsAvailableError(RuntimeError):
    def __init__(self, data_root: Path) -> None:
        super().__init__(f"no sessions found in store: {data_root}")


class SessionNotFoundError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"explicit session id not found in store: {session_id!r}")


def _marker_path(data_root: Path, marker_filename: str) -> Path:
    return data_root / marker_filename


def _load_marker(path: Path) -> dict:
    if not path.exists():
        return {"replayed": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("replay marker %s unreadable — treating as empty", path)
        return {"replayed": []}
    if not isinstance(data, dict) or not isinstance(data.get("replayed"), list):
        return {"replayed": []}
    return data


def _save_marker(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def select_session_id(
    data_root: Path,
    marker_filename: str = ".replayed.json",
    explicit_session_id: Optional[str] = None,
) -> str:
    """
    Select the session id to replay this run. Deterministic under unchanged
    store state.
    """
    available = list_store_session_ids(data_root)
    if not available:
        raise NoSessionsAvailableError(data_root)

    if explicit_session_id is not None:
        if explicit_session_id not in available:
            raise SessionNotFoundError(explicit_session_id)
        return explicit_session_id

    marker = _load_marker(_marker_path(data_root, marker_filename))
    replayed = set(marker.get("replayed", []))

    candidates = [s for s in available if s not in replayed]
    if not candidates:
        candidates = available

    return candidates[0]


def record_replayed(data_root: Path, marker_filename: str, session_id: str) -> None:
    """Persist that *session_id* has now been replayed (called after a successful submission)."""
    marker_path = _marker_path(data_root, marker_filename)
    marker = _load_marker(marker_path)
    replayed: list[str] = marker.get("replayed", [])

    available = set(list_store_session_ids(data_root))
    if session_id not in replayed:
        if available and available.issubset(set(replayed) | {session_id}):
            replayed = [session_id]
        else:
            replayed.append(session_id)

    _save_marker(marker_path, {"replayed": replayed})
