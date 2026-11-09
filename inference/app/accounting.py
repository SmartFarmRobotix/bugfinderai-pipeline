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
Per-session application-level byte
accounting for the inference service's own outbound calls (GateKeeper auth, the session
submission push to the orchestrator, and the best-image pair push to
Reporting).

This is the boundary-crossing side of the accounting story in mixed mode:
the feeder's own local calls to the edge-resident inference service
never cross the edge/cloud boundary and so contribute nothing here; what crosses is
exactly what this module wraps -- the inference service's own outbound HTTP calls to the
published GateKeeper endpoint.

One JSON record per session, persisted under ``<storage_dir>/accounting``
so it survives process restarts and is readable both by this service's own
``GET /api/v1/sessions/{id}/accounting`` endpoint and directly off disk by
anything sharing the volume (a directory shared with the feeder).
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_DIRNAME = "accounting"


def _accounting_dir(storage_dir: Path) -> Path:
    d = storage_dir / _DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(storage_dir: Path, session_id: str) -> Path:
    return _accounting_dir(storage_dir) / f"{session_id}.json"


def load_record(storage_dir: Path, session_id: str) -> Optional[dict[str, Any]]:
    path = _record_path(storage_dir, session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _empty_record(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "scenario": None,
        "profile": None,
        "bytes_up": 0,
        "bytes_down": 0,
        "mb_up": 0.0,
        "mb_down": 0.0,
        "mb_total": 0.0,
        "transfers": [],
    }


@contextmanager
def _locked_record_file(path: Path) -> Iterator[None]:
    """
    Advisory lock (fcntl.flock) around a read-modify-write critical section.
    The directory shared with the feeder (``DATA_DIR``) means this
    service and the session feeder's own accounting.py can append to the
    *same* per-session record file from two different containers; without
    this, a lost update would silently undercount one side's transfers.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def record_transfer(
    storage_dir: Path,
    session_id: str,
    *,
    scenario: str,
    component: str,
    endpoint: str,
    direction: str,
    kind: str,
    byte_count: int,
    crosses_boundary: bool,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    """
    Append one measured transfer to session_id's accounting record and
    persist it. ``direction`` is "up" (request bytes leaving this service)
    or "down" (response bytes received). Only transfers with
    ``crosses_boundary=True`` count toward ``bytes_up``/``bytes_down`` --
    same-host/local traffic is measured (visible in ``transfers`` for
    audit) but never inflates the boundary metric.
    """
    path = _record_path(storage_dir, session_id)
    with _locked_record_file(path):
        record = load_record(storage_dir, session_id) or _empty_record(session_id)
        record["session_id"] = session_id
        record["scenario"] = scenario
        if profile is not None:
            record["profile"] = profile

        if crosses_boundary:
            key = "bytes_up" if direction == "up" else "bytes_down"
            record[key] = record.get(key, 0) + byte_count

        record.setdefault("transfers", []).append(
            {
                "component": component,
                "endpoint": endpoint,
                "direction": direction,
                "kind": kind,
                "bytes": byte_count,
                "crosses_boundary": crosses_boundary,
            }
        )

        record["mb_up"] = round(record["bytes_up"] / 1_000_000, 6)
        record["mb_down"] = round(record["bytes_down"] / 1_000_000, 6)
        record["mb_total"] = round((record["bytes_up"] + record["bytes_down"]) / 1_000_000, 6)

        path.write_text(json.dumps(record))
    return record


def bytes_of(payload: Any) -> int:
    """Approximate wire size of a JSON-serializable payload or raw string/bytes."""
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return len(json.dumps(payload).encode("utf-8"))


def check_tolerance(
    app_level_bytes: int, transport_level_bytes: int, tolerance_pct: float
) -> tuple[bool, float]:
    """
    Application-level totals must match an independent
    transport-level measure within ``tolerance_pct`` of the transport-level
    total. Returns (within_tolerance, pct_diff).
    """
    if transport_level_bytes == 0:
        return app_level_bytes == 0, (0.0 if app_level_bytes == 0 else float("inf"))
    pct_diff = abs(app_level_bytes - transport_level_bytes) / transport_level_bytes * 100.0
    return pct_diff <= tolerance_pct, pct_diff
