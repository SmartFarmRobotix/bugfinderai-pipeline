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
accounting for this component's own outbound calls to the inference service.

In ``http_upload`` mode (cloud/x86_ci profiles) this feeder's calls to the inference service
*are* the device-to-cloud transfer -- the inference service is cloud-resident
there, so this is the upload a device-to-cloud link would carry.
In ``path`` mode (mixed/edge profiles) these same calls target the
edge-local inference service over the same host/network as this component and never cross the edge/cloud boundary at all -- by design they are measured (for
audit) but never counted toward the boundary MB metric. What *does* cross
the boundary in mixed mode is the inference service's own further outbound calls (session
submission, best-image push) -- see inference/app/accounting.py,
the other half of this story.

One JSON record per session, persisted under ``<data_root>/accounting`` so
it is visible in the data directory shared with the inference service (DATA_DIR) in the mixed
topology, plus optionally mirrored to a single host-visible file
(``REPLAY_ACCOUNTING_FILE``) for direct-python invocations
that have no shared container volume to read from.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import DeploymentProfile, IngestionMode, Settings

_DIRNAME = "accounting"

_SCENARIO_BY_PROFILE = {
    DeploymentProfile.CLOUD: "cloud",
    DeploymentProfile.X86_CI: "cloud",
    DeploymentProfile.MIXED: "mixed",
    DeploymentProfile.EDGE: "edge",
}


def scenario_for(settings: Settings) -> str:
    return _SCENARIO_BY_PROFILE[settings.deployment_profile]


def crosses_boundary(settings: Settings) -> bool:
    return settings.ingestion_mode is IngestionMode.HTTP_UPLOAD


def _accounting_dir(settings: Settings) -> Path:
    d = settings.data_root / _DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(settings: Settings, session_id: str) -> Path:
    return _accounting_dir(settings) / f"{session_id}.json"


def _empty_record(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "scenario": None,
        "bytes_up": 0,
        "bytes_down": 0,
        "mb_up": 0.0,
        "mb_down": 0.0,
        "mb_total": 0.0,
        "transfers": [],
    }


def load_record(settings: Settings, session_id: str) -> Optional[dict[str, Any]]:
    path = _record_path(settings, session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@contextmanager
def _locked_record_file(path: Path) -> Iterator[None]:
    """
    Advisory lock (fcntl.flock) around a read-modify-write critical section.
    The data directory shared with the inference service (DATA_DIR) means this
    component and the inference service (inference/app/accounting.py) can append to
    the *same* per-session record file from two different containers; without
    this, a lost update would silently undercount one side's transfers.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


class ByteAccountant:
    """
    Records this run's outbound transfers for one replayed session. Each
    ``record`` call is its own locked read-modify-write against the
    on-disk record (never a single in-memory snapshot written at the end),
    so it merges cleanly with whatever the inference service has concurrently appended on the
    shared volume instead of overwriting it.
    """

    def __init__(self, settings: Settings, session_id: str) -> None:
        self._settings = settings
        self._session_id = session_id

    def record(
        self,
        *,
        endpoint: str,
        direction: str,
        kind: str,
        byte_count: int,
    ) -> dict[str, Any]:
        crosses = crosses_boundary(self._settings)
        path = _record_path(self._settings, self._session_id)
        with _locked_record_file(path):
            current = load_record(self._settings, self._session_id) or _empty_record(self._session_id)
            current["scenario"] = scenario_for(self._settings)
            if crosses:
                key = "bytes_up" if direction == "up" else "bytes_down"
                current[key] = current.get(key, 0) + byte_count
            current["transfers"].append(
                {
                    "component": "client",
                    "endpoint": endpoint,
                    "direction": direction,
                    "kind": kind,
                    "bytes": byte_count,
                    "crosses_boundary": crosses,
                }
            )
            current["mb_up"] = round(current["bytes_up"] / 1_000_000, 6)
            current["mb_down"] = round(current["bytes_down"] / 1_000_000, 6)
            current["mb_total"] = round((current["bytes_up"] + current["bytes_down"]) / 1_000_000, 6)
            path.write_text(json.dumps(current))
        if self._settings.accounting_file:
            try:
                self._settings.accounting_file.write_text(json.dumps(current))
            except OSError:
                pass
        return current

    def finalize(self) -> Optional[dict[str, Any]]:
        """Return the current persisted record (the source of truth for displayed metrics)."""
        return load_record(self._settings, self._session_id)

    def record_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Persist the link-probe's decision
        block under the ``link_probe_decision`` key -- its own
        locked read-modify-write, kept separate from record because it
        carries no byte count of its own (the probe's bytes are recorded
        through a plain ``record(kind="probe")`` call). Always present in
        the accounting record for an auto-mode session, independent of
        whether a result file is configured.
        """
        path = _record_path(self._settings, self._session_id)
        with _locked_record_file(path):
            current = load_record(self._settings, self._session_id) or _empty_record(self._session_id)
            current["link_probe_decision"] = decision
            path.write_text(json.dumps(current))
        if self._settings.accounting_file:
            try:
                self._settings.accounting_file.write_text(json.dumps(current))
            except OSError:
                pass
        return current


def bytes_of(payload: Any) -> int:
    """Approximate wire size of a JSON body, form fields dict, or raw bytes/str."""
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return len(json.dumps(payload).encode("utf-8"))


def multipart_bytes(form_data: dict[str, Any], file_parts: list[tuple[str, tuple[str, bytes, str]]]) -> int:
    """Approximate multipart/form-data body size: form fields plus every file's raw bytes."""
    total = bytes_of(form_data)
    for _field, (_filename, content, _content_type) in file_parts:
        total += len(content)
    return total
