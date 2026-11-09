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
Store-and-forward sync queue: a filesystem queue directory holding one sidecar
JSON manifest per queued report -- session id, content hash, the report
PDF's path, and upload state. State transitions are written atomically
(temp file + os.replace, same filesystem) so a power-cycle mid-write never
leaves a corrupt or half-written manifest -- the reader always sees either
the prior state or the fully-written new one. `pdf_path`/`content_hash`
reference the SYNC COPY (`edge_report.py`'s capped `.sync.pdf`), the one
this module's own `enqueue` actually queues and the sync worker actually
uploads; `local_pdf_path`/`local_content_hash` additionally record the
full-resolution local report the sync copy was rendered alongside, for
provenance only -- the sync worker never reads them. `enqueue` refuses
(writing no manifest) a sync copy over `SYNC_COPY_MAX_BYTES`: the
Gatekeeper's body limit makes an oversized upload a certain failure, so
the queue never accepts one in the first place.

No in-process/in-memory state at all: every function re-reads the
manifest from disk on every call. This is deliberate, not an
optimization gap -- it is what makes "queue survives a power-cycle"
trivially true by construction (there is no cache to go stale), and it is
what the module's own tests exercise by constructing a fresh set of
function calls against the same directory rather than reusing any
in-memory handle.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

PENDING = "pending"
UPLOADING = "uploading"
SYNCED = "synced"
FAILED = "failed"
RETIRED = "retired"

FAILURE_BODY_LIMIT = "body-limit"
FAILURE_REJECTED = "rejected"
FAILURE_TRANSIENT = "transient"

REASSESS_PENDING = "pending"
REASSESS_SUBMITTED = "submitted"
REASSESS_DONE = "done"
REASSESS_FAILED = "failed"
_REASSESS_CLOSED = {REASSESS_DONE, REASSESS_FAILED}

SYNC_COPY_MAX_BYTES = 2_000_000

_SYNCABLE_STATES = {PENDING, UPLOADING, FAILED}


class QueueError(RuntimeError):
    """Base class for sync-queue failures."""


class QueueEntryNotFound(QueueError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"no sync-queue entry for session_id={session_id!r}")
        self.session_id = session_id


class SyncQueueRefused(QueueError):
    """The sync copy is too large to queue -- the Gatekeeper would refuse
    the upload with a 413, so `enqueue` refuses it first and writes no
    manifest."""

    def __init__(self, size: int) -> None:
        super().__init__(f"sync copy {size} bytes exceeds the {SYNC_COPY_MAX_BYTES}-byte bound")
        self.size = size


@dataclass
class QueueEntry:
    session_id: str
    content_hash: str
    pdf_path: str
    state: str
    created_at: str
    updated_at: str
    cloud_report_id: Optional[str] = None
    failure_reason: Optional[str] = None
    local_pdf_path: Optional[str] = None
    local_content_hash: Optional[str] = None
    failed_hash: Optional[str] = None
    failure_count: int = 0
    failure_kind: Optional[str] = None
    pair_path: Optional[str] = None
    reassess_state: Optional[str] = None
    reassess_attempts: int = 0
    reassess_reason: Optional[str] = None
    cloud_reassessed_report_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QueueEntry":
        return cls(
            session_id=data["session_id"],
            content_hash=data["content_hash"],
            pdf_path=data["pdf_path"],
            state=data["state"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            cloud_report_id=data.get("cloud_report_id"),
            failure_reason=data.get("failure_reason"),
            local_pdf_path=data.get("local_pdf_path"),
            local_content_hash=data.get("local_content_hash"),
            failed_hash=data.get("failed_hash"),
            failure_count=data.get("failure_count", 0),
            failure_kind=data.get("failure_kind"),
            pair_path=data.get("pair_path"),
            reassess_state=data.get("reassess_state"),
            reassess_attempts=data.get("reassess_attempts", 0),
            reassess_reason=data.get("reassess_reason"),
            cloud_reassessed_report_id=data.get("cloud_reassessed_report_id"),
        )


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _manifest_path(queue_dir: Path, session_id: str) -> Path:
    return Path(queue_dir) / f"{session_id}.manifest.json"


def _retired_manifest_path(queue_dir: Path, session_id: str) -> Path:
    return Path(queue_dir) / ".retired" / f"{session_id}.manifest.json"


def _sync_confirmed_marker_path(queue_dir: Path, session_id: str) -> Path:
    return Path(queue_dir) / ".synced" / f"{session_id}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` to `path` atomically: write to a same-directory temp
    file, then `os.replace` it over the target. `os.replace` is an atomic
    rename on POSIX when source and destination share a filesystem (always
    true here, both under `queue_dir`) -- a reader never observes a
    partially-written manifest, only the prior content or the new content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp_path, path)


def read_entry(queue_dir: Path, session_id: str) -> QueueEntry:
    path = _manifest_path(queue_dir, session_id)
    if not path.exists():
        raise QueueEntryNotFound(session_id)
    return QueueEntry.from_dict(json.loads(path.read_text()))


def enqueue(
    queue_dir: Path,
    session_id: str,
    sync_pdf_path: str,
    sync_hash: str,
    local_pdf_path: Optional[str] = None,
    local_hash: Optional[str] = None,
    pair_path: Optional[str] = None,
) -> QueueEntry:
    """Queue a finished report for upload -- `sync_pdf_path`/`sync_hash`
    are the capped sync copy this entry actually references and uploads;
    `local_pdf_path`/`local_hash` record the full-resolution local report
    it was rendered alongside, for provenance only. `local_pdf_path` and
    `local_hash` are optional for backward compatibility with callers that
    queue a fake PDF that has no local counterpart (the service's own call
    site always passes both); an omitted pair is recorded as null, not "".
    `pair_path` is the capped best-image pair file the cloud re-assessment
    step pushes; optional, recorded as null when omitted.
    Idempotent: if this session_id is already queued, returns the existing
    entry unchanged rather than resetting its state (re-running edge
    report generation for an already-queued session must never silently
    revert an in-flight or already-synced entry back to pending). Refuses
    (writing no manifest) a sync copy over `SYNC_COPY_MAX_BYTES` -- the
    Gatekeeper would refuse the upload anyway."""
    path = _manifest_path(queue_dir, session_id)
    if path.exists():
        return read_entry(queue_dir, session_id)

    size = os.path.getsize(sync_pdf_path)
    if size > SYNC_COPY_MAX_BYTES:
        raise SyncQueueRefused(size)

    now = _utcnow_iso()
    entry = QueueEntry(
        session_id=session_id,
        content_hash=sync_hash,
        pdf_path=sync_pdf_path,
        state=PENDING,
        created_at=now,
        updated_at=now,
        local_pdf_path=local_pdf_path,
        local_content_hash=local_hash,
        pair_path=pair_path,
    )
    _atomic_write_json(path, entry.to_dict())
    return entry


def _transition(queue_dir: Path, session_id: str, **fields) -> QueueEntry:
    entry = read_entry(queue_dir, session_id)
    for key, value in fields.items():
        setattr(entry, key, value)
    entry.updated_at = _utcnow_iso()
    _atomic_write_json(_manifest_path(queue_dir, session_id), entry.to_dict())
    return entry


def mark_uploading(queue_dir: Path, session_id: str) -> QueueEntry:
    return _transition(queue_dir, session_id, state=UPLOADING, failure_reason=None)


def mark_sync_confirmed(queue_dir: Path, session_id: str) -> None:
    """A DURABLE
    per-session marker, separate from this manifest's own `state` field
    written here, BEFORE `mark_synced` below flips the manifest to SYNCED,
    so the inference service's purge-eligibility check (`disk_lifecycle.
    _is_sync_confirmed`) never depends on a file THIS service's own
    housekeeping pass later deletes (`housekeeping.py` purges
    `<session_id>.manifest.json` once its own artifacts are eligible).
    Reading the manifest directly made purge eligibility depend on
    cross-service purge ORDER: if this service purged first, the inference service would
    permanently see the session as never-synced and could never purge its
    own images/annotations either -- the exact disk-exhaustion outcome. This marker outlives that purge; the inference service removes
    it itself once it purges the session's own artifacts (by then the inference service is
    the marker's only remaining reader, so no cross-service coordination
    is needed to bound its lifetime -- see disk_lifecycle.py's
    run_housekeeping). Idempotent: safe to call repeatedly for the same
    session_id. Written via the same atomic temp-file+os.replace idiom as
    every other manifest write in this module (never a partial marker)."""
    path = _sync_confirmed_marker_path(queue_dir, session_id)
    _atomic_write_json(path, {"session_id": session_id, "synced_at": _utcnow_iso()})


def mark_synced(queue_dir: Path, session_id: str, cloud_report_id: str) -> QueueEntry:
    return _transition(
        queue_dir,
        session_id,
        state=SYNCED,
        cloud_report_id=cloud_report_id,
        failure_reason=None,
        failure_count=0,
        failure_kind=None,
    )


def mark_reassess(
    queue_dir: Path,
    session_id: str,
    state: str,
    reason: Optional[str] = None,
    report_id: Optional[str] = None,
    attempt: bool = False,
) -> QueueEntry:
    """Records the cloud re-assessment's state and reason on the manifest,
    touching nothing of the upload (state, marker, report copy). `report_id`
    is kept once recorded (a None never erases it); `attempt` adds one to
    `reassess_attempts`, and a `submitted` or `done` state that is not an
    attempt resets the count, so the intake's and the pair's failures are
    budgeted separately."""
    entry = read_entry(queue_dir, session_id)
    if attempt:
        attempts = entry.reassess_attempts + 1
    elif state in (REASSESS_SUBMITTED, REASSESS_DONE):
        attempts = 0
    else:
        attempts = entry.reassess_attempts
    return _transition(
        queue_dir,
        session_id,
        reassess_state=state,
        reassess_reason=reason,
        reassess_attempts=attempts,
        cloud_reassessed_report_id=report_id if report_id is not None else entry.cloud_reassessed_report_id,
    )


def mark_failed(
    queue_dir: Path,
    session_id: str,
    reason: str,
    failed_hash: Optional[str] = None,
    kind: Optional[str] = None,
) -> QueueEntry:
    """`failed_hash` records the entry's `content_hash` at the moment of
    this failure -- the sync worker uses it to skip an unchanged entry on
    the next run rather than retrying identical bytes against a limit they
    already failed (a changed hash, from a fresh `enqueue`, is retried).
    `kind` is the failure's class (`FAILURE_*`); each failure adds one to
    `failure_count`, which a success resets."""
    entry = read_entry(queue_dir, session_id)
    return _transition(
        queue_dir,
        session_id,
        state=FAILED,
        failure_reason=reason,
        failed_hash=failed_hash,
        failure_kind=kind,
        failure_count=entry.failure_count + 1,
    )


def retire(queue_dir: Path, session_id: str, reason: str) -> QueueEntry:
    """Give up on an entry that cannot sync: its manifest, in state
    `RETIRED` with `reason`, is written atomically to
    `<queue_dir>/.retired/<session_id>.manifest.json` and only then is the
    top-level manifest removed, so `list_entries` (top level only) stops
    listing it and a crash in between leaves the entry queued, never lost.
    The queued report file itself is not touched."""
    entry = read_entry(queue_dir, session_id)
    entry.state = RETIRED
    entry.failure_reason = reason
    entry.updated_at = _utcnow_iso()
    _atomic_write_json(_retired_manifest_path(queue_dir, session_id), entry.to_dict())
    _manifest_path(queue_dir, session_id).unlink(missing_ok=True)
    return entry


def list_entries(queue_dir: Path) -> List[QueueEntry]:
    directory = Path(queue_dir)
    if not directory.is_dir():
        return []
    entries = []
    for manifest_path in sorted(directory.glob("*.manifest.json")):
        try:
            entries.append(QueueEntry.from_dict(json.loads(manifest_path.read_text())))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return entries


def list_syncable(queue_dir: Path) -> List[QueueEntry]:
    """Entries a sync run should (re)attempt, oldest-queued first."""
    return sorted(
        (entry for entry in list_entries(queue_dir) if entry.state in _SYNCABLE_STATES),
        key=lambda entry: entry.created_at,
    )


def list_open_reassessments(queue_dir: Path) -> List[QueueEntry]:
    """Synced entries whose cloud re-assessment is neither `done` nor
    `failed`, oldest-queued first; an entry without the field (synced
    before the step existed) counts as open."""
    return sorted(
        (
            entry
            for entry in list_entries(queue_dir)
            if entry.state == SYNCED and entry.reassess_state not in _REASSESS_CLOSED
        ),
        key=lambda entry: entry.created_at,
    )


def list_retired(queue_dir: Path) -> List[QueueEntry]:
    """Entries the worker retired, oldest-queued first."""
    directory = Path(queue_dir) / ".retired"
    if not directory.is_dir():
        return []
    entries = []
    for manifest_path in directory.glob("*.manifest.json"):
        try:
            entries.append(QueueEntry.from_dict(json.loads(manifest_path.read_text())))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return sorted(entries, key=lambda entry: entry.created_at)
