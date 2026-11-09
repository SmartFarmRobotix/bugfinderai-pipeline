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
Jetson data lifecycle:
low-space preflight/backpressure and the housekeeping
purge pass for this service's own session artifacts
(`storage.py`'s `sessions/<id>/{images/,metadata.json,results.json}`).

Sync-confirmed status is the single purge gate: every artifact class purges only once that session's
report is confirmed synced, never earlier. It is read directly off the shared sync-queue volume
(`settings.sync_queue_dir`), not via a cross-service HTTP call: both
processes mount the same named volume (deploy/edge/compose.yml). It reads a durable per-session marker
(`.synced/<session_id>.json`, written by the session orchestrator's
`sync_queue.mark_sync_confirmed`) rather than that service's own
transient queue manifest (`<session_id>.manifest.json`) -- the manifest is
deleted by the orchestrator's own housekeeping pass once ITS artifacts are
purged, which made purge eligibility depend on cross-service purge order
when read directly. See `_is_sync_confirmed`'s docstring below.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from app.config import Settings

logger = logging.getLogger("inference.disk_lifecycle")

DiskUsageFn = Callable[[Path], "shutil._ntuple_diskusage"]


class DiskAdmissionRefused(RuntimeError):
    """Free space is below
    EDGE_DISK_HEADROOM_MIN_PCT -- the caller must refuse the new session
    before creating any artifact directory for it."""


def _existing_ancestor(path: Path) -> Path:
    """`shutil.disk_usage` requires an existing path. `storage_dir` is
    always a pre-existing mounted volume in a real deployment, but a fresh
    edge node (or a test) may call this before the directory has been
    created yet -- walk up to the nearest existing ancestor (same
    filesystem/mount, so the usage figures are identical) rather than
    fail with an unrelated FileNotFoundError."""
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def free_pct(path: Path, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> float:
    usage = disk_usage_fn(_existing_ancestor(Path(path)))
    return (usage.free / usage.total) * 100.0


def used_pct(path: Path, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> float:
    usage = disk_usage_fn(_existing_ancestor(Path(path)))
    return (usage.used / usage.total) * 100.0


def check_preflight(settings: Settings, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> None:
    """Checked against
    EDGE_DISK_HEADROOM_MIN_PCT only -- NEVER EDGE_DISK_BUDGET_PCT. Must be called before any
    write for the new session (main.py calls this as the first statement
    in create_session/create_session_from_path, before save_upload_file
    or any directory creation)."""
    free = free_pct(settings.storage_dir, disk_usage_fn)
    if free < settings.edge_disk_headroom_min_pct:
        raise DiskAdmissionRefused(
            f"insufficient free disk space to start a new session: {free:.2f}% free, "
            f"below the required EDGE_DISK_HEADROOM_MIN_PCT={settings.edge_disk_headroom_min_pct}% "
            "(EDGE_DISK_BUDGET_PCT does not gate admission)"
        )


def _sync_confirmed_marker_path(sync_queue_dir: Path, session_id: str) -> Path:
    return Path(sync_queue_dir) / ".synced" / f"{session_id}.json"


def _is_sync_confirmed(sync_queue_dir: Path, session_id: str) -> bool:
    """Reads a
    DURABLE per-session marker (`sync_queue_dir/.synced/<session_id>.json`)
    instead of the queue manifest (`<session_id>.manifest.json`) itself.
    the session orchestrator's sync_worker.py writes this marker BEFORE it
    flips the manifest's own `state` to "synced"
    (`sync_queue.mark_sync_confirmed`), and that service's OWN housekeeping
    pass (`housekeeping.py`) purges the manifest once ITS artifacts are
    eligible -- reading the manifest here made purge eligibility depend on
    cross-service purge ORDER: if the orchestrator purged first, this
    service would permanently see the session as never-synced and could
    never purge its own images/annotations either (the exact
    disk-exhaustion outcome housekeeping exists to prevent). The durable marker
    outlives that purge regardless of order; this service removes it
    itself once it purges the session's own artifacts (see
    run_housekeeping, below) -- by then it is the marker's only remaining
    reader, so no cross-service coordination is needed to bound its
    lifetime."""
    return _sync_confirmed_marker_path(sync_queue_dir, session_id).exists()


def _clear_sync_confirmed_marker(sync_queue_dir: Path, session_id: str) -> None:
    """Bounds the durable marker's lifetime (see `_is_sync_confirmed`
    above): once this service has purged a session's own artifacts, it is
    the marker's only remaining reader (the orchestrator's own purge
    eligibility is sourced from its own DB, never this marker), so
    removing it here needs no coordination with the other service's own
    purge timing, whichever order the two run in."""
    _sync_confirmed_marker_path(sync_queue_dir, session_id).unlink(missing_ok=True)


def _sync_confirmed_marker_dir(sync_queue_dir: Path) -> Path:
    return Path(sync_queue_dir) / ".synced"


def _sync_queue_manifest_path(sync_queue_dir: Path, session_id: str) -> Path:
    return Path(sync_queue_dir) / f"{session_id}.manifest.json"


def sweep_orphaned_sync_markers(settings: Settings) -> List[str]:
    """The durable marker can be RE-created for a
    session this service already purged. `sync_worker.py`'s
    `run_sync_once` calls `sync_queue.mark_sync_confirmed` (which writes
    this marker) on every successful push, including a deduplicated
    retry -- so if the orchestrator crashes after `mark_sync_confirmed`
    but before `mark_synced` flips the manifest to SYNCED, the manifest
    stays "uploading", the retry re-uploads and
    calls `mark_sync_confirmed` again. If THIS service had already purged
    that session_id and cleared its marker by then (`run_housekeeping`
    below), the marker comes back for a session_id no longer under
    `sessions_root` -- and `run_housekeeping`'s candidate scan is built
    from `sessions_root.iterdir`, so it never looks at this session_id
    again to clear the marker a second time. Left alone, it leaks
    indefinitely.

    A marker is provably orphaned -- nothing on either side still needs
    it -- once BOTH halves of its purpose are gone: this service's own
    session directory (nothing left to purge here) AND the orchestrator's
    own queue manifest (`<session_id>.manifest.json`, on the same shared
    `sync_queue_dir` volume; its absence means the orchestrator's own
    housekeeping has already purged its side too, so sync is fully done
    with this session). Checking the manifest directly off the shared
    volume mirrors `_is_sync_confirmed`'s own pattern -- no cross-service
    call needed.

    Swept unconditionally, independent of the disk-budget gate (matches
    `resume_leftover_tombstones`): this only ever removes a handful of
    tiny leftover JSON files, never anything under budget/backpressure
    control."""
    marker_dir = _sync_confirmed_marker_dir(settings.sync_queue_dir)
    if not marker_dir.is_dir():
        return []
    sessions_root = settings.storage_dir / "sessions"
    swept: List[str] = []
    for marker_path in sorted(marker_dir.glob("*.json")):
        session_id = marker_path.stem
        if (sessions_root / session_id).exists():
            continue
        if _sync_queue_manifest_path(settings.sync_queue_dir, session_id).exists():
            continue
        try:
            marker_path.unlink()
            swept.append(session_id)
        except OSError as exc:
            logger.warning(
                "housekeeping: failed to sweep orphaned sync-confirmed marker for session_id=%s: %s",
                session_id, exc,
            )
    return swept


def _purging_dir(root: Path) -> Path:
    return Path(root) / ".purging"


def _stage_for_purge(root: Path, relative_path: Path, artifact_id: str) -> Optional[Path]:
    """Move `root/relative_path`
    into `root/.purging/<artifact_id>` -- an atomic same-filesystem rename,
    so a listing of `root` sees either the original artifact (before this
    call returns) or nothing at all under that path (after) -- never a
    partially-deleted artifact under its original, listed path. Returns
    None (no-op) if the artifact is already absent -- purge is idempotent."""
    src = Path(root) / relative_path
    if not src.exists():
        return None
    purging_dir = _purging_dir(root)
    purging_dir.mkdir(parents=True, exist_ok=True)
    tombstone = purging_dir / artifact_id
    os.replace(src, tombstone)
    return tombstone


def _complete_purge(tombstone: Path) -> None:
    """The recursive-delete half of the purge -- not atomic against
    process interruption; a tombstone left behind by an
    interruption here is what `resume_leftover_tombstones` finds and
    finishes on the next startup pass."""
    if tombstone.is_dir():
        shutil.rmtree(tombstone)
    else:
        tombstone.unlink()


def resume_leftover_tombstones(settings: Settings) -> List[str]:
    """a housekeeping pass on startup
    resumes-to-completion (or removes) any `.purging/<artifact-id>`
    tombstone a prior interruption left behind -- it is never treated as
    an active artifact again (it is already outside the listed
    `sessions/` namespace). Safe to call unconditionally at import/
    startup: an empty or absent `.purging/` directory is the normal case
    and this is a no-op then."""
    purging_dir = _purging_dir(settings.storage_dir)
    if not purging_dir.is_dir():
        return []
    resumed: List[str] = []
    for tombstone in sorted(purging_dir.iterdir()):
        try:
            _complete_purge(tombstone)
            resumed.append(tombstone.name)
        except OSError as exc:
            logger.error("housekeeping: failed to resume leftover tombstone %s: %s", tombstone, exc)
    return resumed


@dataclass
class HousekeepingResult:
    resumed_tombstones: List[str] = field(default_factory=list)
    purged_sessions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    budget_triggered: bool = False
    swept_orphan_markers: List[str] = field(default_factory=list)


def run_housekeeping(settings: Settings, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> HousekeepingResult:
    """Independent of session capture/
    inference -- called from the manual
    `/internal/housekeeping/run` endpoint, and from main.py's automatic
    session-start `_kick_housekeeping` background task, but never SYNCHRONOUSLY/inline from the admission or
    session-processing path itself: the automatic kick is scheduled via
    FastAPI BackgroundTasks, which only runs after the response has
    already been built and is being sent, so a slow purge still cannot
    delay admission or block capture/inference.

    1. Resume/clean any leftover `.purging/` tombstone from a prior
       interruption (always, regardless of current disk usage).
    2. Sweep orphaned sync-confirmed markers (see
       `sweep_orphaned_sync_markers`; always, regardless of current disk
       usage).
    3. Only when usage is AT OR ABOVE EDGE_DISK_BUDGET_PCT: purge
       sync-confirmed sessions' artifacts, oldest first, until usage
       returns within budget or no eligible session remains. A failure
       purging one session is logged and skipped -- the rest of the pass,
       and any concurrently running session, are unaffected.
    """
    result = HousekeepingResult()
    result.resumed_tombstones = resume_leftover_tombstones(settings)
    result.swept_orphan_markers = sweep_orphaned_sync_markers(settings)

    if used_pct(settings.storage_dir, disk_usage_fn) < settings.edge_disk_budget_pct:
        return result
    result.budget_triggered = True

    sessions_root = settings.storage_dir / "sessions"
    if not sessions_root.is_dir():
        return result

    candidates = []
    for session_path in sessions_root.iterdir():
        if not session_path.is_dir():
            continue
        session_id = session_path.name
        if not _is_sync_confirmed(settings.sync_queue_dir, session_id):
            continue
        try:
            mtime = session_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, session_id))
    candidates.sort(key=lambda item: item[0])

    for _, session_id in candidates:
        if used_pct(settings.storage_dir, disk_usage_fn) < settings.edge_disk_budget_pct:
            break
        try:
            tombstone = _stage_for_purge(
                settings.storage_dir, Path("sessions") / session_id, f"sessions-{session_id}"
            )
            if tombstone is not None:
                _complete_purge(tombstone)
            result.purged_sessions.append(session_id)
        except OSError as exc:
            logger.error("housekeeping: failed to purge session_id=%s: %s", session_id, exc)
            result.errors.append(f"{session_id}: {exc}")
            continue

        try:
            _clear_sync_confirmed_marker(settings.sync_queue_dir, session_id)
        except OSError as exc:
            logger.warning(
                "housekeeping: purged session_id=%s but failed to clear its sync-confirmed marker: %s",
                session_id, exc,
            )

    return result
