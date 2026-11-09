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
Edge data lifecycle: this
service's own housekeeping pass, covering the two artifact classes it
owns -- the local report (`settings.edge_report_dir/<session_id>.pdf`)
and the sync-queue entry (`settings.sync_queue_dir/<session_id>.manifest.
json`, `sync_queue.py`). The inference service's own `disk_lifecycle.py`
owns the third/fourth classes (raw images/annotations, session metadata)
with the same design: one policy, applied independently by each process that owns part of
the edge data root, not one shared process.

Sync-confirmed status is read from THIS service's own database -- this process is the authoritative source for that fact, unlike
the inference service, which reads it off the shared sync-queue volume. Purging `<session_id>.manifest.json` here
must never remove the inference service's own purge-eligibility signal for that session:
the inference service stopped reading this manifest for that purpose precisely because this
pass deletes it (see `sync_queue.mark_sync_confirmed` and
`disk_lifecycle._is_sync_confirmed` in the inference service). This pass
purges the manifest unconditionally once eligible; it never touches the
durable `.synced/<session_id>.json` marker in the same `sync_queue_dir` --
that marker outlives this purge by construction, in either purge order.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings
from models import ReportIndexRecord

logger = logging.getLogger("orchestrator.housekeeping")

DiskUsageFn = Callable[[Path], "shutil._ntuple_diskusage"]


def _existing_ancestor(path: Path) -> Path:
    """`shutil.disk_usage` requires an existing path -- walk up to the
    nearest existing ancestor (same filesystem/mount) rather than fail on
    a report/queue directory that hasn't been created yet."""
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def used_pct(path: Path, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> float:
    usage = disk_usage_fn(_existing_ancestor(Path(path)))
    return (usage.used / usage.total) * 100.0


def _purging_dir(root: Path) -> Path:
    return Path(root) / ".purging"


def _stage_for_purge(root: Path, relative_path: Path, artifact_id: str) -> Optional[Path]:
    """Atomic same-filesystem
    move into `root/.purging/<artifact_id>`. Returns None (no-op) if the
    artifact is already absent -- purge is idempotent."""
    src = Path(root) / relative_path
    if not src.exists():
        return None
    purging_dir = _purging_dir(root)
    purging_dir.mkdir(parents=True, exist_ok=True)
    tombstone = purging_dir / artifact_id
    os.replace(src, tombstone)
    return tombstone


def _complete_purge(tombstone: Path) -> None:
    if tombstone.is_dir():
        shutil.rmtree(tombstone)
    else:
        tombstone.unlink()


def resume_leftover_tombstones(settings: Settings) -> List[str]:
    """Startup pass, resumed
    independently for each of this service's two artifact roots. Safe to call unconditionally at import/startup."""
    resumed: List[str] = []
    for root in (Path(settings.edge_report_dir), Path(settings.sync_queue_dir)):
        purging_dir = _purging_dir(root)
        if not purging_dir.is_dir():
            continue
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


def run_housekeeping(db: Session, settings: Settings, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> HousekeepingResult:
    """Independent of session
    processing -- called from the manual `/internal/housekeeping/run`
    endpoint, and from main.py's automatic session-start
    `_kick_housekeeping` background task, but never
    SYNCHRONOUSLY/inline from the intake/advance_session path itself: the
    automatic kick is scheduled via FastAPI BackgroundTasks, which only
    runs after the response has already been built, so a slow purge still
    cannot delay intake. Purges the report + sync-queue entry for
    sync-confirmed sessions (oldest first) only once usage is at or above
    EDGE_DISK_BUDGET_PCT; a failure purging one session is logged and
    skipped, never aborting the rest of the pass."""
    result = HousekeepingResult()
    result.resumed_tombstones = resume_leftover_tombstones(settings)

    if used_pct(settings.edge_report_dir, disk_usage_fn) < settings.edge_disk_budget_pct:
        return result
    result.budget_triggered = True

    eligible = db.scalars(
        select(ReportIndexRecord)
        .where(ReportIndexRecord.origin == "edge-local", ReportIndexRecord.state == "synced")
        .order_by(ReportIndexRecord.created_at.asc())
    ).all()

    for record in eligible:
        if used_pct(settings.edge_report_dir, disk_usage_fn) < settings.edge_disk_budget_pct:
            break
        session_id = record.session_id
        try:
            report_tombstone = _stage_for_purge(
                Path(settings.edge_report_dir), Path(f"{session_id}.pdf"), f"report-{session_id}"
            )
            if report_tombstone is not None:
                _complete_purge(report_tombstone)

            for extra in (f"{session_id}.sync.pdf", f"{session_id}.sync-pair.json"):
                extra_tombstone = _stage_for_purge(Path(settings.edge_report_dir), Path(extra), f"report-{session_id}-{extra.split('.', 1)[1]}")
                if extra_tombstone is not None:
                    _complete_purge(extra_tombstone)

            queue_tombstone = _stage_for_purge(
                Path(settings.sync_queue_dir), Path(f"{session_id}.manifest.json"), f"queue-{session_id}"
            )
            if queue_tombstone is not None:
                _complete_purge(queue_tombstone)

            result.purged_sessions.append(session_id)
        except OSError as exc:
            logger.error("housekeeping: failed to purge session_id=%s: %s", session_id, exc)
            result.errors.append(f"{session_id}: {exc}")
            continue

    return result
