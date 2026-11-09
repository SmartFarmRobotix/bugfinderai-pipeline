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
Interrupted housekeeping (this service's own artifacts: the local report and the sync-queue entry): a
crash mid-purge (a) never leaves the original listed path showing a
partially-deleted artifact, and (b) a subsequent housekeeping pass either
finishes deleting or removes the leftover `.purging/<artifact-id>`
tombstone. Plus the sync-confirmed purge gate (never purge ahead of
confirmed sync) and oldest-first ordering.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import housekeeping
from models import ReportIndexRecord

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _fake_disk_usage(used_pct: float, total: int = 1_000_000):
    used = int(total * used_pct / 100)
    free = total - used

    def _fn(path):
        return _DiskUsage(total=total, used=used, free=free)

    return _fn


class _Settings:
    def __init__(self, report_dir, queue_dir, budget=80.0, headroom=10.0):
        self.edge_report_dir = str(report_dir)
        self.sync_queue_dir = str(queue_dir)
        self.edge_disk_budget_pct = budget
        self.edge_disk_headroom_min_pct = headroom


def _make_report_index(db, session_id, state, created_at=None):
    import datetime

    record = ReportIndexRecord(
        session_id=session_id,
        report_id=None,
        content_hash="hash",
        origin="edge-local",
        state=state,
    )
    if created_at is not None:
        record.created_at = created_at
    db.add(record)
    db.commit()
    return record


def _write_report_and_queue_entry(report_dir: Path, queue_dir: Path, session_id: str):
    report_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{session_id}.pdf").write_bytes(b"%PDF-1.4 fake")
    (queue_dir / f"{session_id}.manifest.json").write_text('{"state": "synced"}')


class TestPurgeGate:
    def test_unsynced_session_never_purged_regardless_of_disk_pressure(self, db, tmp_path):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        _write_report_and_queue_entry(report_dir, queue_dir, "sess-unsynced")
        _make_report_index(db, "sess-unsynced", state="generated")

        settings = _Settings(report_dir, queue_dir)
        result = housekeeping.run_housekeeping(db, settings, disk_usage_fn=_fake_disk_usage(95.0))

        assert result.purged_sessions == []
        assert (report_dir / "sess-unsynced.pdf").is_file()
        assert (queue_dir / "sess-unsynced.manifest.json").is_file()

    def test_synced_session_purged_when_over_budget(self, db, tmp_path):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        _write_report_and_queue_entry(report_dir, queue_dir, "sess-synced")
        _make_report_index(db, "sess-synced", state="synced")

        settings = _Settings(report_dir, queue_dir)
        result = housekeeping.run_housekeeping(db, settings, disk_usage_fn=_fake_disk_usage(95.0))

        assert result.purged_sessions == ["sess-synced"]
        assert not (report_dir / "sess-synced.pdf").exists()
        assert not (queue_dir / "sess-synced.manifest.json").exists()

    def test_below_budget_does_not_purge_even_when_synced(self, db, tmp_path):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        _write_report_and_queue_entry(report_dir, queue_dir, "sess-synced")
        _make_report_index(db, "sess-synced", state="synced")

        settings = _Settings(report_dir, queue_dir)
        result = housekeeping.run_housekeeping(db, settings, disk_usage_fn=_fake_disk_usage(50.0))

        assert result.budget_triggered is False
        assert result.purged_sessions == []
        assert (report_dir / "sess-synced.pdf").is_file()


class TestInterruptedHousekeeping:
    def test_leftover_tombstone_never_visible_as_active_and_is_resumed(self, tmp_path):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        _write_report_and_queue_entry(report_dir, queue_dir, "sess-crashed")
        settings = _Settings(report_dir, queue_dir)

        tombstone = housekeeping._stage_for_purge(
            Path(settings.edge_report_dir), Path("sess-crashed.pdf"), "report-sess-crashed"
        )
        assert tombstone is not None
        assert tombstone.is_file()

        assert not (report_dir / "sess-crashed.pdf").exists()

        resumed = housekeeping.resume_leftover_tombstones(settings)
        assert resumed == ["report-sess-crashed"]
        assert not tombstone.exists()

        assert housekeeping.resume_leftover_tombstones(settings) == []


class TestOldestFirst:
    def test_purges_oldest_synced_session_first(self, db, tmp_path):
        import datetime

        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        _write_report_and_queue_entry(report_dir, queue_dir, "newer")
        _write_report_and_queue_entry(report_dir, queue_dir, "older")
        _make_report_index(
            db, "newer", state="synced", created_at=datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
        )
        _make_report_index(
            db, "older", state="synced", created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        )

        settings = _Settings(report_dir, queue_dir)
        calls = {"n": 0}

        def _fn(path):
            calls["n"] += 1
            return _DiskUsage(total=1000, used=850, free=150) if calls["n"] <= 2 else _DiskUsage(
                total=1000, used=700, free=300
            )

        result = housekeeping.run_housekeeping(db, settings, disk_usage_fn=_fn)
        assert result.purged_sessions == ["older"]
        assert (report_dir / "newer.pdf").is_file()
