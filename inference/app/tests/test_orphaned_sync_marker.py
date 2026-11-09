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

"""The durable
sync-confirmed marker (`.synced/<session_id>.json`) can be RE-created for
a session this service already purged -- the session orchestrator's
sync_worker.py calls `sync_queue.mark_sync_confirmed` (which writes this
marker) on every successful push, including a retried/deduplicated one.
If the orchestrator crashes after `mark_sync_confirmed` but before
`mark_synced` flips its manifest to SYNCED, the manifest stays
"uploading"; the retry re-uploads (idempotently) and re-creates the
marker. If this service had already purged that session_id by then, the
marker comes back for a session_id no longer under `sessions_root`, and
`run_housekeeping`'s own candidate scan (built from
`sessions_root.iterdir`) never looks at it again -- the marker leaks
indefinitely.

`disk_lifecycle.sweep_orphaned_sync_markers` closes this: a marker is
swept once BOTH halves of its purpose are gone -- this service's own
session directory, AND the orchestrator's own queue manifest on the same
shared `sync_queue_dir` volume (its absence means the orchestrator's own
housekeeping already purged its side too).
"""
from __future__ import annotations

import json

from app import disk_lifecycle
from app.tests.test_disk_lifecycle import _fake_disk_usage, _make_session
from app.tests.test_main import _make_settings


class TestSweepOrphanedSyncMarkers:
    def test_orphan_marker_swept_when_session_dir_and_manifest_both_gone(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        marker = tmp_path / "sync_queue" / ".synced" / "orphan-session.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"session_id": "orphan-session"}))

        swept = disk_lifecycle.sweep_orphaned_sync_markers(settings)

        assert swept == ["orphan-session"]
        assert not marker.exists()

    def test_marker_not_swept_while_manifest_still_present(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        marker = tmp_path / "sync_queue" / ".synced" / "retry-in-progress.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"session_id": "retry-in-progress"}))
        manifest = tmp_path / "sync_queue" / "retry-in-progress.manifest.json"
        manifest.write_text(json.dumps({"session_id": "retry-in-progress", "state": "synced"}))

        swept = disk_lifecycle.sweep_orphaned_sync_markers(settings)

        assert swept == []
        assert marker.exists()

    def test_marker_not_swept_while_session_directory_still_present(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "still-active", synced=True, mtime=1000)
        marker = tmp_path / "sync_queue" / ".synced" / "still-active.json"
        assert marker.exists()

        swept = disk_lifecycle.sweep_orphaned_sync_markers(settings)

        assert swept == []
        assert marker.exists()


class TestOrphanedMarkerSweepSequence:
    """Fast, inference-only sweep-sequence check: writes the marker/manifest
    FILES that the session orchestrator's sync_queue.py would produce at
    each step of the orphaned-marker crash sequence, in the same order, and
    asserts the inference service's own `disk_lifecycle` behaves correctly against each
    resulting state. This is a SIMULATION of the sequence's file states,
    not a driver of it: it never calls `sync_queue.mark_sync_confirmed`,
    `mark_synced`, or the session orchestrator's own housekeeping:
    those real calls are only in-process importable from that service's
    own venv (disjoint dependency set -- SQLAlchemy/alembic vs none
    from this one), so a same-process, single-venv test cannot exercise
    them. Useful as a quick regression for the inference service's own read/purge/sweep
    logic in isolation; it does NOT prove the orchestrator's real
    mark_sync_confirmed/mark_synced/housekeeping calls, wired together
    with these, actually produce this sequence -- that cross-service sequence is not covered here."""

    def test_marker_recreated_after_purge_is_eventually_swept_not_leaked(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        session_id = "crash-between-confirm-and-synced"
        marker = tmp_path / "sync_queue" / ".synced" / f"{session_id}.json"
        manifest = tmp_path / "sync_queue" / f"{session_id}.manifest.json"

        _make_session(tmp_path, session_id, synced=False, mtime=1000)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"session_id": session_id, "state": "uploading"}))
        assert not marker.exists()

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"session_id": session_id}))
        assert manifest.exists()
        assert json.loads(manifest.read_text())["state"] == "uploading"

        fn = _fake_disk_usage(95.0)
        first_pass = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert first_pass.purged_sessions == [session_id]
        assert not (tmp_path / "sessions" / session_id).exists()
        assert not marker.exists()
        assert manifest.exists()

        marker.write_text(json.dumps({"session_id": session_id}))
        manifest.write_text(json.dumps({"session_id": session_id, "state": "synced"}))
        assert marker.exists()

        interim_pass = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert interim_pass.swept_orphan_markers == []
        assert marker.exists()

        manifest.unlink()

        second_pass = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert second_pass.swept_orphan_markers == [session_id]
        assert not marker.exists()

    def test_recreated_marker_survives_passes_until_manifest_also_gone(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        session_id = "orch-not-done-yet"
        marker = tmp_path / "sync_queue" / ".synced" / f"{session_id}.json"
        manifest = tmp_path / "sync_queue" / f"{session_id}.manifest.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"session_id": session_id}))
        manifest.write_text(json.dumps({"session_id": session_id, "state": "synced"}))

        fn = _fake_disk_usage(95.0)

        for _ in range(3):
            result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
            assert result.swept_orphan_markers == []
            assert marker.exists()

        manifest.unlink()
        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert result.swept_orphan_markers == [session_id]
        assert not marker.exists()
