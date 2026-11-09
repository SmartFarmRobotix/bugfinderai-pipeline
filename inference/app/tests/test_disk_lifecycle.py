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
Disk lifecycle tests:
  (a) session start with disk usage inside the 80-90%-used band (fixture:
      85% used) is admitted, and the concurrently-running housekeeping
      pass is confirmed active in the SAME fixture;
  (b) session start at exactly 10% free (90% used) is admitted, at 9%
      free (91% used) is refused with no partial session directory
      created;
  (c) interrupted-housekeeping test: a crash mid-purge never leaves the
      original listed path showing a partially-deleted artifact, and the
      next housekeeping pass either finishes deleting or removes the
      leftover `.purging/<artifact-id>` tombstone.
"""

from __future__ import annotations

import json
from collections import namedtuple

import pytest

from app import disk_lifecycle
from app.config import Settings
from app.tests.test_main import _client_for, _make_settings

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _fake_disk_usage(used_pct: float, total: int = 1_000_000):
    used = int(total * used_pct / 100)
    free = total - used

    def _fn(path):
        return _DiskUsage(total=total, used=used, free=free)

    return _fn


class TestFreeAndUsedPct:
    def test_free_pct_and_used_pct_sum_to_100(self, tmp_path):
        fn = _fake_disk_usage(85.0)
        assert disk_lifecycle.used_pct(tmp_path, fn) == pytest.approx(85.0)
        assert disk_lifecycle.free_pct(tmp_path, fn) == pytest.approx(15.0)


class TestPreflightFixtureBand:
    def test_85_pct_used_admits_and_housekeeping_is_active_same_fixture(self, tmp_path):
        settings = _make_settings(tmp_path, EDGE_DISK_BUDGET_PCT="80", EDGE_DISK_HEADROOM_MIN_PCT="10")
        fn = _fake_disk_usage(85.0)

        disk_lifecycle.check_preflight(settings, disk_usage_fn=fn)

        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert result.budget_triggered is True

    def test_exactly_10_pct_free_admits(self, tmp_path):
        settings = _make_settings(tmp_path, EDGE_DISK_BUDGET_PCT="80", EDGE_DISK_HEADROOM_MIN_PCT="10")
        fn = _fake_disk_usage(90.0)

        disk_lifecycle.check_preflight(settings, disk_usage_fn=fn)

    def test_9_pct_free_refuses(self, tmp_path):
        settings = _make_settings(tmp_path, EDGE_DISK_BUDGET_PCT="80", EDGE_DISK_HEADROOM_MIN_PCT="10")
        fn = _fake_disk_usage(91.0)

        with pytest.raises(disk_lifecycle.DiskAdmissionRefused):
            disk_lifecycle.check_preflight(settings, disk_usage_fn=fn)

    def test_below_budget_does_not_trigger_housekeeping(self, tmp_path):
        settings = _make_settings(tmp_path, EDGE_DISK_BUDGET_PCT="80", EDGE_DISK_HEADROOM_MIN_PCT="10")
        fn = _fake_disk_usage(50.0)

        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)
        assert result.budget_triggered is False
        assert result.purged_sessions == []


class TestAutomaticHousekeepingTriggerFiresWithoutManualCall:
    """The unit-level test above
    proves the two functions COULD compose correctly, but run_housekeeping
    was previously reachable only through a manual call to
    `/internal/housekeeping/run` -- no production path invoked it. This
    test instead drives the real HTTP session-creation endpoint (the only
    call in the whole test) and asserts housekeeping ran as an automatic
    side effect of that single call, under the same 85%-used fixture band,
    with admission still governed only by EDGE_DISK_HEADROOM_MIN_PCT."""

    def test_85_pct_used_session_start_triggers_automatic_housekeeping(self, tmp_path, monkeypatch):
        from app import disk_lifecycle as disk_lifecycle_module
        from app import main as main_module

        calls: list = []
        _real_run_housekeeping = main_module.run_housekeeping

        def _spy_run_housekeeping(settings, *args, **kwargs):
            result = _real_run_housekeeping(settings, *args, **kwargs)
            calls.append(result)
            return result

        monkeypatch.setattr(disk_lifecycle_module, "free_pct", lambda *a, **k: 15.0)
        monkeypatch.setattr(disk_lifecycle_module, "used_pct", lambda *a, **k: 85.0)
        monkeypatch.setattr(main_module, "run_housekeeping", _spy_run_housekeeping)

        client = _client_for(tmp_path, EDGE_DISK_BUDGET_PCT="80", EDGE_DISK_HEADROOM_MIN_PCT="10")

        response = client.post(
            "/api/v1/sessions",
            files=[("files", ("a.jpg", b"fake-jpeg-bytes", "image/jpeg"))],
            data={"session_id": "auto-housekeeping-session"},
        )

        assert response.status_code == 200

        assert len(calls) == 1
        assert calls[0].budget_triggered is True


class TestPreflightRefusalCreatesNoPartialSessionDirectory:
    def test_9_pct_free_refuses_via_http_with_no_partial_session_directory(self, tmp_path, monkeypatch):
        from app import main as main_module

        def _fake_check_preflight(settings):
            raise disk_lifecycle.DiskAdmissionRefused("9% free < 10% headroom (fixture)")

        monkeypatch.setattr(main_module, "check_preflight", _fake_check_preflight)

        client = _client_for(tmp_path)
        response = client.post(
            "/api/v1/sessions",
            files=[("files", ("a.jpg", b"fake-jpeg-bytes", "image/jpeg"))],
            data={"session_id": "refused-session"},
        )
        assert response.status_code == 507

        sessions_root = tmp_path / "sessions"
        assert not (sessions_root / "refused-session").exists()


def _make_session(storage_dir, session_id, synced: bool, mtime: float):
    import os

    session_dir = storage_dir / "sessions" / session_id
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "images" / "img.jpg").write_bytes(b"fake")
    (session_dir / "metadata.json").write_text("{}")
    os.utime(session_dir, (mtime, mtime))

    if synced:
        marker_dir = storage_dir / "sync_queue" / ".synced"
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{session_id}.json").write_text(json.dumps({"session_id": session_id}))


class TestSyncConfirmedMarkerSurvivesManifestDeletion:
    """Purge
    eligibility must not depend on the session orchestrator's OWN queue
    manifest, which that service's own housekeeping pass deletes once its
    own artifacts are purged -- see disk_lifecycle._is_sync_confirmed's
    docstring. These tests never write `<session_id>.manifest.json` at
    all (simulating the orchestrator having already purged it) to prove
    the durable `.synced/<session_id>.json` marker alone is sufficient."""

    def test_sync_confirmed_even_though_manifest_was_already_deleted(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "orch-purged-first", synced=True, mtime=1000)
        assert not (tmp_path / "sync_queue" / "orch-purged-first.manifest.json").exists()

        fn = _fake_disk_usage(95.0)
        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)

        assert result.purged_sessions == ["orch-purged-first"]
        assert not (tmp_path / "sessions" / "orch-purged-first").exists()

    def test_marker_cleared_after_purge(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "sess-marker-cleared", synced=True, mtime=1000)
        marker = tmp_path / "sync_queue" / ".synced" / "sess-marker-cleared.json"
        assert marker.exists()

        fn = _fake_disk_usage(95.0)
        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)

        assert result.purged_sessions == ["sess-marker-cleared"]
        assert not marker.exists()

    def test_absent_marker_is_not_sync_confirmed(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "never-synced", synced=False, mtime=1000)

        fn = _fake_disk_usage(95.0)
        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)

        assert result.purged_sessions == []
        assert (tmp_path / "sessions" / "never-synced").exists()


class TestHousekeepingPurgesSyncConfirmedOldestFirst:
    def test_purges_only_sync_confirmed_sessions(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "synced-session", synced=True, mtime=1000)
        _make_session(tmp_path, "unsynced-session", synced=False, mtime=2000)

        fn = _fake_disk_usage(95.0)
        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=fn)

        assert result.purged_sessions == ["synced-session"]
        assert not (tmp_path / "sessions" / "synced-session").exists()
        assert (tmp_path / "sessions" / "unsynced-session").exists()

    def test_purges_oldest_first(self, tmp_path):
        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "newer-synced", synced=True, mtime=2000)
        _make_session(tmp_path, "older-synced", synced=True, mtime=1000)

        calls = {"n": 0}

        def _fn(path):
            calls["n"] += 1
            return _DiskUsage(total=1000, used=850, free=150) if calls["n"] <= 2 else _DiskUsage(
                total=1000, used=700, free=300
            )

        result = disk_lifecycle.run_housekeeping(settings, disk_usage_fn=_fn)
        assert result.purged_sessions == ["older-synced"]
        assert (tmp_path / "sessions" / "newer-synced").exists()


class TestInterruptedHousekeeping:
    def test_leftover_tombstone_never_visible_as_active_and_is_resumed(self, tmp_path):
        from pathlib import Path

        settings = _make_settings(
            tmp_path,
            EDGE_DISK_BUDGET_PCT="80",
            EDGE_DISK_HEADROOM_MIN_PCT="10",
            INFERENCE_SYNC_QUEUE_DIR=str(tmp_path / "sync_queue"),
        )
        _make_session(tmp_path, "crashed-session", synced=True, mtime=1000)

        tombstone = disk_lifecycle._stage_for_purge(
            settings.storage_dir, Path("sessions") / "crashed-session", "sessions-crashed-session"
        )
        assert tombstone is not None
        assert tombstone.is_dir()

        assert not (tmp_path / "sessions" / "crashed-session").exists()

        resumed = disk_lifecycle.resume_leftover_tombstones(settings)
        assert resumed == ["sessions-crashed-session"]
        assert not tombstone.exists()

        assert disk_lifecycle.resume_leftover_tombstones(settings) == []
