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

"""A periodic housekeeping pass is required, not merely a pass reachable as
a side effect of session activity. This test proves the periodic,
interval-scheduled worker (main.py's `_periodic_housekeeping_loop`) runs
and purges eligible artifacts with ZERO session-submission calls anywhere
in the test -- the one thing `test_automatic_housekeeping_trigger.py`
cannot prove, since that test's only call IS a session submission.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import config as config_module
import housekeeping
import main
from models import ReportIndexRecord


class TestPeriodicHousekeepingRunsWithZeroSessionActivity:
    def test_periodic_pass_purges_with_no_session_submission_at_all(self, db, monkeypatch, tmp_path):
        report_dir = tmp_path / "reports"
        queue_dir = tmp_path / "queue"
        report_dir.mkdir()
        queue_dir.mkdir()

        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", "a-real-unique-edge-secret")
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)
        monkeypatch.setenv("ORCH_EDGE_REPORT_DIR", str(report_dir))
        monkeypatch.setenv("ORCH_SYNC_QUEUE_DIR", str(queue_dir))
        monkeypatch.setenv("EDGE_DISK_BUDGET_PCT", "80")
        monkeypatch.setenv("EDGE_DISK_HEADROOM_MIN_PCT", "10")
        monkeypatch.setenv("EDGE_HOUSEKEEPING_INTERVAL_SEC", "0.05")
        config_module.get_settings.cache_clear()

        (report_dir / "already-synced.pdf").write_bytes(b"%PDF-1.4 fake")
        (queue_dir / "already-synced.manifest.json").write_text('{"state": "synced"}')
        db.add(
            ReportIndexRecord(
                session_id="already-synced",
                report_id=None,
                content_hash="hash",
                origin="edge-local",
                state="synced",
            )
        )
        db.commit()

        monkeypatch.setattr(housekeeping, "used_pct", lambda *a, **k: 85.0)

        monkeypatch.setattr(main, "SessionLocal", sessionmaker(bind=db.get_bind()))

        with TestClient(main.app) as client:  # noqa: F841 - entering triggers lifespan startup
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not (report_dir / "already-synced.pdf").exists():
                    break
                time.sleep(0.05)

        assert not (report_dir / "already-synced.pdf").exists(), (
            "periodic housekeeping pass never purged the pre-existing eligible session "
            "despite zero session-submission activity"
        )
        assert not (queue_dir / "already-synced.manifest.json").exists()

        config_module.get_settings.cache_clear()
