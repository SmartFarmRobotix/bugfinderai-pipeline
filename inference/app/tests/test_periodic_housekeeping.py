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
and purges eligible artifacts with ZERO session-creation calls anywhere
in the test -- the one thing test_disk_lifecycle.py's
`TestAutomaticHousekeepingTriggerFiresWithoutManualCall` cannot prove,
since that test's only call IS a session creation.
"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from app import config as config_module
from app import disk_lifecycle as disk_lifecycle_module
from app import main as main_module


class TestPeriodicHousekeepingRunsWithZeroSessionActivity:
    def test_periodic_pass_purges_with_no_session_creation_at_all(self, tmp_path, monkeypatch):
        storage_dir = tmp_path
        queue_dir = tmp_path / "sync_queue"

        monkeypatch.setenv("INFERENCE_USING_GATEKEEPER", "false")
        monkeypatch.setenv("INFERENCE_ORCH_PUSH_ENABLED", "false")
        monkeypatch.setenv("INFERENCE_STORAGE_DIR", str(storage_dir))
        monkeypatch.setenv("INFERENCE_SYNC_QUEUE_DIR", str(queue_dir))
        monkeypatch.setenv("EDGE_DISK_BUDGET_PCT", "80")
        monkeypatch.setenv("EDGE_DISK_HEADROOM_MIN_PCT", "10")
        monkeypatch.setenv("EDGE_HOUSEKEEPING_INTERVAL_SEC", "0.05")
        config_module.get_settings.cache_clear()

        session_dir = storage_dir / "sessions" / "already-synced"
        (session_dir / "images").mkdir(parents=True)
        (session_dir / "images" / "img.jpg").write_bytes(b"fake")
        (session_dir / "metadata.json").write_text("{}")
        marker_dir = queue_dir / ".synced"
        marker_dir.mkdir(parents=True)
        (marker_dir / "already-synced.json").write_text(json.dumps({"session_id": "already-synced"}))

        monkeypatch.setattr(disk_lifecycle_module, "free_pct", lambda *a, **k: 15.0)
        monkeypatch.setattr(disk_lifecycle_module, "used_pct", lambda *a, **k: 85.0)

        with TestClient(main_module.app) as client:  # noqa: F841 - entering triggers lifespan startup
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not session_dir.exists():
                    break
                time.sleep(0.05)

        assert not session_dir.exists(), (
            "periodic housekeeping pass never purged the pre-existing eligible session "
            "despite zero session-creation activity"
        )
        assert not (marker_dir / "already-synced.json").exists()

        config_module.get_settings.cache_clear()
