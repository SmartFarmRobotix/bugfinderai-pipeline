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

"""The periodic
housekeeping loop used to call `housekeeping.run_housekeeping` directly
on the event loop (with its DB session created there too), so a slow
purge (a real `rmtree` on the Jetson) blocked request scheduling and
delayed shutdown until the purge returned -- cancellation was clean only
while the loop happened to be asleep between passes. This test proves the
fixed behavior: with a purge genuinely in flight (running in
`asyncio.to_thread`'s worker thread, well short of finishing), process
shutdown (the app's lifespan cancelling and awaiting the periodic task)
completes promptly rather than blocking for the purge's full duration.
"""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

import config as config_module
import housekeeping
import main as main_module

SLOW_PURGE_SECONDS = 3.0
SHUTDOWN_BOUND_SECONDS = 1.5


class TestShutdownStaysPromptWithSlowPurgeInFlight:
    def test_shutdown_does_not_wait_for_an_in_flight_purge_to_finish(self, monkeypatch):
        monkeypatch.setenv("ORCH_DEPLOYMENT_PROFILE", "edge")
        monkeypatch.setenv("ORCH_EDGE_SHARED_SECRET", "a-real-unique-edge-secret")
        monkeypatch.delenv("ORCH_GATEKEEPER_USERNAME", raising=False)
        monkeypatch.delenv("ORCH_GATEKEEPER_PASSWORD", raising=False)
        monkeypatch.setenv("EDGE_DISK_BUDGET_PCT", "80")
        monkeypatch.setenv("EDGE_DISK_HEADROOM_MIN_PCT", "10")
        monkeypatch.setenv("EDGE_HOUSEKEEPING_INTERVAL_SEC", "0.05")
        config_module.get_settings.cache_clear()

        purge_started = threading.Event()

        def _slow_run_housekeeping(_db, _settings):
            purge_started.set()
            time.sleep(SLOW_PURGE_SECONDS)
            return housekeeping.HousekeepingResult(
                purged_sessions=[], resumed_tombstones=[], errors=[], budget_triggered=False
            )

        monkeypatch.setattr(housekeeping, "run_housekeeping", _slow_run_housekeeping)

        with TestClient(main_module.app) as client:  # noqa: F841 - entering triggers lifespan startup
            assert purge_started.wait(timeout=5.0), (
                "periodic housekeeping pass never started -- cannot exercise "
                "an in-flight purge"
            )

            shutdown_start = time.monotonic()
        shutdown_elapsed = time.monotonic() - shutdown_start

        assert shutdown_elapsed < SHUTDOWN_BOUND_SECONDS, (
            f"shutdown took {shutdown_elapsed:.2f}s while a purge was still in flight "
            f"(bound {SHUTDOWN_BOUND_SECONDS}s, purge sleeps {SLOW_PURGE_SECONDS}s) -- "
            "shutdown is blocking on the purge instead of running it off the event loop"
        )

        config_module.get_settings.cache_clear()
