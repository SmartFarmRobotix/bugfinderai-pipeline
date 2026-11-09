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
What ``/healthz`` knows about the detector runtime.

The state is written from two kinds of threads (the detector executor's
worker and the event loop that answers ``/healthz``), so every access goes
through one lock. Times are monotonic seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class HealthState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_detection_ok_at: Optional[float] = None
        self.in_flight: int = 0
        self.last_probe_ok_at: Optional[float] = None
        self.last_probe_error: Optional[str] = None
        self.consecutive_probe_failures: int = 0

    def record_detection_ok(self) -> None:
        """A detection returned: the runtime works, so the failure streak ends."""
        with self._lock:
            self.last_detection_ok_at = time.monotonic()
            self.consecutive_probe_failures = 0

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def record_probe_ok(self) -> None:
        with self._lock:
            self.last_probe_ok_at = time.monotonic()
            self.last_probe_error = None
            self.consecutive_probe_failures = 0

    def record_probe_failure(self, reason: str) -> int:
        """Count one failed probe; returns the consecutive count."""
        with self._lock:
            self.last_probe_error = reason
            self.consecutive_probe_failures += 1
            return self.consecutive_probe_failures

    def proof(self, max_age_seconds: float) -> Optional[str]:
        """The cheapest proof available without running a probe, in the
        order: in-flight, recent detection, recent
        probe. ``None`` means a probe must run now."""
        now = time.monotonic()
        with self._lock:
            if self.in_flight > 0:
                return "in-flight"
            if self.last_detection_ok_at is not None and now - self.last_detection_ok_at < max_age_seconds:
                return "recent-detection"
            if self.last_probe_ok_at is not None and now - self.last_probe_ok_at < max_age_seconds:
                return "recent-probe"
        return None
