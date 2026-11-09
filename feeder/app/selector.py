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
Session feeder — scenario selector.

Cloud is selected at or above CLOUD_THRESHOLD_KBPS (5000 kbps); any lower measured value selects Mixed;
Edge is selected only by probe failure.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional

CLOUD_THRESHOLD_KBPS = 5000

PROBE_FAILURE_REASONS = frozenset({"connection-refused", "timeout", "bad-status", "no-samples"})


def select_scenario(median_kbps: Optional[float] = None, failure_reason: Optional[str] = None) -> str:
    """
    Map a probe result to an effective scenario. Any listed failure
    reason fails closed to edge, reason preserved by the caller. Absent a
    failure, median_kbps must be a finite positive real number -- any
    other shape is a loud error, never a default scenario. Any
    successfully measured value below the Cloud boundary selects mixed,
    whatever its value.
    """
    if failure_reason is not None:
        if failure_reason not in PROBE_FAILURE_REASONS:
            raise ValueError(f"Unknown probe failure reason: {failure_reason!r}")
        return "edge"

    if median_kbps is None:
        raise ValueError("select_scenario requires either median_kbps or failure_reason")
    if isinstance(median_kbps, bool) or not isinstance(median_kbps, Real):
        raise TypeError(f"median_kbps must be a real number, got {median_kbps!r}")

    value = float(median_kbps)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"median_kbps must be finite, got {median_kbps!r}")
    if value <= 0:
        raise ValueError(f"median_kbps must be positive, got {median_kbps!r}")

    if value >= CLOUD_THRESHOLD_KBPS:
        return "cloud"
    return "mixed"
