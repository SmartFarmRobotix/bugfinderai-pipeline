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
Tests for app/selector.py: the Cloud boundary and the edge failure reasons.
"""
from __future__ import annotations

import math
import random

import pytest

from app import selector


def test_constants_equal_the_documented_literal_and_no_edge_threshold_exists():
    assert selector.CLOUD_THRESHOLD_KBPS == 5000
    assert not hasattr(selector, "EDGE_THRESHOLD_KBPS")


@pytest.mark.parametrize(
    "median_kbps,expected",
    [
        (20000, "cloud"),
        (10000, "cloud"),
        (5000, "cloud"),
        (4999, "mixed"),
        (56, "mixed"),
        (40, "mixed"),
        (1, "mixed"),
    ],
)
def test_band_and_boundary_table(median_kbps, expected):
    assert selector.select_scenario(median_kbps=median_kbps) == expected


def test_every_measured_value_below_cloud_boundary_selects_mixed_seeded_sweep():
    """
    This sweep detects drift and mistakes -- any cutoff of non-negligible
    width -- with overwhelming probability. A cutoff placed to land only
    between fixed-seed samples falls outside this test's detection model,
    which targets accidental regressions, not values hand-picked to evade
    this specific sample set. Deliberately NOT a source scan: a source
    scan would fail-open against a cutoff hidden in a helper or lookup
    shape, and would itself contradict the validation comparisons above.
    """
    rng = random.Random(20260902)
    low, high = 1e-9, 5000.0
    log_low, log_high = math.log(low), math.log(high)
    for _ in range(200):
        value = math.exp(rng.uniform(log_low, log_high))
        assert selector.select_scenario(median_kbps=value) == "mixed", value


@pytest.mark.parametrize(
    "failure_reason", ["connection-refused", "timeout", "bad-status", "no-samples"]
)
def test_every_listed_failure_reason_selects_edge(failure_reason):
    assert selector.select_scenario(failure_reason=failure_reason) == "edge"


def test_unknown_failure_reason_raises():
    with pytest.raises(ValueError):
        selector.select_scenario(failure_reason="not-a-pinned-reason")


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0,
        0.0,
        -1,
        -100.5,
    ],
)
def test_degenerate_numeric_inputs_raise_never_default(bad_value):
    with pytest.raises(ValueError):
        selector.select_scenario(median_kbps=bad_value)


@pytest.mark.parametrize("bad_value", ["fast", None, [1, 2], {"kbps": 10}, True, False])
def test_non_number_inputs_raise_never_default(bad_value):
    if bad_value is None:
        with pytest.raises(ValueError):
            selector.select_scenario(median_kbps=bad_value)
    else:
        with pytest.raises(TypeError):
            selector.select_scenario(median_kbps=bad_value)
