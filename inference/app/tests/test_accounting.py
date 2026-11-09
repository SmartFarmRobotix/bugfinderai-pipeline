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
Unit tests for the inference service's outbound
byte-accounting module.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app import accounting


def test_record_transfer_produces_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp)
        record = accounting.record_transfer(
            storage_dir,
            "sess-1",
            scenario="mixed",
            component="inf",
            endpoint="http://gk/api/proxy/orchestrator/api/v1/sessions/",
            direction="up",
            kind="session_submission",
            byte_count=1200,
            crosses_boundary=True,
        )
    assert record["session_id"] == "sess-1"
    assert record["scenario"] == "mixed"
    assert record["bytes_up"] == 1200
    assert record["bytes_down"] == 0
    assert record["mb_total"] == 0.0012


def test_transfers_accumulate_across_calls_and_persist():
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp)
        accounting.record_transfer(
            storage_dir, "sess-2", scenario="mixed", component="inf",
            endpoint="e1", direction="up", kind="session_submission",
            byte_count=1000, crosses_boundary=True,
        )
        accounting.record_transfer(
            storage_dir, "sess-2", scenario="mixed", component="inf",
            endpoint="e2", direction="up", kind="best_image_pair",
            byte_count=150_000, crosses_boundary=True,
        )
        record = accounting.record_transfer(
            storage_dir, "sess-2", scenario="mixed", component="inf",
            endpoint="e2", direction="down", kind="best_image_pair",
            byte_count=40, crosses_boundary=True,
        )
        assert record["bytes_up"] == 151_000
        assert record["bytes_down"] == 40
        assert len(record["transfers"]) == 3
        reloaded = accounting.load_record(storage_dir, "sess-2")
        assert reloaded == record


def test_non_boundary_transfer_recorded_but_not_counted():
    """Cloud-scenario (single-host) traffic is visible for audit but never
    inflates the boundary MB metric -- that's measured client-side instead."""
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp)
        record = accounting.record_transfer(
            storage_dir, "sess-3", scenario="cloud", component="inf",
            endpoint="e", direction="up", kind="session_submission",
            byte_count=999_999, crosses_boundary=False,
        )
    assert record["bytes_up"] == 0
    assert record["transfers"][0]["bytes"] == 999_999
    assert record["transfers"][0]["crosses_boundary"] is False


def test_edge_scenario_transfer_is_recorded_and_not_counted():
    """Edge is a third, distinct scenario label --
    record_transfer must carry it through verbatim, same as cloud/mixed.
    The edge profile's push to the edge-local orchestrator never
    crosses a real boundary, so crosses_boundary=False here too."""
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp)
        record = accounting.record_transfer(
            storage_dir, "sess-4", scenario="edge", component="inf",
            endpoint="http://orchestrator:8020/api/v1/sessions/",
            direction="up", kind="session_submission",
            byte_count=1200, crosses_boundary=False,
        )
    assert record["scenario"] == "edge"
    assert record["bytes_up"] == 0
    assert record["transfers"][0]["bytes"] == 1200
    assert record["transfers"][0]["crosses_boundary"] is False


def test_check_tolerance_within_and_beyond():
    within, pct = accounting.check_tolerance(1000, 1000, 5.0)
    assert within is True and pct == 0.0

    within, pct = accounting.check_tolerance(1040, 1000, 5.0)
    assert within is True and round(pct, 2) == 4.0

    within, pct = accounting.check_tolerance(1200, 1000, 5.0)
    assert within is False and round(pct, 2) == 20.0


def test_check_tolerance_zero_transport_baseline():
    within, _ = accounting.check_tolerance(0, 0, 5.0)
    assert within is True
    within, _ = accounting.check_tolerance(500, 0, 5.0)
    assert within is False


def test_bytes_of_json_and_raw():
    assert accounting.bytes_of({"a": 1}) == len(b'{"a": 1}')
    assert accounting.bytes_of("hello") == 5
    assert accounting.bytes_of(b"hello") == 5
