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

"""The decision block's recording (result
file when configured, accounting record always) and probe-byte
accounting attributed as boundary-crossing via the same mechanism the cloud submission bytes use.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List

import httpx
import pytest

from app import accounting, link_probe, main
from app.config import DeploymentProfile
from tests.conftest import make_settings
from tests.stub_inf import StubINF
from tests.test_probe_ramp import _ModelClock, _ModelledTransport


def _args(session_id=None, no_push=False, dry_run=False) -> argparse.Namespace:
    return argparse.Namespace(session_id=session_id, no_push=no_push, dry_run=dry_run)


def _copy_fixture(fixtures_dir: Path, dest: Path) -> Path:
    shutil.copytree(fixtures_dir / "minimal_conforming", dest)
    return dest


class _SequenceClock:
    def __init__(self, timestamps: List[float]) -> None:
        self._it = iter(timestamps)

    def __call__(self) -> float:
        return next(self._it)


class _SuccessTransport:
    def upload(self, url, headers, payload, timeout):
        return link_probe.TransportSample(bytes_written=len(payload))

    def close(self) -> None:
        pass


class _PartialFailureTransport:
    """Reports a specific partial byte count on failure -- never the requested size."""

    def __init__(self, bytes_written: int) -> None:
        self._bytes_written = bytes_written

    def upload(self, url, headers, payload, timeout):
        raise link_probe.ProbeConnectionRefused("refused", bytes_written=self._bytes_written)

    def close(self) -> None:
        pass


class _StalledTransport:
    def __init__(self, bytes_written: int = 0) -> None:
        self._bytes_written = bytes_written

    def upload(self, url, headers, payload, timeout):
        raise link_probe.ProbeTimeout("timed out", bytes_written=self._bytes_written)

    def close(self) -> None:
        pass


def _canonical_bytes(value) -> bytes:
    """
    Byte-identical serialization of a parsed request structure -- forces an actual serialization round-trip
    rather than relying on Python dict `==`, so the comparison is against
    the payload's wire-level shape, not merely against whichever named keys
    a test happens to check.
    """
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _stub_with_failing_push(base_stub: StubINF) -> httpx.MockTransport:
    """Same stub inference service, except every push call fails -- for proving the
    result file still carries the auto-mode decision block on a failed
    push."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/push"):
            return httpx.Response(500, json={"detail": "push failed"})
        return base_stub.handler(request)

    return httpx.MockTransport(handler)


class TestSuccessfulAutoCloudRecording:
    def test_result_file_and_accounting_record_both_carry_the_decision(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_PAYLOAD_BYTES="262144",
            REPLAY_AUTO_PUSH="true",
            REPLAY_RESULT_FILE=str(result_file),
        )
        stub = StubINF()
        clock = _ModelClock()
        transport = _ModelledTransport(20000, 0.05, clock)
        rc = main.run(
            settings,
            _args(),
            probe_transport=transport,
            probe_clock=clock,
            client_transport=stub.transport,
        )
        assert rc == 0

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"
        assert record["link_probe_decision"]["median_kbps"] == record["link_probe_decision"]["samples_kbps"][0]
        assert record["link_probe_decision"]["failure_reason"] is None
        assert set(record["link_probe_decision"].keys()) == {
            "samples_kbps",
            "median_kbps",
            "scenario",
            "failure_reason",
            "payload_bytes",
            "rtt_ms",
            "ramp_samples_kbps",
        }

        probe_transfers = [t for t in record["transfers"] if t["kind"] == "probe"]
        assert len(probe_transfers) == 1
        assert transport.sizes == [1, 262144, 524288, 1048576]
        assert probe_transfers[0]["bytes"] == sum(transport.sizes)
        assert probe_transfers[0]["crosses_boundary"] is True

        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"] == record["link_probe_decision"]
        assert set(result_payload["link_probe_decision"].keys()) == {
            "samples_kbps",
            "median_kbps",
            "scenario",
            "failure_reason",
            "payload_bytes",
            "rtt_ms",
            "ramp_samples_kbps",
        }

    def test_record_and_log_line_name_the_settled_payload_and_the_round_trip(self, tmp_path, fixtures_dir, caplog):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="2",
            REPLAY_AUTO_PUSH="true",
            REPLAY_RESULT_FILE=str(result_file),
        )
        clock = _ModelClock()
        transport = _ModelledTransport(8000, 0.25, clock)
        with caplog.at_level("INFO", logger="feeder.main"):
            rc = main.run(
                settings,
                _args(),
                probe_transport=transport,
                probe_clock=clock,
                client_transport=StubINF().transport,
            )
        assert rc == 0

        decision = accounting.load_record(settings, "sess-min")["link_probe_decision"]
        assert decision["scenario"] == "cloud"
        assert decision["payload_bytes"] == 1048576
        assert decision["rtt_ms"] == transport.elapsed[0] * 1000.0
        assert len(decision["ramp_samples_kbps"]) == 5
        assert decision["ramp_samples_kbps"] == [pytest.approx(8000, rel=0.02)] * 5
        assert json.loads(result_file.read_text())["link_probe_decision"] == decision
        assert re.search(
            r"auto profile resolved -> cloud \(median=8000\.\d{3} kbps, payload=1048576 bytes, rtt=250 ms\)",
            caplog.text,
        )

    def test_auto_mixed_success_also_leaves_both_records(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="true",
            REPLAY_RESULT_FILE=str(result_file),
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.262144]),
            client_transport=stub.transport,
        )
        assert rc == 0

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "mixed"
        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"]["scenario"] == "mixed"


class TestFailedProbeRecording:
    def test_edge_failure_leaves_accounting_record_with_no_result_file_configured(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        would_be_result_file = tmp_path / "never-written-result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_AUTO_PUSH="false",
            REPLAY_RESULT_FILE=None,
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_PartialFailureTransport(bytes_written=12345),
            probe_clock=_SequenceClock([0.0]),
            client_transport=stub.transport,
        )
        assert rc == 0

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "edge"
        assert record["link_probe_decision"]["failure_reason"] == "connection-refused"

        probe_transfers = [t for t in record["transfers"] if t["kind"] == "probe"]
        assert len(probe_transfers) == 1
        assert probe_transfers[0]["bytes"] == 12345, "must equal the transport's own report, never the requested size"
        assert probe_transfers[0]["bytes"] != settings.probe_payload_bytes

        assert not would_be_result_file.exists(), "no result file may be written when REPLAY_RESULT_FILE is unset"

    def test_stalled_transport_selects_edge_with_reason_timeout_recorded(self, tmp_path, fixtures_dir):
        """A transport stall (the typed ProbeTimeout) is one of the
        two Edge triggers exercised end-to-end here (the other,
        connection-refused, is the case above) -- the decision record must
        carry reason "timeout", never a generic/unlabeled failure.
        """
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_AUTO_PUSH="false",
            REPLAY_RESULT_FILE=str(result_file),
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_StalledTransport(bytes_written=4096),
            probe_clock=_SequenceClock([0.0]),
            client_transport=stub.transport,
        )
        assert rc == 0

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "edge"
        assert record["link_probe_decision"]["failure_reason"] == "timeout"

        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"]["scenario"] == "edge"
        assert result_payload["link_probe_decision"]["failure_reason"] == "timeout"


class TestResultFileReachedRegardlessOfPushOutcome:
    """
    A configured result file must carry the auto-mode decision block on
    every auto run, not only the one branch where a push both attempted
    and succeeded.
    """

    def test_auto_no_push_still_writes_result_file_with_decision(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="true",
            REPLAY_RESULT_FILE=str(result_file),
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(no_push=True),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.0131072]),
            client_transport=stub.transport,
        )
        assert rc == 0
        assert result_file.exists(), "the decision block must reach the result file even with no push attempted"

        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"]["scenario"] == "cloud"
        assert result_payload["pushed"] is False
        assert result_payload["pdm_session_id"] is None
        assert result_payload["report_id"] is None
        assert len(stub.requests_to("/api/v1/sessions/sess-min/push")) == 0

    def test_auto_failed_push_still_writes_result_file_with_decision(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        result_file = tmp_path / "result.json"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="true",
            REPLAY_RESULT_FILE=str(result_file),
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.0131072]),
            client_transport=_stub_with_failing_push(stub),
        )
        assert rc == 2, "a failed push must still fail the run"
        assert result_file.exists(), "the decision block must reach the result file even on a failed push"

        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"]["scenario"] == "cloud"
        assert result_payload["pushed"] is False


class TestPinnedProfileHasNoDecisionBlock:
    def test_pinned_run_records_carry_no_decision_block(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.EDGE,
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()
        rc = main.run(settings, _args(), client_transport=stub.transport)
        assert rc == 0

        record = accounting.load_record(settings, "sess-min")
        assert "link_probe_decision" not in record


class TestSubmissionPayloadDeepEqualToPinnedCounterpart:
    def test_auto_cloud_payload_equals_pinned_cloud_payload(self, tmp_path, fixtures_dir):
        data_root_pinned = _copy_fixture(fixtures_dir, tmp_path / "pinned")
        pinned_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(data_root_pinned),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_AUTO_PUSH="false",
        )
        stub_pinned = StubINF()
        assert main.run(pinned_settings, _args(), client_transport=stub_pinned.transport) == 0
        pinned_req = stub_pinned.requests_to("/api/v1/sessions")[0]

        data_root_auto = _copy_fixture(fixtures_dir, tmp_path / "auto")
        auto_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root_auto),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
        )
        stub_auto = StubINF()
        assert (
            main.run(
                auto_settings,
                _args(),
                probe_transport=_SuccessTransport(),
                probe_clock=_SequenceClock([0.0, 0.0131072]),
                client_transport=stub_auto.transport,
            )
            == 0
        )
        auto_req = stub_auto.requests_to("/api/v1/sessions")[0]

        assert _canonical_bytes(auto_req.form_fields) == _canonical_bytes(pinned_req.form_fields)
        assert _canonical_bytes({name: content.hex() for name, content in auto_req.files.items()}) == _canonical_bytes(
            {name: content.hex() for name, content in pinned_req.files.items()}
        )

    def test_auto_mixed_payload_equals_pinned_mixed_payload(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        pinned_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.MIXED,
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_AUTO_PUSH="false",
        )
        stub_pinned = StubINF()
        assert main.run(pinned_settings, _args(), client_transport=stub_pinned.transport) == 0
        pinned_req = stub_pinned.requests_to("/api/v1/sessions/path")[0]

        auto_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
        )
        stub_auto = StubINF()
        assert (
            main.run(
                auto_settings,
                _args(),
                probe_transport=_SuccessTransport(),
                probe_clock=_SequenceClock([0.0, 0.262144]),
                client_transport=stub_auto.transport,
            )
            == 0
        )
        auto_req = stub_auto.requests_to("/api/v1/sessions/path")[0]

        assert _canonical_bytes(auto_req.json_body) == _canonical_bytes(pinned_req.json_body)

    def test_auto_edge_payload_equals_pinned_edge_payload(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        pinned_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.EDGE,
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_AUTO_PUSH="false",
        )
        stub_pinned = StubINF()
        assert main.run(pinned_settings, _args(), client_transport=stub_pinned.transport) == 0
        pinned_req = stub_pinned.requests_to("/api/v1/sessions/path")[0]

        auto_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_AUTO_PUSH="false",
        )
        stub_auto = StubINF()
        assert (
            main.run(
                auto_settings,
                _args(),
                probe_transport=_PartialFailureTransport(bytes_written=0),
                probe_clock=_SequenceClock([0.0]),
                client_transport=stub_auto.transport,
            )
            == 0
        )
        auto_req = stub_auto.requests_to("/api/v1/sessions/path")[0]

        assert _canonical_bytes(auto_req.json_body) == _canonical_bytes(pinned_req.json_body)
