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

"""DeploymentProfile.AUTO wiring in
app/config.py + app/main.py -- the dual target, the fail-closed startup
refusal, ingestion_mode's refusal for unresolved AUTO, and run's
exactly-once resolution ordered before any submission-side call.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from app import accounting, main
from app.config import DeploymentProfile, IngestionMode, Settings
from tests.conftest import make_settings
from tests.stub_inf import StubINF


def _args(session_id=None, no_push=False, dry_run=False) -> argparse.Namespace:
    return argparse.Namespace(session_id=session_id, no_push=no_push, dry_run=dry_run)


def _copy_fixture(fixtures_dir: Path, dest: Path) -> Path:
    shutil.copytree(fixtures_dir / "minimal_conforming", dest)
    return dest


def _copy_empty_images_fixture(fixtures_dir: Path, dest: Path) -> Path:
    """
    A conforming session whose one manifest-referenced file is an
    unsupported format -- ``load_replay_session`` succeeds but
    ``session.images`` ends up empty, exercising main.py's own
    zero-images return without a manifest-validation error.
    """
    _copy_fixture(fixtures_dir, dest)
    session_dir = dest / "sessions" / "sess-min"
    images_dir = session_dir / "images"
    (images_dir / "a.jpg").unlink()
    (images_dir / "a.bin").write_bytes(b"not-an-image")
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"] = [{"filename": "a.bin"}]
    manifest_path.write_text(json.dumps(manifest))
    return dest


class _SequenceClock:
    def __init__(self, timestamps: List[float]) -> None:
        self._it = iter(timestamps)

    def __call__(self) -> float:
        return next(self._it)


class _SuccessTransport:
    def __init__(self) -> None:
        self.call_count = 0
        self.closed = False

    def upload(self, url, headers, payload, timeout):
        from app.link_probe import TransportSample

        self.call_count += 1
        return TransportSample(bytes_written=len(payload))

    def close(self) -> None:
        self.closed = True


class _FailingTransport:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.closed = False

    def upload(self, url, headers, payload, timeout):
        raise self._exc

    def close(self) -> None:
        self.closed = True


class TestAutoConfigValidation:
    def test_auto_without_cloud_url_refused_at_construction(self):
        with pytest.raises(ValidationError, match="REPLAY_CLOUD_INFERENCE_BASE_URL"):
            make_settings(REPLAY_DEPLOYMENT_PROFILE="auto", REPLAY_CLOUD_INFERENCE_BASE_URL=None)

    def test_auto_with_cloud_url_constructs_fine(self):
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto", REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test"
        )
        assert settings.deployment_profile is DeploymentProfile.AUTO

    def test_ingestion_mode_raises_for_unresolved_auto(self):
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto", REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test"
        )
        with pytest.raises(ValueError, match="AUTO must be resolved"):
            _ = settings.ingestion_mode


class TestAutoRunWiring:
    @pytest.mark.parametrize(
        "profile,expected_endpoint",
        [
            (DeploymentProfile.CLOUD, "/api/v1/sessions"),
            (DeploymentProfile.MIXED, "/api/v1/sessions/path"),
            (DeploymentProfile.EDGE, "/api/v1/sessions/path"),
            (DeploymentProfile.X86_CI, "/api/v1/sessions"),
        ],
    )
    def test_pinned_profile_makes_no_probe_calls_and_no_resolution(
        self, tmp_path, fixtures_dir, profile, expected_endpoint
    ):
        data_root = _copy_fixture(fixtures_dir, tmp_path / f"store-{profile.value}")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=profile,
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()

        class _FailIfCalledTransport:
            def upload(self, *a, **kw):
                raise AssertionError("probe transport must not be called for a pinned profile")

            def close(self) -> None:
                pass

        resolve_calls: List[str] = []
        real_resolve = main._resolve_effective_profile

        def _tracking_resolve(settings_, probe_transport, clock):
            resolve_calls.append("resolve")
            return real_resolve(settings_, probe_transport, clock)

        with patch("app.main._resolve_effective_profile", side_effect=_tracking_resolve):
            rc = main.run(
                settings, _args(), probe_transport=_FailIfCalledTransport(), client_transport=stub.transport
            )

        assert rc == 0
        assert resolve_calls == [], "a pinned profile must trigger zero resolution calls"
        assert len(stub.requests_to(expected_endpoint)) == 1
        record = accounting.load_record(settings, "sess-min")
        assert record is not None
        assert "link_probe_decision" not in record

    def test_resolution_happens_exactly_once_before_first_submission_call(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
        )
        call_log: List[str] = []
        stub = StubINF()

        def _tracking_handler(request):
            call_log.append(f"submission:{request.method}:{request.url.path}")
            return stub.handler(request)

        transport = httpx.MockTransport(_tracking_handler)

        real_resolve = main._resolve_effective_profile

        def _tracking_resolve(settings_, probe_transport, clock):
            call_log.append("resolve")
            return real_resolve(settings_, probe_transport, clock)

        probe_transport = _SuccessTransport()
        clock = _SequenceClock([0.0, 0.0131072])

        with patch("app.main._resolve_effective_profile", side_effect=_tracking_resolve):
            rc = main.run(
                settings, _args(), probe_transport=probe_transport, probe_clock=clock, client_transport=transport
            )

        assert rc == 0
        assert call_log.count("resolve") == 1
        assert call_log[0] == "resolve", "resolution must happen before any submission-side transport call"
        assert any(entry.startswith("submission:") for entry in call_log[1:])
        assert probe_transport.call_count == 1, "the probe transport itself must be invoked exactly once too"

    def test_auto_resolved_cloud_submits_to_cloud_target_same_sequence_as_pinned_cloud(self, tmp_path, fixtures_dir):
        data_root_pinned = _copy_fixture(fixtures_dir, tmp_path / "pinned")
        pinned_settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(data_root_pinned),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_AUTO_PUSH="false",
        )
        stub_pinned = StubINF()
        rc_pinned = main.run(pinned_settings, _args(), client_transport=stub_pinned.transport)
        assert rc_pinned == 0
        pinned_sequence = [(r.method, r.path) for r in stub_pinned.requests]

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
        rc_auto = main.run(
            auto_settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.0131072]),
            client_transport=stub_auto.transport,
        )
        assert rc_auto == 0

        auto_sequence = [(r.method, r.path) for r in stub_auto.requests]
        assert auto_sequence == pinned_sequence
        assert all(r.host == "cloud.test" for r in stub_auto.requests)

        record = accounting.load_record(auto_settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"

    def test_auto_resolved_mixed_submits_to_local_target_in_path_mode(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
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
        assert len(stub.requests_to("/api/v1/sessions/path")) == 1
        assert all(r.host == "local.test" for r in stub.requests)
        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "mixed"

    def test_auto_resolved_gprs_class_link_selects_mixed_not_edge(self, tmp_path, fixtures_dir):
        """A GPRS-class (~40 kbps) measured link selects Mixed, never Edge."""
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 6.5536]),
            client_transport=stub.transport,
        )
        assert rc == 0
        assert len(stub.requests_to("/api/v1/sessions/path")) == 1
        assert all(r.host == "local.test" for r in stub.requests)
        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "mixed"
        assert record["link_probe_decision"]["median_kbps"] == pytest.approx(40.0, rel=1e-6)
        assert record["link_probe_decision"]["samples_kbps"] == pytest.approx([40.0], rel=1e-6)

    def test_auto_resolved_edge_on_probe_failure_submits_to_local_target_in_path_mode(self, tmp_path, fixtures_dir):
        from app import link_probe

        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_FailingTransport(link_probe.ProbeConnectionRefused("refused")),
            probe_clock=_SequenceClock([0.0]),
            client_transport=stub.transport,
        )
        assert rc == 0
        assert len(stub.requests_to("/api/v1/sessions/path")) == 1
        assert all(r.host == "local.test" for r in stub.requests)
        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "edge"
        assert record["link_probe_decision"]["failure_reason"] == "connection-refused"

    def test_auto_resolved_edge_on_stall_submits_to_local_target_in_path_mode(self, tmp_path, fixtures_dir):
        """A transport STALL (not merely connection-refused) also
        resolves the session to edge with reason "timeout" recorded."""
        from app import link_probe

        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()
        rc = main.run(
            settings,
            _args(),
            probe_transport=_FailingTransport(link_probe.ProbeTimeout("timed out")),
            probe_clock=_SequenceClock([0.0]),
            client_transport=stub.transport,
        )
        assert rc == 0
        assert len(stub.requests_to("/api/v1/sessions/path")) == 1
        assert all(r.host == "local.test" for r in stub.requests)
        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "edge"
        assert record["link_probe_decision"]["failure_reason"] == "timeout"


class TestDecisionRecordSurvivesEveryLaterExit:
    def test_auto_dry_run_prints_resolved_mode_and_leaves_the_decision_record(self, tmp_path, fixtures_dir, caplog):
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
            REPLAY_RESULT_FILE=str(result_file),
        )
        with caplog.at_level("INFO"):
            rc = main.run(
                settings,
                _args(dry_run=True),
                probe_transport=_SuccessTransport(),
                probe_clock=_SequenceClock([0.0, 0.0131072]),
            )
        assert rc == 0, "dry-run on a resolved AUTO profile must not crash"
        assert any("DRY RUN" in r.message and "http_upload" in r.message for r in caplog.records), (
            "dry-run must print the RESOLVED mode, not the unresolved auto value"
        )

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"
        result_payload = json.loads(result_file.read_text())
        assert result_payload["link_probe_decision"]["scenario"] == "cloud"

    def test_auto_empty_images_exit_still_leaves_the_decision_record(self, tmp_path, fixtures_dir):
        data_root = _copy_empty_images_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
        )
        rc = main.run(
            settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.0131072]),
        )
        assert rc == 1, "zero submittable images must still fail the run"

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"

    def test_auto_unreachable_inf_still_leaves_the_decision_record(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
        )
        stub = StubINF(healthy=False)
        with patch("app.client.time.sleep"):
            rc = main.run(
                settings,
                _args(),
                probe_transport=_SuccessTransport(),
                probe_clock=_SequenceClock([0.0, 0.0131072]),
                client_transport=stub.transport,
            )
        assert rc == 1, "an unreachable InferenceService must still fail the run"

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"

    def test_auto_failed_submission_still_leaves_the_decision_record(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
        )
        base_stub = StubINF()

        def _failing_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/api/v1/sessions":
                return httpx.Response(500, json={"detail": "submission failed"})
            return base_stub.handler(request)

        rc = main.run(
            settings,
            _args(),
            probe_transport=_SuccessTransport(),
            probe_clock=_SequenceClock([0.0, 0.0131072]),
            client_transport=httpx.MockTransport(_failing_handler),
        )
        assert rc == 2, "a failed submission must still fail the run"

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud"

    def test_auto_marker_write_failure_still_leaves_the_decision_record(self, tmp_path, fixtures_dir):
        data_root = _copy_fixture(fixtures_dir, tmp_path / "store")
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE="auto",
            REPLAY_DATA_ROOT=str(data_root),
            REPLAY_SESSION_ID="sess-min",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud.test",
            REPLAY_INFERENCE_BASE_URL="http://local.test",
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_AUTO_PUSH="false",
        )
        stub = StubINF()

        with patch("app.main.record_replayed", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                main.run(
                    settings,
                    _args(),
                    probe_transport=_SuccessTransport(),
                    probe_clock=_SequenceClock([0.0, 0.0131072]),
                    client_transport=stub.transport,
                )

        record = accounting.load_record(settings, "sess-min")
        assert record["link_probe_decision"]["scenario"] == "cloud", (
            "the decision record must exist even though the marker write below it raised"
        )
