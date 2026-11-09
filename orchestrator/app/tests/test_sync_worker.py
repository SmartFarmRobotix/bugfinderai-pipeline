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
Sync behaviour: a sync round-trip lands the report in the cloud Reporting service and marks it synced
locally; a sync run with a placeholder credential refuses to upload and reports the misconfiguration.

Uses a real `http.server.HTTPServer` Gatekeeper and Reporting service stand-in (mirrors
test_intake_auth.py's `gatekeeper_network_standin` idiom) rather than
mocking `requests` -- a genuine listening socket is a stronger guarantee
that the worker's HTTP calls are shaped the way a real GateKeeper proxy
would require (method, path, auth header, multipart body). The stand-in also records
each ingest call's raw body (`last_upload_bodies`) so the tests can prove the
uploaded bytes are the sync copy's, not the local report's, and can be
told to answer specific calls with a given HTTP status
(`status_by_call`) to exercise the 413-naming paths.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import logging
import threading
from pathlib import Path
from typing import List

import pytest

import sync_queue
import sync_worker
from cloud_intake_client import (
    PairPushConnectionError,
    PairPushHTTPError,
    PairPushResult,
    ReassessConnectionError,
    ReassessHTTPError,
    ReassessResult,
)
from contract import CONTRACT_VERSION, DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput, SessionSubmission
from models import ReportIndexRecord, SessionRecord
from service import intake_submission


class _FakeGatekeeperRSHandler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeGKRS/1.0"

    def log_message(self, format, *args):  # noqa: A002 - silence test noise
        pass

    def do_POST(self):  # noqa: N802 - http.server naming
        self.server.hit_paths.append(self.path)  # type: ignore[attr-defined]
        if self.path == "/api/login/":
            self._send_json(200, {"access": "fake-sync-token"})
            return
        if self.path.startswith("/api/proxy/reporting/api/v1/openagri-report/pest-session-report/edge-sync/"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            self.server.last_upload_bodies.append(body)  # type: ignore[attr-defined]
            if self.server.status_by_call:  # type: ignore[attr-defined]
                status = self.server.status_by_call.pop(0)  # type: ignore[attr-defined]
                self._send_text(status, "request entity too large" if status == 413 else "error")
                return
            report_uuid = self.server.report_uuid_by_call.pop(0) if self.server.report_uuid_by_call else "uuid-x"  # type: ignore[attr-defined]
            self._send_json(200, {"uuid": report_uuid, "deduplicated": False})
            return
        self._send_json(404, {"detail": "not found"})

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status: int, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def gk_rs_standin():
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeGatekeeperRSHandler)
    server.hit_paths: List[str] = []  # type: ignore[attr-defined]
    server.report_uuid_by_call: List[str] = []  # type: ignore[attr-defined]
    server.status_by_call: List[int] = []  # type: ignore[attr-defined]
    server.last_upload_bodies: List[bytes] = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


class _SyncSettings:
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 5
    rs_service_name = "reporting"
    rs_edge_report_ingest_path = "/api/v1/openagri-report/pest-session-report/edge-sync/"
    sync_call_timeout_seconds = 5
    sync_max_transient_failures = 5
    sync_reassess_enabled = False

    def __init__(self, base_url: str, queue_dir: str, username="sync-agent", password="sync-test-secret"):
        self.gatekeeper_base_url = base_url
        self.sync_queue_dir = queue_dir
        self.sync_gatekeeper_username = username
        self.sync_gatekeeper_password = password


def _write_pdf(path, content: bytes) -> str:
    path.write_bytes(content)
    return str(path)


class TestSyncRoundTrip:
    def test_pending_entry_uploads_and_marks_synced(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.report_uuid_by_call.append("cloud-uuid-1")  # type: ignore[attr-defined]

        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 fake sync copy")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 fake local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == ["sess-1"]
        assert result.failed == []

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "cloud-uuid-1"

        assert (queue_dir / ".synced" / "sess-1.json").exists()

        assert "/api/login/" in gk_rs_standin.hit_paths  # type: ignore[attr-defined]
        assert any(
            p.startswith("/api/proxy/reporting/") for p in gk_rs_standin.hit_paths  # type: ignore[attr-defined]
        )

    def test_second_run_over_already_synced_queue_uploads_nothing(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.report_uuid_by_call.append("cloud-uuid-1")  # type: ignore[attr-defined]

        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 fake sync copy")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 fake local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        sync_worker.run_sync_once(db, settings)
        gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]

        result = sync_worker.run_sync_once(db, settings)
        assert result.uploaded == []
        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined]


class TestUploadedBytesAreTheSyncCopys:
    def test_uploaded_body_is_the_sync_copy_not_the_local_report(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.report_uuid_by_call.append("cloud-uuid-1")  # type: ignore[attr-defined]

        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 CAPPED-SYNC-COPY-BYTES")
        local_pdf = _write_pdf(
            tmp_path / "sess-1.pdf", b"%PDF-1.4 FULL-RESOLUTION-LOCAL-REPORT-BYTES-MUCH-LONGER-THAN-THE-SYNC-COPY"
        )
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == ["sess-1"]
        uploaded_body = gk_rs_standin.last_upload_bodies[-1]  # type: ignore[attr-defined]
        assert b"CAPPED-SYNC-COPY-BYTES" in uploaded_body
        assert b"FULL-RESOLUTION-LOCAL-REPORT-BYTES" not in uploaded_body


class TestBodyLimitFailureAndRetrySkip:
    def test_413_is_named_in_the_failure_reason_and_records_the_failed_hash(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.append(413)  # type: ignore[attr-defined]

        sync_pdf_content = b"%PDF-1.4 sync copy"
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", sync_pdf_content)
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == []
        assert result.failed == [("sess-1", "refused by the Gatekeeper's body limit (413)")]

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.FAILED
        assert entry.failure_reason == "refused by the Gatekeeper's body limit (413)"
        assert entry.failed_hash == hashlib.sha256(sync_pdf_content).hexdigest()

    def test_a_second_run_with_the_same_file_is_skipped_not_retried(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.append(413)  # type: ignore[attr-defined]

        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 sync copy")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        sync_worker.run_sync_once(db, settings)
        gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]

        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == []
        assert result.failed == []
        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined] -- no HTTP call at all

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.FAILED
        assert entry.failure_reason == "refused by the Gatekeeper's body limit (413)"

    def test_a_changed_file_is_retried(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.append(413)  # type: ignore[attr-defined]
        gk_rs_standin.report_uuid_by_call.append("cloud-uuid-2")  # type: ignore[attr-defined]

        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 sync copy")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings(base_url, str(queue_dir))
        sync_worker.run_sync_once(db, settings)

        Path(sync_pdf).write_bytes(b"%PDF-1.4 sync copy, re-rendered")

        gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]
        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == ["sess-1"]
        assert any(p.startswith("/api/proxy/reporting/") for p in gk_rs_standin.hit_paths)  # type: ignore[attr-defined]

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "cloud-uuid-2"
        assert entry.content_hash == hashlib.sha256(b"%PDF-1.4 sync copy, re-rendered").hexdigest()


def _queue_legacy_failed_entry(queue_dir, session_id: str, pdf_path: str, reason: str = "500") -> None:
    """A `failed` manifest in the older format: a failed hash,
    no failure count and no failure kind."""
    sync_queue.enqueue(queue_dir, session_id, pdf_path, f"hash-{session_id}")
    sync_queue.mark_failed(
        queue_dir, session_id, reason, failed_hash=sync_worker._file_sha256(pdf_path)
    )
    manifest = Path(queue_dir) / f"{session_id}.manifest.json"
    data = json.loads(manifest.read_text())
    del data["failure_count"], data["failure_kind"]
    manifest.write_text(json.dumps(data))


def _add_index_row(db, session_id: str) -> None:
    db.add(
        ReportIndexRecord(
            session_id=session_id, report_id=None, content_hash="hash", origin="edge-local", state="pending"
        )
    )
    db.commit()


class TestEntriesThatCannotSyncAreRetired:
    def test_an_oversized_failed_entry_is_retired_before_any_request(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        big_pdf = _write_pdf(tmp_path / "sess-big.pdf", b"%PDF-1.4 small at enqueue")
        queue_dir = tmp_path / "queue"
        _queue_legacy_failed_entry(queue_dir, "sess-big", big_pdf, reason="500 {\"error\": \"internal\"}")
        Path(big_pdf).write_bytes(b"%PDF-1.4 " + b"\0" * 2_499_991)
        _add_index_row(db, "sess-big")

        result = sync_worker.run_sync_once(db, _SyncSettings(base_url, str(queue_dir)))

        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined] -- no request of any kind
        assert result.uploaded == [] and result.failed == []
        assert [sid for sid, _ in result.retired] == ["sess-big"]
        reason = result.retired[0][1]
        assert "2500000 bytes" in reason and "2000000" in reason
        assert sync_queue.list_entries(queue_dir) == []
        (retired,) = sync_queue.list_retired(queue_dir)
        assert retired.state == sync_queue.RETIRED
        assert retired.failure_reason == reason
        assert Path(big_pdf).exists()
        record = db.query(ReportIndexRecord).filter_by(session_id="sess-big").one()
        assert record.state == "retired"

    def test_a_failed_entry_whose_file_is_gone_is_retired_before_any_request(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        pdf = _write_pdf(tmp_path / "sess-gone.pdf", b"%PDF-1.4 small")
        queue_dir = tmp_path / "queue"
        _queue_legacy_failed_entry(queue_dir, "sess-gone", pdf)
        Path(pdf).unlink()

        result = sync_worker.run_sync_once(db, _SyncSettings(base_url, str(queue_dir)))

        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined]
        assert [sid for sid, _ in result.retired] == ["sess-gone"]
        assert pdf in result.retired[0][1]
        (retired,) = sync_queue.list_retired(queue_dir)
        assert pdf in retired.failure_reason

    def test_the_index_row_of_a_retired_session_reads_retired_and_others_are_untouched(
        self, gk_rs_standin, tmp_path, db
    ):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        queue_dir = tmp_path / "queue"
        gone = _write_pdf(tmp_path / "sess-gone.pdf", b"%PDF-1.4 small")
        sync_queue.enqueue(queue_dir, "sess-gone", gone, "hash-gone")
        Path(gone).unlink()
        sync_queue.enqueue(queue_dir, "sess-ok", _write_pdf(tmp_path / "sess-ok.pdf", b"%PDF-1.4 ok"), "hash-ok")
        _add_index_row(db, "sess-gone")
        _add_index_row(db, "sess-ok")

        sync_worker.run_sync_once(db, _SyncSettings(base_url, str(queue_dir)))

        states = {r.session_id: r.state for r in db.query(ReportIndexRecord).all()}
        assert states == {"sess-gone": "retired", "sess-ok": "synced"}


class TestFailureClasses:
    def _queue_one(self, tmp_path):
        pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 sync copy")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", pdf, "hash-1")
        return queue_dir

    def test_a_500_on_every_run_is_retried_and_retired_at_the_maximum(self, gk_rs_standin, tmp_path, db):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.extend([500, 500, 500])  # type: ignore[attr-defined]
        queue_dir = self._queue_one(tmp_path)
        settings = _SyncSettings(base_url, str(queue_dir))
        settings.sync_max_transient_failures = 3
        _add_index_row(db, "sess-1")

        counts = []
        results = []
        for _ in range(3):
            gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]
            results.append(sync_worker.run_sync_once(db, settings))
            assert any(p.startswith("/api/proxy/reporting/") for p in gk_rs_standin.hit_paths)  # type: ignore[attr-defined]
            if results[-1].retired:
                counts.append(sync_queue.list_retired(queue_dir)[0].failure_count)
            else:
                counts.append(sync_queue.read_entry(queue_dir, "sess-1").failure_count)

        assert counts == [1, 2, 3]
        assert [len(r.retired) for r in results] == [0, 0, 1]
        assert all(len(r.failed) == 1 for r in results)
        (retired,) = sync_queue.list_retired(queue_dir)
        assert "500" in retired.failure_reason
        assert retired.failure_reason == results[2].failed[0][1]
        assert sync_queue.list_entries(queue_dir) == []
        assert db.query(ReportIndexRecord).filter_by(session_id="sess-1").one().state == "retired"

        gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]
        sync_worker.run_sync_once(db, settings)
        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined] -- nothing left to attempt

    @pytest.mark.parametrize("status", [413, 404])
    def test_a_rejection_is_not_retried_while_the_file_is_unchanged(self, gk_rs_standin, tmp_path, db, status):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.append(status)  # type: ignore[attr-defined]
        queue_dir = self._queue_one(tmp_path)
        settings = _SyncSettings(base_url, str(queue_dir))

        sync_worker.run_sync_once(db, settings)
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.failure_kind == (
            sync_queue.FAILURE_BODY_LIMIT if status == 413 else sync_queue.FAILURE_REJECTED
        )

        gk_rs_standin.hit_paths.clear()  # type: ignore[attr-defined]
        second = sync_worker.run_sync_once(db, settings)

        assert gk_rs_standin.hit_paths == []  # type: ignore[attr-defined]
        assert second.failed == [] and second.retired == []
        assert sync_queue.read_entry(queue_dir, "sess-1").state == sync_queue.FAILED

    def test_a_transient_failure_is_retried_with_an_unchanged_file_and_a_500_then_a_200_syncs(
        self, gk_rs_standin, tmp_path, db
    ):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.status_by_call.append(500)  # type: ignore[attr-defined]
        gk_rs_standin.report_uuid_by_call.append("cloud-uuid-9")  # type: ignore[attr-defined]
        queue_dir = self._queue_one(tmp_path)
        settings = _SyncSettings(base_url, str(queue_dir))

        sync_worker.run_sync_once(db, settings)
        failed = sync_queue.read_entry(queue_dir, "sess-1")
        assert (failed.failure_kind, failed.failure_count) == (sync_queue.FAILURE_TRANSIENT, 1)

        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == ["sess-1"]
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.failure_count == 0
        assert entry.failure_kind is None

    def test_an_unreachable_service_is_transient(self, tmp_path, db, monkeypatch):
        from reporting_sync_client import ReportSyncConnectionError

        def _unreachable(*args, **kwargs):
            raise ReportSyncConnectionError("ReportingService/GateKeeper unreachable")

        monkeypatch.setattr(sync_worker, "push_synced_report", _unreachable)
        queue_dir = self._queue_one(tmp_path)
        settings = _SyncSettings("http://gatekeeper.invalid", str(queue_dir))

        result = sync_worker.run_sync_once(db, settings)

        assert len(result.failed) == 1
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.failure_kind, entry.failure_count) == (sync_queue.FAILURE_TRANSIENT, 1)

    def test_a_malformed_success_answer_is_transient(self, tmp_path):
        from reporting_sync_client import ReportSyncHTTPError

        assert sync_worker._failure_kind(ReportSyncHTTPError(200, "non-JSON response")) == sync_queue.FAILURE_TRANSIENT
        assert sync_worker._failure_kind(ReportSyncHTTPError(200, "response missing 'uuid'")) == sync_queue.FAILURE_TRANSIENT
        assert sync_worker._failure_kind(ReportSyncHTTPError(503, "x")) == sync_queue.FAILURE_TRANSIENT
        assert sync_worker._failure_kind(ReportSyncHTTPError(422, "x")) == sync_queue.FAILURE_REJECTED
        assert sync_worker._failure_kind(ReportSyncHTTPError(413, "x")) == sync_queue.FAILURE_BODY_LIMIT


class TestSyncCredentialRejection:
    @pytest.mark.parametrize("placeholder", ["change-me", "some-key", "dev_jwt_signing_key"])
    def test_placeholder_password_refuses_to_sync(self, tmp_path, placeholder):
        settings = _SyncSettings("http://127.0.0.1:1", str(tmp_path), password=placeholder)
        with pytest.raises(sync_worker.SyncCredentialError):
            sync_worker.validate_sync_credentials(settings)

    @pytest.mark.parametrize("placeholder", ["change-me", "some-key"])
    def test_placeholder_username_refuses_to_sync(self, tmp_path, placeholder):
        settings = _SyncSettings("http://127.0.0.1:1", str(tmp_path), username=placeholder)
        with pytest.raises(sync_worker.SyncCredentialError):
            sync_worker.validate_sync_credentials(settings)

    def test_missing_credentials_refuses_to_sync(self, tmp_path):
        settings = _SyncSettings("http://127.0.0.1:1", str(tmp_path), username=None, password=None)
        with pytest.raises(sync_worker.SyncCredentialError):
            sync_worker.validate_sync_credentials(settings)

    def test_run_sync_once_never_attempts_upload_with_placeholder_credential(self, tmp_path, db):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 sync copy")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", b"%PDF-1.4 local report")
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")

        settings = _SyncSettings("http://127.0.0.1:1", str(queue_dir), password="change-me")
        with pytest.raises(sync_worker.SyncCredentialError):
            sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.PENDING


def _submission(session_id: str) -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=0.6,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    envelope = ReportingEnvelope(
        session_id=session_id,
        best_image_id="img-1",
        image_count=2,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status="designated",
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
        ),
    )
    return SessionSubmission(contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope)


_REPORTED = ReassessResult(status="accepted", state="reported", report_id="cloud-report-1")
_EMBEDDED = PairPushResult(stored=True, images_embedded=True)


class _CloudFakes:
    """The intake and pair calls, each answering from a queue (an exception
    instance is raised; the last answer repeats), all logged in one list."""

    def __init__(self, monkeypatch, intake_answers=None, pair_answers=None):
        self.calls: List[tuple] = []
        self._intake = list(intake_answers if intake_answers is not None else [_REPORTED])
        self._pair = list(pair_answers if pair_answers is not None else [_EMBEDDED])
        monkeypatch.setattr(sync_worker, "submit_for_reassessment", self._submit)
        monkeypatch.setattr(sync_worker, "push_image_pair", self._push)

    @staticmethod
    def _next(queue):
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _submit(self, settings, session_id, submission_json, edge_report_id):
        self.calls.append(("intake", session_id, submission_json, edge_report_id))
        return self._next(self._intake)

    def _push(self, settings, session_id, pair):
        self.calls.append(("pair", session_id, pair))
        return self._next(self._pair)

    def names(self) -> List[str]:
        return [call[0] for call in self.calls]


class _ReassessSettings(_SyncSettings):
    sync_reassess_enabled = True


_PAIR_BODY = {"contract_version": CONTRACT_VERSION, "image_id": "img-1", "artifact_hash": "h"}


class TestReassessmentStep:
    @staticmethod
    def _setup(tmp_path, db, gk_rs_standin, with_pair=True, with_record=True, limit=5, index_row=True):
        base_url = f"http://127.0.0.1:{gk_rs_standin.server_address[1]}"
        gk_rs_standin.report_uuid_by_call.append("edge-report-uuid")
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf", b"%PDF-1.4 fake sync copy")
        pair_path = None
        if with_pair:
            pair_file = tmp_path / "sess-1.sync-pair.json"
            pair_file.write_text(json.dumps(_PAIR_BODY))
            pair_path = str(pair_file)
        queue_dir = tmp_path / "queue"
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", pair_path=pair_path)
        if with_record:
            intake_submission(db, _submission("sess-1"))
        if index_row:
            _add_index_row(db, "sess-1")
        settings = _ReassessSettings(base_url, str(queue_dir))
        settings.sync_max_transient_failures = limit
        return settings, queue_dir

    @staticmethod
    def _assert_synced_as_without_the_step(db, queue_dir):
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "edge-report-uuid"
        assert (queue_dir / ".synced" / "sess-1.json").exists()
        row = db.query(ReportIndexRecord).filter_by(session_id="sess-1").one()
        assert (row.state, row.report_id) == ("synced", "edge-report-uuid")

    def test_intake_then_pair_in_order_and_recorded_done(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        fakes = _CloudFakes(monkeypatch)

        result = sync_worker.run_sync_once(db, settings)

        assert fakes.names() == ["intake", "pair"]
        _, _, submission, edge_report_id = fakes.calls[0]
        record = db.query(SessionRecord).filter_by(session_id="sess-1").one()
        assert edge_report_id == "edge-report-uuid"
        assert submission == {
            "contract_version": record.contract_version,
            "risk_input": json.loads(record.risk_input_json),
            "reporting_envelope": json.loads(record.reporting_envelope_json),
        }
        assert fakes.calls[1][2] == _PAIR_BODY
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_DONE
        assert entry.cloud_reassessed_report_id == "cloud-report-1"
        assert result.uploaded == ["sess-1"]
        assert result.reassessed == ["sess-1"]
        self._assert_synced_as_without_the_step(db, queue_dir)

    def test_three_unreachable_runs_end_failed_and_the_copy_is_uploaded_once(
        self, monkeypatch, gk_rs_standin, tmp_path, db
    ):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin, limit=3)
        fakes = _CloudFakes(monkeypatch, intake_answers=[ReassessConnectionError("unreachable")])

        sync_worker.run_sync_once(db, settings)
        self._assert_synced_as_without_the_step(db, queue_dir)
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_PENDING, 1)

        sync_worker.run_sync_once(db, settings)
        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_FAILED, 3)
        assert "unreachable" in entry.reassess_reason
        assert fakes.names() == ["intake"] * 3
        assert len(gk_rs_standin.last_upload_bodies) == 1
        self._assert_synced_as_without_the_step(db, queue_dir)

        sync_worker.run_sync_once(db, settings)
        assert fakes.names() == ["intake"] * 3

    def test_duplicate_reported_answer_is_success(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(
            monkeypatch, intake_answers=[ReassessResult(status="duplicate", state="reported", report_id="cloud-report-7")]
        )

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.cloud_reassessed_report_id) == (sync_queue.REASSESS_DONE, "cloud-report-7")

    def test_conflict_is_final_after_one_attempt_with_the_detail(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        body = json.dumps({"detail": "session_id 'sess-1' already submitted with different content"})
        fakes = _CloudFakes(monkeypatch, intake_answers=[ReassessHTTPError(409, body)])

        result = sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_FAILED
        assert "already submitted with different content" in entry.reassess_reason
        assert entry.reassess_attempts == 0
        assert fakes.names() == ["intake"]
        assert [sid for sid, _ in result.reassess_failed] == ["sess-1"]
        self._assert_synced_as_without_the_step(db, queue_dir)

    @pytest.mark.parametrize("status", [401, 403, 500, 502])
    def test_auth_and_server_errors_count_an_attempt(self, monkeypatch, gk_rs_standin, tmp_path, db, status):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(monkeypatch, intake_answers=[ReassessHTTPError(status, "no")])

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_PENDING, 1)

    def test_another_client_error_is_final(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(monkeypatch, intake_answers=[ReassessHTTPError(422, "unprocessable")])

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_FAILED
        assert "422" in entry.reassess_reason and "unprocessable" in entry.reassess_reason

    def test_accepted_without_a_report_stays_open_and_a_later_duplicate_completes_it(
        self, monkeypatch, gk_rs_standin, tmp_path, db
    ):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        fakes = _CloudFakes(
            monkeypatch,
            intake_answers=[
                ReassessResult(status="accepted", state="assessed", report_id=None),
                ReassessResult(status="duplicate", state="reported", report_id="cloud-report-2"),
            ],
        )

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_PENDING, 1)
        assert entry.cloud_reassessed_report_id is None
        assert fakes.names() == ["intake"]
        self._assert_synced_as_without_the_step(db, queue_dir)

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.cloud_reassessed_report_id) == (sync_queue.REASSESS_DONE, "cloud-report-2")
        assert fakes.names() == ["intake", "intake", "pair"]

    def test_accepted_reported_without_an_id_is_not_success(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        fakes = _CloudFakes(
            monkeypatch, intake_answers=[ReassessResult(status="accepted", state="reported", report_id=None)]
        )

        sync_worker.run_sync_once(db, settings)

        assert sync_queue.read_entry(queue_dir, "sess-1").reassess_state == sync_queue.REASSESS_PENDING
        assert fakes.names() == ["intake"]

    @pytest.mark.parametrize(
        "unconfirmed",
        [PairPushResult(stored=True, images_embedded=False), PairPushResult(stored=False, images_embedded=False)],
    )
    def test_an_unconfirmed_pair_stays_pending_and_is_pushed_again(
        self, monkeypatch, gk_rs_standin, tmp_path, db, unconfirmed
    ):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        fakes = _CloudFakes(monkeypatch, pair_answers=[unconfirmed, unconfirmed, _EMBEDDED])

        sync_worker.run_sync_once(db, settings)
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_SUBMITTED
        assert entry.cloud_reassessed_report_id == "cloud-report-1"

        sync_worker.run_sync_once(db, settings)
        assert sync_queue.read_entry(queue_dir, "sess-1").reassess_state == sync_queue.REASSESS_SUBMITTED

        sync_worker.run_sync_once(db, settings)
        assert sync_queue.read_entry(queue_dir, "sess-1").reassess_state == sync_queue.REASSESS_DONE
        assert fakes.names() == ["intake", "pair", "pair", "pair"]

    def test_pair_transient_errors_are_budgeted(
        self, monkeypatch, gk_rs_standin, tmp_path, db
    ):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin, limit=2)
        _CloudFakes(monkeypatch, pair_answers=[PairPushConnectionError("down")])

        sync_worker.run_sync_once(db, settings)
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_SUBMITTED, 1)

        sync_worker.run_sync_once(db, settings)
        assert sync_queue.read_entry(queue_dir, "sess-1").reassess_state == sync_queue.REASSESS_FAILED

    def test_pair_refusal_is_final(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(monkeypatch, pair_answers=[PairPushHTTPError(422, json.dumps({"detail": "bad pair"}))])

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_FAILED
        assert "bad pair" in entry.reassess_reason

    def test_a_referenced_pair_file_that_is_gone_fails_with_the_reason(
        self, monkeypatch, gk_rs_standin, tmp_path, db
    ):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        (tmp_path / "sess-1.sync-pair.json").unlink()
        fakes = _CloudFakes(monkeypatch)

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_FAILED
        assert "sess-1.sync-pair.json" in entry.reassess_reason
        assert fakes.names() == ["intake"]

    def test_an_entry_without_a_pair_file_still_re_assesses(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin, with_pair=False)
        fakes = _CloudFakes(monkeypatch)

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_DONE
        assert entry.cloud_reassessed_report_id == "cloud-report-1"
        assert "no image pair" in entry.reassess_reason
        assert fakes.names() == ["intake"]

    def test_a_missing_session_record_fails_without_a_call(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin, with_record=False)
        fakes = _CloudFakes(monkeypatch)

        sync_worker.run_sync_once(db, settings)

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_FAILED
        assert "no session record" in entry.reassess_reason
        assert fakes.calls == []
        self._assert_synced_as_without_the_step(db, queue_dir)

    def test_an_unexpected_error_never_undoes_the_sync(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(monkeypatch, intake_answers=[RuntimeError("boom")])

        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == ["sess-1"]
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert (entry.reassess_state, entry.reassess_attempts) == (sync_queue.REASSESS_PENDING, 1)
        assert "boom" in entry.reassess_reason
        self._assert_synced_as_without_the_step(db, queue_dir)

    def test_a_legacy_synced_entry_is_picked_up_by_the_second_pass(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin, with_pair=False, index_row=False)
        sync_queue.mark_sync_confirmed(queue_dir, "sess-1")
        sync_queue.mark_synced(queue_dir, "sess-1", "old-edge-uuid")
        fakes = _CloudFakes(monkeypatch)

        result = sync_worker.run_sync_once(db, settings)

        assert result.uploaded == []
        assert gk_rs_standin.last_upload_bodies == []
        assert fakes.calls[0][3] == "old-edge-uuid"
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state == sync_queue.REASSESS_DONE
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "old-edge-uuid"

    def test_the_step_switched_off_makes_no_call_and_writes_no_state(self, monkeypatch, gk_rs_standin, tmp_path, db):
        settings, queue_dir = self._setup(tmp_path, db, gk_rs_standin)
        settings.sync_reassess_enabled = False
        fakes = _CloudFakes(monkeypatch)

        sync_worker.run_sync_once(db, settings)

        assert fakes.calls == []
        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.reassess_state is None
        assert json.loads((queue_dir / "sess-1.manifest.json").read_text())["reassess_state"] is None
        self._assert_synced_as_without_the_step(db, queue_dir)

        sync_worker.run_sync_once(db, settings)
        assert fakes.calls == []


class TestSummaryLine:
    @staticmethod
    def _run_main(monkeypatch, db, settings, caplog):
        monkeypatch.setattr(sync_worker, "get_settings", lambda: settings)
        monkeypatch.setattr(sync_worker, "SessionLocal", lambda: db)
        monkeypatch.setattr(sync_worker.logging, "basicConfig", lambda **kwargs: None)
        logging.getLogger("orchestrator.sync_worker").disabled = False
        with caplog.at_level("INFO", logger="orchestrator.sync_worker"):
            sync_worker.main()
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith("sync run complete")]

    def test_the_line_names_the_reassessment_counts(self, monkeypatch, gk_rs_standin, tmp_path, db, caplog):
        settings, _ = TestReassessmentStep._setup(tmp_path, db, gk_rs_standin)
        _CloudFakes(monkeypatch)

        lines = self._run_main(monkeypatch, db, settings, caplog)

        assert lines == ["sync run complete: uploaded=1 failed=0 retired=0 reassessed=1 reassess_failed=0"]

    def test_the_line_is_the_previous_one_with_the_step_off(self, monkeypatch, gk_rs_standin, tmp_path, db, caplog):
        settings, _ = TestReassessmentStep._setup(tmp_path, db, gk_rs_standin)
        settings.sync_reassess_enabled = False
        _CloudFakes(monkeypatch)

        lines = self._run_main(monkeypatch, db, settings, caplog)

        assert lines == ["sync run complete: uploaded=1 failed=0 retired=0"]
