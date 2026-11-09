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
app/client.py's write-stall / response-wait split, checked against the real
SessionFeederClient and real loopback sockets. Hermeticity: fixtures bind
loopback only, in-process, torn down by the test -- no docker, no
external network.
"""
from __future__ import annotations

import inspect
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest

from app import accounting, client as client_module
from app.client import SessionFeederClient
from app.config import DeploymentProfile, Settings
from app.store import ReplayImage, ReplaySession
from tests.conftest import make_settings


def _session_with_one_image(tmp_path: Path, *, image_bytes: int, session_id: str = "sess-stall") -> ReplaySession:
    img_path = tmp_path / "big.jpg"
    img_path.write_bytes(bytes(image_bytes))
    return ReplaySession(
        session_id=session_id,
        session_dir=tmp_path,
        crop="wheat",
        session_start="2026-01-01T00:00:00Z",
        session_end="2026-01-01T00:05:00Z",
        growth_stage=None,
        images_per_plant=None,
        images=[ReplayImage(filename="big.jpg", path=img_path, position=None, timestamp=None)],
    )


class _LoopbackNeverRespondingServer:
    """
    Loopback-only, in-process TCP peer that accepts one connection and
    never reads from -- nor writes to -- it again. Combined with a payload
    sized past this system's observed send-buffer autotune ceiling, the
    client's own write genuinely exhausts its send buffer and blocks
    (send-phase stall); combined with a SMALL payload, the write completes
    near-instantly (it fits in the kernel buffer) and the client instead
    blocks awaiting a response that never arrives (response-phase expiry).
    SO_RCVBUF is shrunk on the LISTENING socket (not the accepted one) so
    the accepted connection inherits a small window from the handshake
    onward.
    """

    def __init__(self, *, rcvbuf: Optional[int] = None) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if rcvbuf is not None:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._conn: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        self._conn = conn

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass


_VALID_SESSION_RESPONSE_BODY = b'{"session_id": "sess-min", "best_image_id": "", "infestation_score": 0.0}'


class _LoopbackDelayedResponseServer:
    """
    Loopback-only, in-process TCP peer that consumes one small request
    fully and promptly, then DELAYS before sending one complete, valid
    response -- the body is consumed promptly,
    only the response itself is late.
    """

    def __init__(self, *, delay: float) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._delay = delay
        self._conn: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        self._conn = conn
        try:
            buf = b""
            conn.settimeout(10.0)
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            content_length = _parse_content_length(buf)
            already = len(buf.split(b"\r\n\r\n", 1)[1])
            remaining = max(content_length - already, 0)
            while remaining > 0:
                chunk = conn.recv(min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
            time.sleep(self._delay)
            header = (
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(_VALID_SESSION_RESPONSE_BODY)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
            conn.sendall(header + _VALID_SESSION_RESPONSE_BODY)
        except OSError:
            return

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass


def _parse_content_length(headers_blob: bytes) -> int:
    for line in headers_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            return int(line.split(b":", 1)[1].strip())
    return 0


class TestSendPhaseStall:
    def test_genuine_send_stall_raises_upload_stalled_with_accounting(self, tmp_path):
        """A send-phase stall fails with the `upload-stalled` marker, not a generic "HTTP request failed" error."""
        session = _session_with_one_image(tmp_path, image_bytes=3_000_000)
        server = _LoopbackNeverRespondingServer(rcvbuf=4096)
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=f"http://127.0.0.1:{server.port}",
            REPLAY_UPLOAD_STALL_SECONDS="1",
            REPLAY_RESPONSE_WAIT_SECONDS="30",
        )
        try:
            with SessionFeederClient(settings) as client:
                start = time.monotonic()
                result = client.submit_session(session)
                elapsed = time.monotonic() - start
        finally:
            server.close()

        assert not result.success
        assert "upload-stalled" in result.error
        assert "1" in result.error
        assert "response-timeout" not in result.error
        assert elapsed < 10.0, "must fail close to the 1s stall bound, not the 30s response bound"

        record = accounting.load_record(settings, session.session_id)
        assert record is not None
        upload_transfers = [
            t for t in record["transfers"] if t["kind"] == "session_upload" and t["direction"] == "up"
        ]
        assert len(upload_transfers) == 1, "the attempt's upload entry must exist under the existing transfer kind"
        assert upload_transfers[0]["bytes"] > 0, "must carry the attempted (offered-to-transport) byte count"


class TestResponsePhaseExpiry:
    def test_response_never_arrives_raises_response_timeout_distinct_marker(self, tmp_path):
        session = _session_with_one_image(tmp_path, image_bytes=2_000)
        server = _LoopbackNeverRespondingServer()
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=f"http://127.0.0.1:{server.port}",
            REPLAY_UPLOAD_STALL_SECONDS="30",
            REPLAY_RESPONSE_WAIT_SECONDS="1",
        )
        try:
            with SessionFeederClient(settings) as client:
                start = time.monotonic()
                result = client.submit_session(session)
                elapsed = time.monotonic() - start
        finally:
            server.close()

        assert not result.success
        assert "response-timeout" in result.error
        assert "1" in result.error
        assert "upload-stalled" not in result.error
        assert elapsed < 10.0, "must fail close to the 1s response bound, not the 30s stall bound"

        record = accounting.load_record(settings, session.session_id)
        assert record is not None
        upload_transfers = [
            t for t in record["transfers"] if t["kind"] == "session_upload" and t["direction"] == "up"
        ]
        assert len(upload_transfers) == 1
        assert upload_transfers[0]["bytes"] > 0


class TestSlowButMovingUploadCompletes:
    def test_paced_server_upload_exceeds_stall_bound_severalfold_and_succeeds(self, tmp_path):
        """Httpx bounds are per-operation, so the stall bound never kills a steadily progressing transfer."""
        session = _session_with_one_image(tmp_path, image_bytes=50_000)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        conns: list[socket.socket] = []

        def serve() -> None:
            conn, _ = sock.accept()
            conns.append(conn)
            buf = b""
            conn.settimeout(10.0)
            try:
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                content_length = _parse_content_length(buf)
                already = len(buf.split(b"\r\n\r\n", 1)[1])
                remaining = max(content_length - already, 0)
                while remaining > 0:
                    time.sleep(0.2)
                    chunk = conn.recv(min(4096, remaining))
                    if not chunk:
                        return
                    remaining -= len(chunk)
                header = (
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(_VALID_SESSION_RESPONSE_BODY)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode()
                conn.sendall(header + _VALID_SESSION_RESPONSE_BODY)
            except OSError:
                return

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        bound = 0.3
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=f"http://127.0.0.1:{port}",
            REPLAY_UPLOAD_STALL_SECONDS=str(bound),
            REPLAY_RESPONSE_WAIT_SECONDS="10",
        )
        try:
            with SessionFeederClient(settings) as client:
                start = time.monotonic()
                result = client.submit_session(session)
                elapsed = time.monotonic() - start
        finally:
            sock.close()
            for c in conns:
                c.close()

        assert result.success, result.error
        assert elapsed > bound * 3, "total duration must exceed the stall bound severalfold while still succeeding"


class TestSlowServiceResponseIsAwaited:
    def test_delayed_response_within_response_bound_succeeds(self, tmp_path):
        """Only the legacy REPLAY_REQUEST_TIMEOUT_SECONDS is set to 1 s, together with REPLAY_RESPONSE_WAIT_SECONDS=30;
        the server delays its answer by 3 s. The response-wait bound governs, so the delayed response is awaited."""
        session = _session_with_one_image(tmp_path, image_bytes=1_000)
        legacy_read_bound = 1.0
        response_bound = 30.0
        delay = 3.0
        server = _LoopbackDelayedResponseServer(delay=delay)
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=f"http://127.0.0.1:{server.port}",
            REPLAY_REQUEST_TIMEOUT_SECONDS=str(legacy_read_bound),
            REPLAY_RESPONSE_WAIT_SECONDS=str(response_bound),
        )
        try:
            with SessionFeederClient(settings) as client:
                start = time.monotonic()
                result = client.submit_session(session)
                elapsed = time.monotonic() - start
        finally:
            server.close()

        assert result.success, result.error
        assert delay > legacy_read_bound, "the delay must exceed the legacy read bound"
        assert elapsed > legacy_read_bound, "the response's arrival time must exceed the legacy bound and still succeed"

    def test_push_uses_the_same_shared_client_timeout_construction(self, tmp_path):
        """The push path uses the same client timeouts: one shared
        httpx.Client construction, not a second bespoke one."""
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_UPLOAD_STALL_SECONDS="7",
            REPLAY_RESPONSE_WAIT_SECONDS="77",
        )
        with SessionFeederClient(settings) as client:
            assert client._http.timeout.write == 7.0
            assert client._http.timeout.read == 77.0


class TestConfigUnitMatrix:
    def test_legacy_key_alone_populates_both_new_bounds(self):
        settings = make_settings(REPLAY_REQUEST_TIMEOUT_SECONDS="45")
        assert settings.effective_upload_stall_seconds == 45
        assert settings.effective_response_wait_seconds == 45

    def test_new_stall_key_alone_overrides_its_half_legacy_fills_the_other(self):
        settings = make_settings(REPLAY_UPLOAD_STALL_SECONDS="15", REPLAY_REQUEST_TIMEOUT_SECONDS="45")
        assert settings.effective_upload_stall_seconds == 15
        assert settings.effective_response_wait_seconds == 45

    def test_new_response_key_alone_overrides_its_half_legacy_fills_the_other(self):
        settings = make_settings(REPLAY_RESPONSE_WAIT_SECONDS="200", REPLAY_REQUEST_TIMEOUT_SECONDS="45")
        assert settings.effective_upload_stall_seconds == 45
        assert settings.effective_response_wait_seconds == 200

    def test_both_new_keys_together_ignore_the_legacy_value(self):
        settings = make_settings(
            REPLAY_UPLOAD_STALL_SECONDS="15",
            REPLAY_RESPONSE_WAIT_SECONDS="200",
            REPLAY_REQUEST_TIMEOUT_SECONDS="45",
        )
        assert settings.effective_upload_stall_seconds == 15
        assert settings.effective_response_wait_seconds == 200

    def test_neither_key_set_falls_back_to_the_new_built_in_defaults(self):
        settings = make_settings()
        assert settings.effective_upload_stall_seconds == 60
        assert settings.effective_response_wait_seconds == 300


class TestDeprecationLogLine:
    def test_logged_exactly_when_legacy_key_is_set(self, caplog):
        with caplog.at_level("WARNING"):
            make_settings(REPLAY_REQUEST_TIMEOUT_SECONDS="45")
        assert any("REPLAY_REQUEST_TIMEOUT_SECONDS" in r.message and "deprecated" in r.message for r in caplog.records)

    def test_not_logged_when_legacy_key_is_unset(self, caplog):
        with caplog.at_level("WARNING"):
            make_settings()
        assert not any("deprecated" in r.message for r in caplog.records)


class TestConstructedTimeoutIsExactlyFourPerOperationFields:
    def test_timeout_object_carries_the_expected_bounds_and_nothing_else(self, tmp_path):
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_UPLOAD_STALL_SECONDS="12",
            REPLAY_RESPONSE_WAIT_SECONDS="34",
        )
        with SessionFeederClient(settings) as client:
            timeout = client._http.timeout
            assert timeout.connect == 10.0
            assert timeout.write == 12.0
            assert timeout.read == 34.0
            assert timeout.pool == 10.0

    def test_upload_call_path_wraps_no_deadline_mechanism_around_the_request(self):
        """
        No total-request deadline of any kind: the
        method that issues the upload POST contains no deadline-wrapping
        construct (wait_for, a Timer, a signal alarm) around the request, because
        a finite behavioral test alone can only rule out caps SHORTER than
        its own duration.
        """
        source = inspect.getsource(client_module.SessionFeederClient._upload_session)
        for forbidden in ("wait_for(", "signal.alarm", "threading.Timer", "asyncio"):
            assert forbidden not in source, f"found a deadline-wrapping construct: {forbidden!r}"
