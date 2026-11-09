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
The production HttpxProbeTransport exercised against real loopback sockets: connection reuse across a run's samples, the production exception
translation for a genuinely refused connection, and the transport's
close guaranteed on every exit path.
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest

from app import link_probe
from tests.conftest import make_settings


def _probe_settings(**overrides):
    defaults = dict(
        REPLAY_DEPLOYMENT_PROFILE="x86_ci",
        REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud-inf.test",
    )
    defaults.update(overrides)
    return make_settings(**defaults)


def _free_local_port() -> int:
    """
    A local TCP port nothing is listening on: bind then immediately
    close, so the connection attempt this test makes right after is
    refused by the OS (ECONNREFUSED), not merely slow.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ConnectionCountingSink:
    """
    Loopback-only, in-process HTTP/1.1 sink that counts distinct
    ACCEPTED TCP connections while answering every request 200 OK on
    whichever connection it arrived on -- proving reuse (the real-transport test method): N samples through ONE reused
    httpx.Client must show 1 connection and N requests. No HTTP
    library on the server side: a minimal hand-rolled parser reads
    exactly one request's headers + Content-Length body per iteration,
    replies without `Connection: close`, and loops on the SAME socket
    for the next request -- real HTTP/1.1 keep-alive, not a stub.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self.connection_count = 0
        self.request_count = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self.connection_count += 1
            threading.Thread(target=self._serve_connection, args=(conn,), daemon=True).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        buf = b""
        try:
            while True:
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                header_end = buf.index(b"\r\n\r\n") + 4
                headers_blob = buf[:header_end]
                body_so_far = buf[header_end:]
                content_length = 0
                for line in headers_blob.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        content_length = int(line.split(b":", 1)[1].strip())
                while len(body_so_far) < content_length:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    body_so_far += chunk
                buf = body_so_far[content_length:]
                with self._lock:
                    self.request_count += 1
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


class TestRealTransportConnectionReuse:
    @pytest.mark.parametrize("n_samples", [1, 3, 5])
    def test_n_samples_ride_exactly_one_connection(self, n_samples):
        sink = _ConnectionCountingSink()
        try:
            settings = _probe_settings(
                REPLAY_PROBE_SAMPLES=str(n_samples),
                REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
                REPLAY_CLOUD_INFERENCE_BASE_URL=f"http://127.0.0.1:{sink.port}",
            )
            transport = link_probe.HttpxProbeTransport()
            outcome = link_probe.run_probe(settings, transport, time.monotonic)
        finally:
            sink.close()

        assert outcome.failure_reason is None
        assert len(outcome.samples_kbps) == n_samples
        assert outcome.median_kbps is not None
        assert sink.connection_count == 1, f"expected exactly 1 accepted connection, sink saw {sink.connection_count}"
        assert sink.request_count == n_samples


class TestRealTransportRefusedConnectionIsTyped:
    def test_refused_connection_short_circuits_to_typed_reason_never_unhandled(self):
        port = _free_local_port()
        settings = _probe_settings(
            REPLAY_PROBE_SAMPLES="3",
            REPLAY_CLOUD_INFERENCE_BASE_URL=f"http://127.0.0.1:{port}",
        )
        transport = link_probe.HttpxProbeTransport()

        outcome = link_probe.run_probe(settings, transport, time.monotonic)

        assert outcome.failure_reason == link_probe.FAILURE_CONNECTION_REFUSED
        assert outcome.median_kbps is None
        assert len(outcome.samples_kbps) == 0, "fail-closed short-circuit: no further samples attempted"


class TestTransportClosedOnEveryPath:
    """
    `run_probe`'s `try/finally: transport.close`
    must release the underlying `httpx.Client` on every exit path. Each
    case captures the REAL `httpx.Client` instance `HttpxProbeTransport`
    lazily creates (by wrapping its `_get_client` seam) and asserts
    httpx's own documented `Client.is_closed` property afterward -- not
    a hand-rolled flag standing in for httpx's own closed-state.
    """

    def _capture_created_client(self, monkeypatch):
        created: dict = {}
        original = link_probe.HttpxProbeTransport._get_client

        def _wrapped(self_transport):
            client = original(self_transport)
            created["client"] = client
            return client

        monkeypatch.setattr(link_probe.HttpxProbeTransport, "_get_client", _wrapped)
        return created

    @pytest.mark.parametrize("path_case", ["green-3-samples", "connection-refused", "timeout", "bad-status", "unexpected-raise", "no-samples"])
    def test_client_closed_on_every_path(self, monkeypatch, path_case):
        created = self._capture_created_client(monkeypatch)
        sink = None
        try:
            if path_case == "green-3-samples":
                sink = _ConnectionCountingSink()
                settings = _probe_settings(
                    REPLAY_PROBE_SAMPLES="3",
                    REPLAY_CLOUD_INFERENCE_BASE_URL=f"http://127.0.0.1:{sink.port}",
                )
            elif path_case == "connection-refused":
                settings = _probe_settings(REPLAY_CLOUD_INFERENCE_BASE_URL=f"http://127.0.0.1:{_free_local_port()}")
            elif path_case == "no-samples":
                settings = _probe_settings(REPLAY_PROBE_SAMPLES="0")
            else:
                settings = _probe_settings()
                if path_case == "timeout":
                    def _raise_timeout(self_client, *a, **kw):
                        raise httpx.ReadTimeout("simulated timeout", request=httpx.Request("POST", "http://cloud-inf.test"))
                    monkeypatch.setattr(httpx.Client, "post", _raise_timeout)
                elif path_case == "bad-status":
                    monkeypatch.setattr(httpx.Client, "post", lambda self_client, *a, **kw: httpx.Response(500))
                elif path_case == "unexpected-raise":
                    def _raise_unexpected(self_client, *a, **kw):
                        raise RuntimeError("boom -- not one of the typed ProbeTransportError reasons")
                    monkeypatch.setattr(httpx.Client, "post", _raise_unexpected)

            transport = link_probe.HttpxProbeTransport()
            if path_case == "unexpected-raise":
                with pytest.raises(RuntimeError, match="boom"):
                    link_probe.run_probe(settings, transport, time.monotonic)
            else:
                outcome = link_probe.run_probe(settings, transport, time.monotonic)
                expected_reason = {
                    "green-3-samples": None,
                    "connection-refused": link_probe.FAILURE_CONNECTION_REFUSED,
                    "timeout": link_probe.FAILURE_TIMEOUT,
                    "bad-status": link_probe.FAILURE_BAD_STATUS,
                    "no-samples": link_probe.FAILURE_NO_SAMPLES,
                }[path_case]
                assert outcome.failure_reason == expected_reason
                if path_case == "green-3-samples":
                    assert len(outcome.samples_kbps) == 3
        finally:
            if sink is not None:
                sink.close()

        if path_case == "no-samples":
            if "client" in created:
                assert created["client"].is_closed
        else:
            assert "client" in created, "the real httpx.Client was never created on this path"
            assert created["client"].is_closed, f"real httpx.Client not closed on the {path_case} path"

    def test_closed_after_unexpected_exception_mid_run_which_still_propagates(self):
        class _BoomTransport:
            def __init__(self) -> None:
                self.closed = False

            def upload(self, url, headers, payload, timeout):
                raise RuntimeError("boom -- not one of the typed ProbeTransportError reasons")

            def close(self) -> None:
                self.closed = True

        settings = _probe_settings()
        transport = _BoomTransport()

        with pytest.raises(RuntimeError, match="boom"):
            link_probe.run_probe(settings, transport, time.monotonic)

        assert transport.closed, "the transport must be closed even though the exception propagated"
