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

"""Tests for app/link_probe.py."""
from __future__ import annotations

import math
import socket
import threading
import time
from typing import List, Optional

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


class _FixedClock:
    """Advances by a fixed step each call -- 1.0s per 32768-byte sample."""

    def __init__(self, step: float) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        t = self._t
        self._t += self._step
        return t


class _SuccessTransport:
    def __init__(self) -> None:
        self.calls: List[dict] = []
        self.closed = False

    def upload(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "payload_len": len(payload), "timeout": timeout})
        return link_probe.TransportSample(bytes_written=len(payload))

    def close(self) -> None:
        self.closed = True


class _FailingTransport:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.call_count = 0
        self.closed = False

    def upload(self, url, headers, payload, timeout):
        self.call_count += 1
        raise self._exc

    def close(self) -> None:
        self.closed = True


class TestProbeTiming:
    def test_median_kbps_from_injected_clock(self):
        settings = _probe_settings(
            REPLAY_PROBE_SAMPLES="3", REPLAY_PROBE_PAYLOAD_BYTES="32768", REPLAY_PROBE_MIN_SAMPLE_SECONDS="0"
        )
        transport = _SuccessTransport()
        clock = _FixedClock(1.0)

        outcome = link_probe.run_probe(settings, transport, clock)

        assert outcome.failure_reason is None
        assert len(outcome.samples_kbps) == 3
        for kbps in outcome.samples_kbps:
            assert kbps == pytest.approx(262.144, rel=1e-6)
        assert outcome.median_kbps == pytest.approx(262.144, rel=1e-6)
        assert outcome.bytes_written == 32768 * 3

    def test_zero_elapsed_clamps_to_a_minimal_tick_and_returns_finite_kbps(self):
        """A degenerate (non-positive) timing measurement must not crash the
        probe with a ZeroDivisionError -- deterministic clamp."""
        settings = _probe_settings(REPLAY_PROBE_SAMPLES="1")
        transport = _SuccessTransport()
        clock = _FixedClock(0.0)

        outcome = link_probe.run_probe(settings, transport, clock)

        assert outcome.failure_reason is None
        assert len(outcome.samples_kbps) == 1
        assert math.isfinite(outcome.samples_kbps[0])
        assert outcome.samples_kbps[0] > 0


class TestProbeFailureShapes:
    def test_connection_refused_maps_to_typed_reason_no_exception(self):
        settings = _probe_settings()
        transport = _FailingTransport(link_probe.ProbeConnectionRefused("refused", bytes_written=0))
        outcome = link_probe.run_probe(settings, transport, _FixedClock(1.0))
        assert outcome.failure_reason == link_probe.FAILURE_CONNECTION_REFUSED
        assert outcome.median_kbps is None

    def test_timeout_maps_to_typed_reason_and_short_circuits(self):
        settings = _probe_settings(REPLAY_PROBE_SAMPLES="3")
        transport = _FailingTransport(link_probe.ProbeTimeout("timed out", bytes_written=100_000))
        outcome = link_probe.run_probe(settings, transport, _FixedClock(1.0))
        assert outcome.failure_reason == link_probe.FAILURE_TIMEOUT
        assert transport.call_count == 1, "the first timeout must short-circuit -- no further samples attempted"
        assert outcome.bytes_written == 100_000

    def test_bad_status_maps_to_typed_reason(self):
        settings = _probe_settings()
        transport = _FailingTransport(link_probe.ProbeBadStatus(500, bytes_written=262144))
        outcome = link_probe.run_probe(settings, transport, _FixedClock(1.0))
        assert outcome.failure_reason == link_probe.FAILURE_BAD_STATUS
        assert outcome.bytes_written == 262144

    def test_zero_configured_samples_yields_no_samples_failure(self):
        settings = _probe_settings(REPLAY_PROBE_SAMPLES="0")
        transport = _SuccessTransport()
        outcome = link_probe.run_probe(settings, transport, _FixedClock(1.0))
        assert outcome.failure_reason == link_probe.FAILURE_NO_SAMPLES
        assert outcome.median_kbps is None


class TestProbeWiring:
    def test_construction_consumes_configured_values_cloud_key_set(self):
        settings = _probe_settings(
            REPLAY_PROBE_SAMPLES="2",
            REPLAY_PROBE_PAYLOAD_BYTES="1000",
            REPLAY_PROBE_MIN_SAMPLE_SECONDS="0",
            REPLAY_PROBE_TIMEOUT_SECONDS="12",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud-inf.test",
            REPLAY_CLOUD_INFERENCE_API_KEY="cloud-key",
            REPLAY_INFERENCE_API_KEY="local-key",
        )
        transport = _SuccessTransport()
        link_probe.run_probe(settings, transport, _FixedClock(1.0))

        assert len(transport.calls) == 2
        for call in transport.calls:
            assert call["url"] == "http://cloud-inf.test/api/v1/probe/sink"
            assert call["headers"]["X-API-Key"] == "cloud-key"
            assert call["payload_len"] == 1000
            assert call["timeout"] == 12

    def test_construction_falls_back_to_primary_key_when_cloud_key_unset(self):
        settings = _probe_settings(
            REPLAY_PROBE_SAMPLES="1",
            REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud-inf.test",
            REPLAY_INFERENCE_API_KEY="local-key",
        )
        transport = _SuccessTransport()
        link_probe.run_probe(settings, transport, _FixedClock(1.0))

        assert transport.calls[0]["headers"]["X-API-Key"] == "local-key"


class TestHttpxProbeTransportStatusHandling:
    """
    HttpxProbeTransport.upload against httpx.Response objects built
    in-memory (per httpx's own documented contract: response.is_success is
    True exactly for 2xx -- https://www.python-httpx.org/api/#response)
    so 1xx/3xx/4xx/5xx are all non-success. No real socket is opened:
    httpx.Client.post is monkeypatched (re-seamed -- a
    persistent httpx.Client never calls the module-level httpx.post) to
    return a constructed Response, which is httpx's own object exercising
    its own status-classification logic (is_success), not a hand-rolled
    stand-in for it.
    """

    def test_3xx_status_is_treated_as_bad_status_not_success(self, monkeypatch):
        monkeypatch.setattr(httpx.Client, "post", lambda self, *a, **kw: httpx.Response(303))
        transport = link_probe.HttpxProbeTransport()

        with pytest.raises(link_probe.ProbeBadStatus) as exc_info:
            transport.upload("http://cloud-inf.test/api/v1/probe/sink", {}, b"x" * 10, 5.0)
        assert exc_info.value.status_code == 303
        transport.close()

    def test_2xx_status_succeeds_with_payload_length_as_bytes_written(self, monkeypatch):
        monkeypatch.setattr(httpx.Client, "post", lambda self, *a, **kw: httpx.Response(200))
        transport = link_probe.HttpxProbeTransport()

        sample = transport.upload("http://cloud-inf.test/api/v1/probe/sink", {}, b"x" * 10, 5.0)
        assert sample.bytes_written == 10
        transport.close()


class _LoopbackNeverReadingServer:
    """
    Loopback-only, in-process TCP peer that accepts one connection and then
    never reads from it again -- no HTTP parsing, no external egress, torn
    down by the test. SO_RCVBUF is shrunk on the LISTENING socket (not the accepted
    one) so the accepted connection inherits a small window from the
    handshake onward: combined with a payload well past this system's
    observed send-buffer autotune ceiling, the client's own write
    genuinely exhausts its send buffer and blocks -- not a simulated delay.
    """

    def __init__(self, *, rcvbuf: int = 4096) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


class _LoopbackSlowResponseServer:
    """
    Loopback-only, in-process TCP peer that reads one small request fully,
    then writes its response body in small, steadily-paced chunks -- each
    gap short enough that no single read the client performs stalls past
    the bound, while the FULL response takes far longer than the bound to
    arrive in total. No HTTP parsing beyond finding the blank line that
    ends the request headers; torn down by the test.
    """

    def __init__(self, *, chunk_bytes: int, num_chunks: int, interval: float) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._chunk_bytes = chunk_bytes
        self._num_chunks = num_chunks
        self._interval = interval
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
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            total_len = self._chunk_bytes * self._num_chunks
            header = f"HTTP/1.1 200 OK\r\nContent-Length: {total_len}\r\nConnection: close\r\n\r\n".encode()
            conn.sendall(header)
            for _ in range(self._num_chunks):
                time.sleep(self._interval)
                conn.sendall(bytes(self._chunk_bytes))
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


class TestHttpxProbeTransportRealSocketProgressSemantics:
    """
    httpx's per-operation timeout semantics, which the probe relies on, are pinned against the REAL
    production transport and real loopback sockets, not only doubles. Two
    directions, each isolating one of httpx's two data-transfer phases (its
    write phase re-arms its own timeout per underlying send syscall,
    exactly like its read phase does per recv -- httpcore's ``_backends/sync.py`` (SyncStream.write/read):
    a genuinely stalled WRITE must die within the bound (no send-side
    progress ever), and a slow-but-steadily-paced READ must not be killed
    despite its TOTAL duration exceeding the bound (every individual
    operation still makes progress). Together they pin the same underlying
    platform claim from both sides.
    """

    def test_stalled_upload_raises_typed_timeout_within_the_bound(self):
        payload = bytes(3_000_000)
        bound = 0.5
        server = _LoopbackNeverReadingServer(rcvbuf=4096)
        try:
            transport = link_probe.HttpxProbeTransport()
            start = time.monotonic()
            with pytest.raises(link_probe.ProbeTimeout) as exc_info:
                transport.upload(f"http://127.0.0.1:{server.port}/api/v1/probe/sink", {}, payload, bound)
            elapsed = time.monotonic() - start
        finally:
            server.close()

        assert isinstance(exc_info.value.__cause__, httpx.WriteTimeout), (
            "must be a genuine send-phase stall, not a response-wait expiry mislabeled"
        )
        assert elapsed < bound * 3, "the typed timeout must fire close to the configured bound, not hang"

    def test_slow_but_steadily_paced_response_is_awaited_past_the_bound(self):
        payload = bytes(1_000)
        bound = 1.0
        server = _LoopbackSlowResponseServer(chunk_bytes=5_000, num_chunks=10, interval=0.3)
        try:
            transport = link_probe.HttpxProbeTransport()
            start = time.monotonic()
            sample = transport.upload(f"http://127.0.0.1:{server.port}/api/v1/probe/sink", {}, payload, bound)
            elapsed = time.monotonic() - start
        finally:
            server.close()

        assert sample.bytes_written == len(payload)
        assert elapsed > bound * 1.5, (
            "the response's TOTAL arrival time must exceed the per-operation bound while "
            "every individual read still makes progress -- slowness alone is not failure"
        )


class TestRunProbeSlowSuccessPureDouble:
    def test_run_probe_accepts_a_slow_successful_sample_with_no_extra_gate(self):
        """
        run_probe imposes no total-duration gate of its own: whatever a
        transport reports success for (however long it took), the sample
        measures. A pure double against the real wall clock, distinct from
        the real-transport socket pinning above.
        """
        settings = _probe_settings(REPLAY_PROBE_SAMPLES="1")

        class _SlowSuccessTransport:
            def upload(self, url, headers, payload, timeout):
                time.sleep(0.05)
                return link_probe.TransportSample(bytes_written=len(payload))

            def close(self) -> None:
                pass

        outcome = link_probe.run_probe(settings, _SlowSuccessTransport(), time.monotonic)

        assert outcome.failure_reason is None
        assert len(outcome.samples_kbps) == 1
        assert outcome.samples_kbps[0] > 0
