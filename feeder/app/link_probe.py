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
Session feeder — link probe.

Takes N timed samples, after the round-trip and ramp uploads, by uploading a
calibrated, deterministic payload to the authenticated probe-sink
endpoint on the CLOUD inference target
(`REPLAY_CLOUD_INFERENCE_BASE_URL` + its API key) -- the same network
path a cloud session upload takes. Client-side
wall-clock timing only: transport and clock are both injectable, and
every failure shape collapses to exactly one of four typed reasons so
the selector never sees a raw exception.

On a high-latency link a small upload measures the round trip, not the
bandwidth, so `run_probe` measures the round trip first, ramps the payload
until an upload lasts `probe_min_sample_seconds` (bounded by
`probe_max_payload_bytes`, itself bound by the sink's cap) and subtracts the
round trip from each sample; a floor of zero restores the plain probe.

The transport reuses ONE connection across a run's samples, matching the persistent-connection shape of the session upload client the
probe exists to predict. `run_probe` owns the
transport's lifecycle and guarantees `close` on every exit path.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

import httpx

from app.config import Settings

PROBE_SINK_PATH = "/api/v1/probe/sink"

_MIN_ELAPSED_SECONDS = 1e-6
HTTP_PAYLOAD_TOO_LARGE = 413

FAILURE_CONNECTION_REFUSED = "connection-refused"
FAILURE_TIMEOUT = "timeout"
FAILURE_BAD_STATUS = "bad-status"
FAILURE_NO_SAMPLES = "no-samples"


@dataclass
class TransportSample:
    """One successful upload attempt's report from a ProbeTransport."""

    bytes_written: int


class ProbeTransportError(Exception):
    """
    Base for a single upload attempt's failure. Always carries what the
    transport actually wrote before failing (never the requested size) so
    the probe's own bytes-written total stays honest on failure paths too.
    """

    def __init__(self, message: str, bytes_written: int = 0) -> None:
        super().__init__(message)
        self.bytes_written = bytes_written


class ProbeConnectionRefused(ProbeTransportError):
    pass


class ProbeTimeout(ProbeTransportError):
    pass


class ProbeBadStatus(ProbeTransportError):
    def __init__(self, status_code: int, message: str = "", bytes_written: int = 0) -> None:
        super().__init__(message or f"probe sink returned HTTP {status_code}", bytes_written)
        self.status_code = status_code


class ProbeTransport(Protocol):
    def upload(self, url: str, headers: dict, payload: bytes, timeout: float) -> TransportSample:
        ...

    def close(self) -> None:
        ...


class HttpxProbeTransport:
    """
    Production transport: a real POST via httpx, over ONE lazily
    created `httpx.Client` reused across every `upload` call in a
    probe run -- the per-request timeout is still passed to each
    individual `post`, so the per-sample stall-bound semantics are
    unchanged; only the underlying connection is shared.
    """

    def __init__(self, verify: bool = True) -> None:
        self._client: Optional[httpx.Client] = None
        self._verify = verify

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(verify=self._verify)
        return self._client

    def upload(self, url: str, headers: dict, payload: bytes, timeout: float) -> TransportSample:
        client = self._get_client()
        try:
            resp = client.post(url, headers=headers, content=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProbeTimeout(str(exc), bytes_written=0) from exc
        except httpx.ConnectError as exc:
            raise ProbeConnectionRefused(str(exc), bytes_written=0) from exc
        except httpx.HTTPError as exc:
            raise ProbeConnectionRefused(str(exc), bytes_written=0) from exc
        if not resp.is_success:
            raise ProbeBadStatus(resp.status_code, bytes_written=len(payload))
        return TransportSample(bytes_written=len(payload))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass
class ProbeOutcome:
    samples_kbps: List[float] = field(default_factory=list)
    median_kbps: Optional[float] = None
    failure_reason: Optional[str] = None
    bytes_written: int = 0
    payload_bytes: Optional[int] = None
    rtt_ms: Optional[float] = None
    ramp_samples_kbps: List[float] = field(default_factory=list)


def _kbps(byte_count: int, elapsed: float, rtt: Optional[float]) -> float:
    """
    Bytes over the transfer time: a request-response upload lasts one round
    trip plus the transfer, so the measured round trip is subtracted. A
    reading no longer than the round trip (a jittery link) keeps its elapsed
    time unchanged -- the conservative side. `rtt` None (floor zero) is the
    plain bytes-over-elapsed quotient.
    """
    if rtt is not None and elapsed > rtt:
        seconds = max(elapsed - rtt, _MIN_ELAPSED_SECONDS)
    else:
        seconds = max(elapsed, _MIN_ELAPSED_SECONDS)
    return (byte_count * 8.0 / 1000.0) / seconds


_FAILURE_BY_TRANSPORT_ERROR = (
    (ProbeTimeout, FAILURE_TIMEOUT),
    (ProbeConnectionRefused, FAILURE_CONNECTION_REFUSED),
    (ProbeBadStatus, FAILURE_BAD_STATUS),
)


def run_probe(
    settings: Settings,
    transport: ProbeTransport,
    clock: Callable[[], float] = time.monotonic,
) -> ProbeOutcome:
    """
    Measures the uplink with request-response uploads to the cloud sink, in
    this order:

    1. When settings.probe_min_sample_seconds > 0, one 1-byte upload; its
       elapsed time is the round trip (`rtt`).
    2. The ramp: an upload of settings.probe_payload_bytes; while its
       elapsed time is below the floor and the payload is below
       settings.probe_max_payload_bytes, the payload doubles (capped at the
       maximum) and uploads again. The earlier readings go to
       `ramp_samples_kbps`; the first upload that lasts the floor, or the
       first at the maximum, settles the size and is the first sample. A
       ramp step above probe_payload_bytes that the sink refuses as too large
       (HTTP 413) settles the size at the previous payload instead of failing
       the probe; the refused size is never a sample and all
       settings.probe_samples samples are uploaded at the settled size. A 413
       to probe_payload_bytes itself fails as bad-status.
    3. The remaining samples (settings.probe_samples - 1 after a settling
       upload) at the settled size.

    Each reading is bytes over elapsed time minus `rtt` (elapsed unchanged
    when it does not exceed `rtt`). With the floor at zero there is no
    round-trip upload and no ramp: probe_samples uploads of
    probe_payload_bytes, bytes over elapsed.

    The first failure of any kind, at any stage, short-circuits the probe to its typed reason (no further upload is attempted), with
    bytes_written carrying everything actually written so far (every
    successful upload plus whatever the failing attempt itself reported).
    The transport is closed on every exit path -- success, every typed
    failure, and an unexpected raise alike.
    """
    url = f"{settings.cloud_inference_base_url.rstrip('/')}{PROBE_SINK_PATH}"
    headers = settings.cloud_auth_headers
    floor = settings.probe_min_sample_seconds

    samples_kbps: List[float] = []
    ramp_samples_kbps: List[float] = []
    total_bytes_written = 0
    rtt: Optional[float] = None
    settled_bytes: Optional[int] = None
    previous_bytes = settings.probe_payload_bytes

    def outcome(failure_reason: Optional[str] = None, median: Optional[float] = None) -> ProbeOutcome:
        return ProbeOutcome(
            samples_kbps=samples_kbps,
            median_kbps=median,
            failure_reason=failure_reason,
            bytes_written=total_bytes_written,
            payload_bytes=settled_bytes,
            rtt_ms=rtt * 1000.0 if rtt is not None else None,
            ramp_samples_kbps=ramp_samples_kbps,
        )

    def timed_upload(size: int) -> tuple[float, int]:
        start = clock()
        sample = transport.upload(url, headers, bytes(size), settings.probe_timeout_seconds)
        elapsed = clock() - start
        if elapsed <= 0:
            elapsed = _MIN_ELAPSED_SECONDS
        return elapsed, sample.bytes_written

    try:
        if settings.probe_samples <= 0:
            return outcome(FAILURE_NO_SAMPLES)

        try:
            if floor > 0:
                rtt, written = timed_upload(1)
                total_bytes_written += written

            payload_bytes = settings.probe_payload_bytes
            while True:
                try:
                    elapsed, written = timed_upload(payload_bytes)
                except ProbeBadStatus as exc:
                    if exc.status_code != HTTP_PAYLOAD_TOO_LARGE or payload_bytes == settings.probe_payload_bytes:
                        raise
                    total_bytes_written += exc.bytes_written
                    settled_bytes = previous_bytes
                    break
                total_bytes_written += written
                kbps = _kbps(written, elapsed, rtt)
                if elapsed >= floor or payload_bytes >= settings.probe_max_payload_bytes:
                    settled_bytes = payload_bytes
                    samples_kbps.append(kbps)
                    break
                ramp_samples_kbps.append(kbps)
                previous_bytes = payload_bytes
                payload_bytes = min(payload_bytes * 2, settings.probe_max_payload_bytes)

            for _ in range(settings.probe_samples - len(samples_kbps)):
                elapsed, written = timed_upload(settled_bytes)
                total_bytes_written += written
                samples_kbps.append(_kbps(written, elapsed, rtt))
        except ProbeTransportError as exc:
            total_bytes_written += exc.bytes_written
            for error_type, reason in _FAILURE_BY_TRANSPORT_ERROR:
                if isinstance(exc, error_type):
                    return outcome(reason)
            raise

        return outcome(median=statistics.median(samples_kbps))
    finally:
        transport.close()
