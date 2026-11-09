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

"""The probe's round-trip measurement, payload ramp and corrected quotient (app/link_probe.py) and the
sizing refusals (app/config.py), against a modelled link and the injected
clock -- no socket, no wall-clock time.
"""
from __future__ import annotations

import statistics
import time
from typing import List, Optional, Tuple

import pytest
from pydantic import ValidationError

from app import link_probe
from tests.conftest import make_settings
from tests.test_probe_transport_real import _ConnectionCountingSink

DEFAULT_SEQUENCE_NEVER_REACHING_FLOOR = [
    1,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    1048576,
    1048576,
]


def _probe_settings(**overrides):
    defaults = dict(
        REPLAY_DEPLOYMENT_PROFILE="x86_ci",
        REPLAY_CLOUD_INFERENCE_BASE_URL="http://cloud-inf.test",
    )
    defaults.update(overrides)
    return make_settings(**defaults)


class _ModelClock:
    """A clock only the modelled transport advances."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ModelledTransport:
    """
    A link of `bandwidth_kbps` and `rtt_seconds`: each upload advances the
    shared clock by rtt + bytes x 8 / 1000 / bandwidth (one request-response),
    and records the payload size and the elapsed time it modelled.
    """

    def __init__(self, bandwidth_kbps: float, rtt_seconds: float, clock: _ModelClock) -> None:
        self._bandwidth_kbps = bandwidth_kbps
        self._rtt_seconds = rtt_seconds
        self._clock = clock
        self.sizes: List[int] = []
        self.elapsed: List[float] = []
        self.closed = False

    def upload(self, url, headers, payload, timeout):
        seconds = self._rtt_seconds + len(payload) * 8.0 / 1000.0 / self._bandwidth_kbps
        self._clock.advance(seconds)
        self.sizes.append(len(payload))
        self.elapsed.append(seconds)
        return link_probe.TransportSample(bytes_written=len(payload))

    def close(self) -> None:
        self.closed = True


class _FailingAt(_ModelledTransport):
    """The modelled link, raising `error` on the upload at index `stage_index`."""

    def __init__(self, bandwidth_kbps, rtt_seconds, clock, stage_index: int, error: Exception) -> None:
        super().__init__(bandwidth_kbps, rtt_seconds, clock)
        self._stage_index = stage_index
        self._error = error
        self.attempts = 0

    def upload(self, url, headers, payload, timeout):
        index = self.attempts
        self.attempts += 1
        if index == self._stage_index:
            raise self._error
        return super().upload(url, headers, payload, timeout)


def _run(bandwidth_kbps: float, rtt_seconds: float, **overrides) -> Tuple[link_probe.ProbeOutcome, _ModelledTransport]:
    clock = _ModelClock()
    transport = _ModelledTransport(bandwidth_kbps, rtt_seconds, clock)
    outcome = link_probe.run_probe(_probe_settings(**overrides), transport, clock)
    return outcome, transport


class TestHighLatencyLinkIsMeasuredNearItsBandwidth:
    @pytest.mark.parametrize(
        "bandwidth_kbps, rtt_seconds, settled_bytes",
        [
            (8000, 0.250, 1048576),
            (3000, 0.150, 524288),
            (50000, 0.001, 1048576),
        ],
    )
    def test_ramp_settles_and_median_is_within_two_percent(self, bandwidth_kbps, rtt_seconds, settled_bytes):
        outcome, transport = _run(bandwidth_kbps, rtt_seconds)

        assert outcome.failure_reason is None
        assert outcome.payload_bytes == settled_bytes
        assert len(outcome.samples_kbps) == 3
        assert outcome.median_kbps == pytest.approx(bandwidth_kbps, rel=0.02)
        assert outcome.rtt_ms == pytest.approx(rtt_seconds * 1000.0 + 1 * 8.0 / bandwidth_kbps, rel=1e-6)
        assert transport.sizes[0] == 1, "the round-trip upload comes first"
        assert transport.sizes[-1] == settled_bytes
        assert transport.closed

    def test_ramp_readings_are_kept_out_of_the_samples(self):
        outcome, transport = _run(8000, 0.250)

        assert len(outcome.ramp_samples_kbps) == 5
        assert len(transport.sizes) == 1 + 5 + 3
        assert outcome.samples_kbps[0] == pytest.approx(8000, rel=0.02)

    def test_the_settling_upload_is_the_first_sample(self):
        outcome, transport = _run(3000, 0.150)

        assert transport.sizes == [1, 32768, 65536, 131072, 262144, 524288, 524288, 524288]
        assert len(outcome.ramp_samples_kbps) == 4


class TestSlowLinkKeepsTheInitialPayload:
    def test_no_ramp_and_each_sample_is_the_corrected_quotient(self):
        outcome, transport = _run(40, 0.400)

        assert outcome.failure_reason is None
        assert outcome.ramp_samples_kbps == []
        assert outcome.payload_bytes == 32768
        assert transport.sizes == [1, 32768, 32768, 32768]
        rtt = transport.elapsed[0]
        assert outcome.rtt_ms == pytest.approx(rtt * 1000.0)
        assert len(outcome.samples_kbps) == 3
        for elapsed, kbps in zip(transport.elapsed[1:], outcome.samples_kbps):
            assert kbps == pytest.approx(32768 * 8 / 1000 / (elapsed - rtt), rel=1e-9)


class _PairClock:
    """Start/end timestamps per upload from exact elapsed values (no accumulated float error)."""

    def __init__(self, elapsed_by_upload: List[float]) -> None:
        self._stamps = iter([t for e in elapsed_by_upload for t in (0.0, e)])

    def __call__(self) -> float:
        return next(self._stamps)


class _Recording:
    def __init__(self) -> None:
        self.sizes: List[int] = []

    def upload(self, url, headers, payload, timeout):
        self.sizes.append(len(payload))
        return link_probe.TransportSample(bytes_written=len(payload))

    def close(self):
        pass


class TestRoundTripNotBelowElapsedUsesElapsedUnchanged:
    def test_a_sample_no_longer_than_the_round_trip_is_not_corrected(self):
        settings = _probe_settings(REPLAY_PROBE_SAMPLES="1", REPLAY_PROBE_MIN_SAMPLE_SECONDS="0.1")
        clock = _PairClock([0.500, 0.400])

        outcome = link_probe.run_probe(settings, _Recording(), clock)

        assert outcome.rtt_ms == pytest.approx(500.0)
        assert outcome.samples_kbps == [32768 * 8.0 / 1000.0 / 0.400]


class TestZeroFloorRestoresThePreviousProbe:
    def test_matches_the_previous_computation_with_no_round_trip_upload(self):
        elapsed_by_call = [0.35, 0.20, 0.50]
        settings = _probe_settings(REPLAY_PROBE_MIN_SAMPLE_SECONDS="0")
        transport = _Recording()

        outcome = link_probe.run_probe(settings, transport, _PairClock(elapsed_by_call))

        reference = [32768 * 8.0 / 1000.0 / e for e in elapsed_by_call]
        assert outcome.samples_kbps == reference
        assert outcome.median_kbps == statistics.median(reference)
        assert transport.sizes == [32768, 32768, 32768]
        assert outcome.rtt_ms is None
        assert outcome.payload_bytes == 32768
        assert outcome.ramp_samples_kbps == []
        assert outcome.bytes_written == 3 * 32768

    def test_a_fast_link_is_not_ramped_when_the_floor_is_zero(self):
        outcome, transport = _run(50000, 0.001, REPLAY_PROBE_MIN_SAMPLE_SECONDS="0")

        assert transport.sizes == [32768, 32768, 32768]
        assert outcome.payload_bytes == 32768


class TestUploadCountIsBounded:
    def test_exact_upload_sequence_and_bytes_written_at_the_defaults(self):
        outcome, transport = _run(50000, 0.001)

        assert transport.sizes == DEFAULT_SEQUENCE_NEVER_REACHING_FLOOR
        assert outcome.bytes_written == sum(DEFAULT_SEQUENCE_NEVER_REACHING_FLOOR) == 4161537
        assert outcome.payload_bytes == 1048576

    def test_a_maximum_equal_to_the_initial_payload_never_ramps(self):
        outcome, transport = _run(
            50000, 0.001, REPLAY_PROBE_MAX_PAYLOAD_BYTES="32768"
        )

        assert transport.sizes == [1, 32768, 32768, 32768]
        assert outcome.bytes_written == 1 + 3 * 32768


_ERRORS = {
    link_probe.FAILURE_TIMEOUT: lambda n: link_probe.ProbeTimeout("timed out", bytes_written=n),
    link_probe.FAILURE_CONNECTION_REFUSED: lambda n: link_probe.ProbeConnectionRefused("refused", bytes_written=n),
    link_probe.FAILURE_BAD_STATUS: lambda n: link_probe.ProbeBadStatus(500, bytes_written=n),
}

_STAGES = {"round-trip": 0, "ramp": 3, "sample": 7}


class TestFailureAtAnyStageShortCircuits:
    @pytest.mark.parametrize("reason", sorted(_ERRORS))
    @pytest.mark.parametrize("stage", sorted(_STAGES))
    def test_typed_reason_upload_count_and_bytes_written(self, stage, reason):
        index = _STAGES[stage]
        clock = _ModelClock()
        transport = _FailingAt(50000, 0.001, clock, index, _ERRORS[reason](777))

        outcome = link_probe.run_probe(_probe_settings(), transport, clock)

        assert outcome.failure_reason == reason
        assert outcome.median_kbps is None
        assert transport.attempts == index + 1, "no upload after the failing one"
        assert outcome.bytes_written == sum(DEFAULT_SEQUENCE_NEVER_REACHING_FLOOR[:index]) + 777
        assert transport.closed, "the transport is closed on a failure path"

    def test_zero_samples_uploads_nothing(self):
        outcome, transport = _run(50000, 0.001, REPLAY_PROBE_SAMPLES="0")

        assert outcome.failure_reason == link_probe.FAILURE_NO_SAMPLES
        assert transport.sizes == []
        assert outcome.bytes_written == 0
        assert transport.closed


class _RefusingAbove(_ModelledTransport):
    """The modelled link behind a sink whose cap is `cap_bytes`: a larger body is answered 413."""

    def __init__(self, bandwidth_kbps, rtt_seconds, clock, cap_bytes: int) -> None:
        super().__init__(bandwidth_kbps, rtt_seconds, clock)
        self._cap_bytes = cap_bytes
        self.refused: List[int] = []

    def upload(self, url, headers, payload, timeout):
        if len(payload) > self._cap_bytes:
            self.refused.append(len(payload))
            raise link_probe.ProbeBadStatus(413, bytes_written=len(payload))
        return super().upload(url, headers, payload, timeout)


class TestSinkCapBelowTheMaximumSettlesTheRamp:
    def test_a_refused_ramp_step_settles_one_step_below(self):
        clock = _ModelClock()
        transport = _RefusingAbove(50000, 0.001, clock, cap_bytes=262144)

        outcome = link_probe.run_probe(_probe_settings(), transport, clock)

        assert outcome.failure_reason is None
        assert outcome.payload_bytes == 262144
        assert transport.refused == [524288]
        assert transport.sizes == [1, 32768, 65536, 131072, 262144, 262144, 262144, 262144]
        assert len(outcome.samples_kbps) == 3
        assert len(outcome.ramp_samples_kbps) == 4
        assert outcome.median_kbps is not None
        assert outcome.bytes_written == sum(transport.sizes) + 524288
        assert transport.closed

    def test_a_refused_initial_payload_is_bad_status(self):
        clock = _ModelClock()
        transport = _RefusingAbove(50000, 0.001, clock, cap_bytes=1000)

        outcome = link_probe.run_probe(_probe_settings(), transport, clock)

        assert outcome.failure_reason == link_probe.FAILURE_BAD_STATUS
        assert outcome.median_kbps is None
        assert transport.sizes == [1]
        assert transport.refused == [32768]
        assert outcome.bytes_written == 1 + 32768
        assert transport.closed

    def test_a_non_413_status_on_a_ramp_step_still_fails(self):
        clock = _ModelClock()
        transport = _FailingAt(
            50000, 0.001, clock, 3, link_probe.ProbeBadStatus(500, bytes_written=131072)
        )

        outcome = link_probe.run_probe(_probe_settings(), transport, clock)

        assert outcome.failure_reason == link_probe.FAILURE_BAD_STATUS


class TestSizingRefusals:
    @pytest.mark.parametrize(
        "overrides, named",
        [
            ({"REPLAY_PROBE_MAX_PAYLOAD_BYTES": "1000"}, "1000"),
            ({"REPLAY_PROBE_MIN_SAMPLE_SECONDS": "-0.5"}, "-0.5"),
            ({"REPLAY_PROBE_PAYLOAD_BYTES": "0"}, "got 0"),
        ],
    )
    def test_invalid_sizing_is_refused_naming_the_value(self, overrides, named):
        with pytest.raises(ValidationError) as excinfo:
            _probe_settings(**overrides)
        assert named in str(excinfo.value)

    def test_defaults_construct(self):
        settings = _probe_settings()

        assert settings.probe_payload_bytes == 32768
        assert settings.probe_max_payload_bytes == 1048576
        assert settings.probe_min_sample_seconds == 1.0

    def test_a_maximum_equal_to_the_initial_payload_constructs(self):
        settings = _probe_settings(REPLAY_PROBE_MAX_PAYLOAD_BYTES="32768")

        assert settings.probe_max_payload_bytes == settings.probe_payload_bytes


class TestRampOverTheRealTransport:
    def test_every_upload_of_the_ramp_rides_one_connection(self):
        sink = _ConnectionCountingSink()
        try:
            settings = _probe_settings(REPLAY_CLOUD_INFERENCE_BASE_URL=f"http://127.0.0.1:{sink.port}")
            outcome = link_probe.run_probe(settings, link_probe.HttpxProbeTransport(), time.monotonic)
        finally:
            sink.close()

        assert outcome.failure_reason is None
        assert sink.connection_count == 1
        assert sink.request_count == 1 + len(outcome.ramp_samples_kbps) + settings.probe_samples
