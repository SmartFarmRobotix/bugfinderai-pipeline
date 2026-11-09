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
Unit tests for the inference service's registration retry.

`_register_service` used to run synchronously in `lifespan` and raise on
any failure, so a GateKeeper not yet reachable (the inference service reliably loses the
startup race against GateKeeper's own slower Django boot) crashed the service's
startup entirely. `_register_service_with_retry` now runs it as a
background task with bounded exponential backoff and unbounded retry
count.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("INFERENCE_USING_GATEKEEPER", "false")
os.environ.setdefault("INFERENCE_ORCH_PUSH_ENABLED", "false")
os.environ.setdefault("INFERENCE_STORAGE_DIR", "/tmp/inf_test_storage_regretry")

from app.config import Settings                                   # noqa: E402
from app.main import (                                            # noqa: E402
    _REGISTRATION_BACKOFF_INITIAL_SECONDS,
    _REGISTRATION_BACKOFF_MAX_SECONDS,
    _register_service_with_retry,
)


def _settings(**overrides) -> Settings:
    kwargs = dict(INFERENCE_USING_GATEKEEPER="true")
    kwargs.update(overrides)
    return Settings(**kwargs)


class TestRegistrationRetry:
    def test_returns_without_raising_once_registration_succeeds(self):
        call_count = {"n": 0}

        def _fake_register(settings):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("simulated: GateKeeper unreachable")

        with patch("app.main._register_service", side_effect=_fake_register), \
             patch("app.main.asyncio.sleep", return_value=None) as mock_sleep:
            asyncio.run(_register_service_with_retry(_settings()))

        assert call_count["n"] == 3, "must keep retrying until registration succeeds"
        assert mock_sleep.await_count == 2, "must sleep between failed attempts, not after success"

    def test_never_raises_even_after_many_consecutive_failures(self):
        """A GateKeeper that never comes back must never crash this task
        retry count is unbounded. Drive several failed attempts (via a
        mocked sleep that stops the loop only by injecting a real
        CancelledError) and confirm the only way out is that injected
        cancellation -- the underlying ConnectionError never itself
        escapes `_register_service_with_retry`."""
        call_count = {"n": 0}
        sleep_calls = {"n": 0}

        def _always_fails(settings):
            call_count["n"] += 1
            raise ConnectionError("simulated: GateKeeper permanently unreachable")

        async def _stop_after_n_sleeps(seconds):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 5:
                raise asyncio.CancelledError()

        with patch("app.main._register_service", side_effect=_always_fails), \
             patch("app.main.asyncio.sleep", side_effect=_stop_after_n_sleeps):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(_register_service_with_retry(_settings()))

        assert call_count["n"] == 5, "must have kept retrying up to the injected cancellation"

    def test_backoff_delay_doubles_and_is_capped(self):
        delays = []

        async def _record_sleep(seconds):
            delays.append(seconds)
            if len(delays) >= 6:
                raise asyncio.CancelledError()

        def _always_fails(settings):
            raise ConnectionError("simulated: GateKeeper unreachable")

        async def _run():
            with patch("app.main._register_service", side_effect=_always_fails), \
                 patch("app.main.asyncio.sleep", side_effect=_record_sleep):
                with pytest.raises(asyncio.CancelledError):
                    await _register_service_with_retry(_settings())

        asyncio.run(_run())

        assert delays[0] == _REGISTRATION_BACKOFF_INITIAL_SECONDS
        for i in range(1, len(delays)):
            assert delays[i] == min(delays[i - 1] * 2, _REGISTRATION_BACKOFF_MAX_SECONDS)
        assert max(delays) <= _REGISTRATION_BACKOFF_MAX_SECONDS
