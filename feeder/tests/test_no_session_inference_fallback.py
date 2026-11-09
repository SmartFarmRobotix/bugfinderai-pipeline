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
With its configured inference address unreachable, SessionFeederClient fails and reports the failure, and
no request reaches any other host. submit_session dispatches to exactly one of _upload_session or
_register_path_session depending on the fixed deployment profile, and both build their URL from the single
settings.inference_base_url (REPLAY_INFERENCE_BASE_URL); there is no second address, list or fallback
ordering in app/config.py or app/client.py. The CLOUD profile (http_upload) and the MIXED/EDGE profiles
(path) are both covered. _UnreachableRecordingTransport raises on every request and records its URL, so a
future fallback would show up as a second recorded request against a different host.
"""

from __future__ import annotations

import httpx
import pytest

from app.client import SessionFeederClient
from app.config import DeploymentProfile
from app.store import load_replay_session
from tests.conftest import make_settings


class _UnreachableRecordingTransport(httpx.BaseTransport):
    """Fake httpx transport that fails every request like a genuinely
    unreachable host (connection refused) -- and records the URL of every
    request it was asked to send, so a test can assert the full set of
    hosts a call site ever attempted."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise httpx.ConnectError(f"refused: {request.url}", request=request)


class TestNoAutomaticSessionInferenceFallback:
    def test_failed_call_is_reported_not_swallowed_silently(self, fixtures_dir, tmp_path):
        """(a) the call raises/reports failure: the unreachable-host
        exception is caught and logged, and submit_session reports failure
        via result.success=False / result.error set -- it never fabricates
        a substitute session and never lets the exception escape uncaught
        either."""
        session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
        transport = _UnreachableRecordingTransport()
        settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD, REPLAY_DATA_ROOT=str(tmp_path))

        with SessionFeederClient(settings, transport=transport) as client:
            result = client.submit_session(session)

        assert result.success is False
        assert "HTTP request failed" in result.error
        assert len(transport.requests) == 1

    def test_no_second_host_is_ever_contacted_on_failure(self, fixtures_dir, tmp_path):
        """(b) no second address is contacted: every request this client
        ever sends targets the one configured inference_base_url, never any
        other host -- fails if a fallback URL is ever introduced. Submits
        twice to prove this holds across repeated failed attempts too, not
        just the first."""
        session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
        transport = _UnreachableRecordingTransport()
        configured_base = "http://inference.test"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=configured_base,
        )

        with SessionFeederClient(settings, transport=transport) as client:
            client.submit_session(session)
            client.submit_session(session)

        assert len(transport.requests) == 2
        for req in transport.requests:
            assert str(req.url).startswith(configured_base + "/"), (
                f"request to {req.url!r} did not target the configured inference "
                f"address {configured_base!r} -- a fallback host was contacted"
            )

    @pytest.mark.parametrize("profile", [DeploymentProfile.MIXED, DeploymentProfile.EDGE])
    def test_path_mode_also_fails_closed_and_targets_only_configured_host(self, fixtures_dir, tmp_path, profile):
        """The same single-address property holds for the PATH ingestion
        mode (mixed/edge deployments) as for HTTP upload (cloud) above:
        _register_path_session builds its URL from the same
        inference_base_url, with no separate config knob of its own."""
        session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
        transport = _UnreachableRecordingTransport()
        configured_base = "http://inference.test"
        settings = make_settings(
            REPLAY_DEPLOYMENT_PROFILE=profile,
            REPLAY_DATA_ROOT=str(tmp_path),
            REPLAY_INFERENCE_BASE_URL=configured_base,
        )

        with SessionFeederClient(settings, transport=transport) as client:
            result = client.submit_session(session)

        assert result.success is False
        assert len(transport.requests) == 1
        assert str(transport.requests[0].url).startswith(configured_base + "/")

    def test_unreachable_host_never_raises_out_of_submit_session(self, fixtures_dir, tmp_path):
        """Not a fallback-adjacent case by itself, but guards the same
        property from the other direction: submit_session must not let a
        connection failure crash the caller -- httpx.HTTPError is caught
        inside the client, exactly like the report-image path the orchestrator's
        test_no_inference_fallback.py
        already proves for a different call site."""
        session = load_replay_session(fixtures_dir / "minimal_conforming", "sess-min")
        transport = _UnreachableRecordingTransport()
        settings = make_settings(REPLAY_DEPLOYMENT_PROFILE=DeploymentProfile.CLOUD, REPLAY_DATA_ROOT=str(tmp_path))

        with SessionFeederClient(settings, transport=transport) as client:
            result = client.submit_session(session)

        assert result.status_code == 0
