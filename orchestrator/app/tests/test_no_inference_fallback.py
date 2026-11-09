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

"""With the orchestrator's configured inference address unreachable, its inference call fails
and is reported as a failure, and no request reaches any other inference host.

`edge_report._edge_image_source` (settings.edge_inference_base_url, alias
ORCH_EDGE_INFERENCE_BASE_URL) is this service's ONLY outbound HTTP call to
an inference host anywhere in the codebase (grep confirms: every other
`requests.*` call in this service targets GateKeeper, PDM, Weather Data, or
Reporting). What this test
proves is narrower and unconditionally true: this client makes exactly one
attempt, at exactly the configured address, and a failed attempt is
reported (logged, `None` returned) rather than silently retried elsewhere.
There is no code path here that could reach a second host even if the
first is unreachable -- the absence of a fallback list/loop, not merely its
current disuse.

Fail-proving: `_CallRecordingUnreachable.__call__` raises on every
invocation, so a hypothetical future fallback (`except: requests.get(OTHER_URL)`)
would produce a second recorded call with a different host and fail
`test_no_second_host_is_ever_contacted_on_failure`.
"""

from __future__ import annotations

import requests

from edge_report import _edge_image_source


class _Settings:
    def __init__(self, base_url: str, api_key: str = "test-key", timeout: int = 5):
        self.edge_inference_base_url = base_url
        self.edge_inference_api_key = api_key
        self.edge_inference_call_timeout_seconds = timeout


class _CallRecordingUnreachable:
    """Fake for `requests.get` that always fails, like a genuinely
    unreachable host (connection refused) -- and records every URL it was
    asked to contact, so a test can assert the full set of hosts a call
    site ever attempted."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, url, headers=None, timeout=None):
        self.calls.append(url)
        raise requests.exceptions.ConnectionError(f"refused: {url}")


class TestNoAutomaticInferenceFallback:
    def test_failed_call_is_reported_not_swallowed_silently(self, monkeypatch):
        """(a) the call raises/reports failure: `requests.get` raising is
        caught and logged, and the fetch reports failure by returning
        `None` -- it never fabricates a substitute result and never lets
        the exception escape uncaught either (which would crash report
        generation entirely, a different and worse failure mode than the
        one this client is designed for: degrade the optional image embed,
        keep the report)."""
        recorder = _CallRecordingUnreachable()
        monkeypatch.setattr(requests, "get", recorder)

        settings = _Settings("http://127.0.0.1:1")
        fetch = _edge_image_source(settings, "some-image-id")

        result = fetch("clean")

        assert result is None
        assert len(recorder.calls) == 1

    def test_no_second_host_is_ever_contacted_on_failure(self, monkeypatch):
        """(b) no second address is contacted: every call this client
        makes targets the one configured `edge_inference_base_url`, never
        any other host -- fails if a fallback URL is ever introduced."""
        recorder = _CallRecordingUnreachable()
        monkeypatch.setattr(requests, "get", recorder)

        configured_base = "http://127.0.0.1:1"
        settings = _Settings(configured_base)
        fetch = _edge_image_source(settings, "some-image-id")

        fetch("clean")
        fetch("annotated")

        assert len(recorder.calls) == 2
        for called_url in recorder.calls:
            assert called_url.startswith(configured_base + "/"), (
                f"call to {called_url!r} did not target the configured "
                f"inference address {configured_base!r} -- a fallback host "
                "was contacted"
            )

    def test_unconfigured_image_id_makes_no_call_at_all(self, monkeypatch):
        """Not a fallback-adjacent case by itself, but guards the same
        property from the other direction: with nothing to fetch, this
        client must not "helpfully" try some other source -- it must not
        call out at all."""
        recorder = _CallRecordingUnreachable()
        monkeypatch.setattr(requests, "get", recorder)

        settings = _Settings("http://127.0.0.1:1")
        fetch = _edge_image_source(settings, "")

        result = fetch("clean")

        assert result is None
        assert recorder.calls == []
