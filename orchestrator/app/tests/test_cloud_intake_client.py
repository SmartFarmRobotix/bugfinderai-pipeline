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

"""The URL, query, body,
headers and timeout of the intake submission and of the image-pair push
against a fake of `requests.post`, and the typed errors per status and on a
connection error. The fake answers the Gatekeeper login and the proxied
call, as `requests` returns them: a status code, a `.text` and a `.json`
that raises `ValueError` on a body that is not JSON.
"""

from __future__ import annotations

import json
from typing import List

import pytest
import requests

from cloud_intake_client import (
    PairPushConnectionError,
    PairPushHTTPError,
    ReassessConnectionError,
    ReassessHTTPError,
    push_image_pair,
    submit_for_reassessment,
)


class _Settings:
    gatekeeper_base_url = "http://gk.example:8001/"
    gatekeeper_token_endpoint = "/api/login/"
    gatekeeper_timeout_seconds = 7
    rs_service_name = "reporting"
    rs_images_upload_path = "/api/v1/openagri-report/pest-session-images/"
    sync_cloud_orchestrator_service_name = "orchestrator"
    sync_call_timeout_seconds = 31
    sync_reassess_timeout_seconds = 121
    sync_gatekeeper_username = "sync-agent"
    sync_gatekeeper_password = "sync-test-secret"


class _Response:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text if body is None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class _FakeRequests:
    """Records every `requests.post` and answers the login and the proxied
    call from queued responses (an exception instance is raised)."""

    def __init__(self, monkeypatch, proxied_answer):
        self.calls: List[dict] = []
        self.proxied_answer = proxied_answer
        monkeypatch.setattr(requests, "post", self._post)

    def _post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if url.endswith("/api/login/"):
            return _Response(200, {"access": "tok-1"})
        if isinstance(self.proxied_answer, Exception):
            raise self.proxied_answer
        return self.proxied_answer


_SUBMISSION = {
    "contract_version": "1.5.0",
    "risk_input": {"session_id": "s-1"},
    "reporting_envelope": {"session_id": "s-1"},
}
_PAIR = {"contract_version": "1.5.0", "image_id": "img-1", "artifact_hash": "h"}


class TestSubmitForReassessment:
    def test_posts_the_stored_submission_with_the_marker_as_query(self, monkeypatch):
        fake = _FakeRequests(
            monkeypatch, _Response(200, {"status": "accepted", "session_id": "s-1", "state": "reported", "report_id": "r-9"})
        )

        result = submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "edge-uuid-1")

        login, call = fake.calls
        assert login["url"] == "http://gk.example:8001/api/login/"
        assert login["json"] == {"username": "sync-agent", "password": "sync-test-secret"}
        assert call["url"] == (
            "http://gk.example:8001/api/proxy/orchestrator/api/v1/sessions/"
            "?origin=edge-sync&edge_report_id=edge-uuid-1"
        )
        assert call["json"] == _SUBMISSION
        assert call["headers"] == {"Authorization": "Bearer tok-1"}
        assert call["timeout"] == 121
        assert (result.status, result.state, result.report_id) == ("accepted", "reported", "r-9")

    def test_an_answer_without_a_report_id_is_returned_as_it_is(self, monkeypatch):
        _FakeRequests(
            monkeypatch, _Response(200, {"status": "accepted", "session_id": "s-1", "state": "assessed", "report_id": None})
        )

        result = submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

        assert (result.status, result.state, result.report_id) == ("accepted", "assessed", None)

    def test_duplicate_answer(self, monkeypatch):
        _FakeRequests(
            monkeypatch, _Response(200, {"status": "duplicate", "session_id": "s-1", "state": "reported", "report_id": "r-1"})
        )

        result = submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

        assert (result.status, result.report_id) == ("duplicate", "r-1")

    @pytest.mark.parametrize("status", [401, 403, 409, 422, 500, 502])
    def test_error_status_raises_with_status_and_body(self, monkeypatch, status):
        _FakeRequests(monkeypatch, _Response(status, text="the detail"))

        with pytest.raises(ReassessHTTPError) as excinfo:
            submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

        assert excinfo.value.status == status
        assert excinfo.value.body == "the detail"

    def test_connection_error_is_typed(self, monkeypatch):
        _FakeRequests(monkeypatch, requests.ConnectionError("refused"))

        with pytest.raises(ReassessConnectionError):
            submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

    def test_timeout_is_a_connection_error(self, monkeypatch):
        _FakeRequests(monkeypatch, requests.Timeout("slow"))

        with pytest.raises(ReassessConnectionError):
            submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

    def test_failed_gatekeeper_login_is_a_connection_error(self, monkeypatch):
        def _post(url, **kwargs):
            return _Response(401, text="bad credentials")

        monkeypatch.setattr(requests, "post", _post)

        with pytest.raises(ReassessConnectionError):
            submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

    @pytest.mark.parametrize("answer", [_Response(200, text="<html>"), _Response(200, {"unexpected": 1})])
    def test_a_2xx_answer_that_is_not_an_intake_response_is_an_http_error(self, monkeypatch, answer):
        _FakeRequests(monkeypatch, answer)

        with pytest.raises(ReassessHTTPError) as excinfo:
            submit_for_reassessment(_Settings(), "s-1", _SUBMISSION, "e")

        assert excinfo.value.status == 200


class TestPushImagePair:
    def test_posts_the_pair_body_to_the_route_with_the_trailing_slash(self, monkeypatch):
        fake = _FakeRequests(monkeypatch, _Response(200, {"stored": True, "images_embedded": True}))

        result = push_image_pair(_Settings(), "s-1", _PAIR)

        _login, call = fake.calls
        assert call["url"] == (
            "http://gk.example:8001/api/proxy/reporting/api/v1/openagri-report/pest-session-images/s-1/"
        )
        assert call["json"] == _PAIR
        assert call["headers"] == {"Authorization": "Bearer tok-1"}
        assert call["timeout"] == 31
        assert (result.stored, result.images_embedded) == (True, True)

    def test_stored_without_embedding(self, monkeypatch):
        _FakeRequests(monkeypatch, _Response(200, {"stored": True, "images_embedded": False}))

        result = push_image_pair(_Settings(), "s-1", _PAIR)

        assert (result.stored, result.images_embedded) == (True, False)

    def test_an_older_answer_without_the_field_is_not_embedded(self, monkeypatch):
        _FakeRequests(monkeypatch, _Response(200, {"stored": True}))

        result = push_image_pair(_Settings(), "s-1", _PAIR)

        assert (result.stored, result.images_embedded) == (True, False)

    @pytest.mark.parametrize("answer", [_Response(200, text="not json"), _Response(200, ["stored"])])
    def test_a_malformed_answer_is_never_delivered(self, monkeypatch, answer):
        _FakeRequests(monkeypatch, answer)

        result = push_image_pair(_Settings(), "s-1", _PAIR)

        assert (result.stored, result.images_embedded) == (False, False)

    @pytest.mark.parametrize("status", [404, 422, 500])
    def test_error_status_raises_with_status_and_body(self, monkeypatch, status):
        _FakeRequests(monkeypatch, _Response(status, text="nope"))

        with pytest.raises(PairPushHTTPError) as excinfo:
            push_image_pair(_Settings(), "s-1", _PAIR)

        assert excinfo.value.status == status
        assert excinfo.value.body == "nope"

    def test_connection_error_is_typed(self, monkeypatch):
        _FakeRequests(monkeypatch, requests.ConnectionError("refused"))

        with pytest.raises(PairPushConnectionError):
            push_image_pair(_Settings(), "s-1", _PAIR)

    def test_failed_gatekeeper_login_is_a_connection_error(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda url, **kwargs: _Response(503, text="down"))

        with pytest.raises(PairPushConnectionError):
            push_image_pair(_Settings(), "s-1", _PAIR)
