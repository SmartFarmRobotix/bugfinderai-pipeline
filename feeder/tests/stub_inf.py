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
A stub of the inference service, used by unit tests:
implements just enough of the real wire shape (/healthz, POST
/api/v1/sessions, POST /api/v1/sessions/path, POST
/api/v1/sessions/{id}/push) to exercise SessionFeederClient's real
request-building logic without a live server. Records every request it
receives so tests can assert on dispatch (which endpoint was/wasn't called)
and on payload content (multipart fields, JSON body).
"""

from __future__ import annotations

import email
import json
from dataclasses import dataclass, field

import httpx


@dataclass
class RecordedRequest:
    method: str
    path: str
    form_fields: dict[str, str]
    files: dict[str, bytes]
    json_body: dict
    host: str = ""


def _parse_multipart(request: httpx.Request) -> tuple[dict[str, str], dict[str, bytes]]:
    content_type = request.headers.get("content-type", "")
    header_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    msg = email.message_from_bytes(header_bytes + request.content)

    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    if not msg.is_multipart():
        return fields, files
    for part in msg.get_payload():
        disposition = part.get("Content-Disposition", "")
        name = None
        filename = None
        for piece in disposition.split(";"):
            piece = piece.strip()
            if piece.startswith("name="):
                name = piece.split("=", 1)[1].strip('"')
            elif piece.startswith("filename="):
                filename = piece.split("=", 1)[1].strip('"')
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[filename] = payload
        elif name:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


@dataclass
class StubINF:
    """In-memory stand-in for the inference service, wired via httpx.MockTransport."""

    healthy: bool = True
    next_session_id: str = "stub-session-001"
    next_best_image_id: str = "stub-image-001"
    next_score: float = 0.0
    push_target_url: str = "http://gatekeeper/api/proxy/pdm/api/v1/inference-session/"
    push_response_body: str = '{"report_id": "report-001", "session_id": "pdm-session-001"}'
    requests: list[RecordedRequest] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if request.method == "GET" and path == "/healthz":
            if self.healthy:
                return httpx.Response(200, json={"status": "ready"})
            return httpx.Response(503, json={"status": "not_ready"})

        if request.method == "POST" and path == "/api/v1/sessions":
            fields, files = _parse_multipart(request)
            self.requests.append(RecordedRequest("POST", path, fields, files, {}, host=request.url.host))
            return httpx.Response(
                200,
                json={
                    "session_id": fields.get("session_id") or self.next_session_id,
                    "best_image_id": self.next_best_image_id,
                    "infestation_score": self.next_score,
                },
            )

        if request.method == "POST" and path == "/api/v1/sessions/path":
            body = json.loads(request.content)
            self.requests.append(RecordedRequest("POST", path, {}, {}, body, host=request.url.host))
            return httpx.Response(
                200,
                json={
                    "session_id": body.get("session_id") or self.next_session_id,
                    "best_image_id": self.next_best_image_id,
                    "infestation_score": self.next_score,
                },
            )

        if request.method == "POST" and path.startswith("/api/v1/sessions/") and path.endswith("/push"):
            self.requests.append(RecordedRequest("POST", path, {}, {}, {}, host=request.url.host))
            return httpx.Response(
                200,
                json={
                    "pushed": True,
                    "target_url": self.push_target_url,
                    "response_body": self.push_response_body,
                },
            )

        return httpx.Response(404, json={"detail": f"no stub route for {request.method} {path}"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def requests_to(self, path: str) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == path]
