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
Uploads an already-rendered edge
report PDF to ReportingService's additive edge-ingest endpoint, through
GateKeeper's proxy, using the sync-scoped token. Distinct from
`reporting_client.py`'s `push_report`: that sends JSON metadata to the
cloud/mixed report-generation endpoint; this sends the PDF bytes
themselves (multipart) to an endpoint that stores an already-finished
report rather than rendering one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from config import Settings
from gatekeeper import get_gatekeeper_token_with_credentials


class ReportSyncCallError(RuntimeError):
    """Base class for edge-report-sync upload failures."""


class ReportSyncConnectionError(ReportSyncCallError):
    """ReportingService (or the GateKeeper proxy in front of it) was unreachable."""


class ReportSyncHTTPError(ReportSyncCallError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"ReportingService edge-report ingest failed: {status_code} {body}")
        self.status_code = status_code
        self.body = body


@dataclass
class ReportSyncResult:
    report_id: str
    deduplicated: bool


def push_synced_report(
    settings: Settings,
    session_id: str,
    content_hash: str,
    pdf_path: str,
) -> ReportSyncResult:
    token = get_gatekeeper_token_with_credentials(
        settings, settings.sync_gatekeeper_username, settings.sync_gatekeeper_password
    )
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.rs_service_name}{settings.rs_edge_report_ingest_path}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    pdf_bytes = Path(pdf_path).read_bytes()
    files = {"pdf": (f"{session_id}.pdf", pdf_bytes, "application/pdf")}
    data = {"session_id": session_id, "content_hash": content_hash}

    try:
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=settings.sync_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise ReportSyncConnectionError(f"ReportingService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise ReportSyncHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise ReportSyncHTTPError(response.status_code, "non-JSON response") from exc

    report_id = body.get("uuid")
    if not report_id:
        raise ReportSyncHTTPError(response.status_code, "response missing 'uuid'")
    return ReportSyncResult(report_id=report_id, deduplicated=bool(body.get("deduplicated")))
