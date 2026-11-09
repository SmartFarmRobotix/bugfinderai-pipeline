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
Report triggering: pushes the
composed payload (`report_payload.compose_report_payload`) to
ReportingService's `/api/v1/openagri-report/pest-session-report/` through
GateKeeper's service proxy -- the same call shape `pdm_client.py`/
`ws_client.py` already use for their own outbound calls.

Also `fetch_report_status`: a best-effort read of ReportingService's by-session status
route, used to answer whether a cloud report still awaits its device
image pair. Unlike `push_report`, this call never raises into its
caller -- a query-time read is worth making only when it can fail
silently into "unknown".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from config import Settings
from gatekeeper import get_gatekeeper_token
from logging_utils import current_request_id

logger = logging.getLogger("orchestrator.reporting_client")


class ReportingCallError(RuntimeError):
    """Base class for ReportingService report-push failures."""


class ReportingConnectionError(ReportingCallError):
    """ReportingService (or the GateKeeper proxy in front of it) was unreachable."""


class ReportingHTTPError(ReportingCallError):
    """ReportingService (via GateKeeper) responded with a non-2xx status,
    a non-JSON body, or a body missing the expected `uuid`."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"ReportingService report push failed: {status_code} {body}")
        self.status_code = status_code
        self.body = body


@dataclass
class ReportPushResult:
    report_id: str


def push_report(settings: Settings, payload: Dict[str, Any]) -> ReportPushResult:
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.rs_service_name}{settings.rs_pest_session_report_path}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    request_id = current_request_id()
    if request_id is not None:
        headers["X-Request-ID"] = request_id

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.rs_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise ReportingConnectionError(f"ReportingService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise ReportingHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise ReportingHTTPError(response.status_code, "non-JSON response") from exc

    report_id = body.get("uuid")
    if not report_id:
        raise ReportingHTTPError(response.status_code, "response missing 'uuid'")
    return ReportPushResult(report_id=report_id)


def fetch_report_status(settings: Settings, session_id: str) -> Optional[bool]:
    """ReportingService's by-session status route through GateKeeper:
    `images_pending` on a 200, `None` on any failure (unreachable,
    non-2xx, non-JSON, or a body missing the field) -- never raises, so a
    query-time caller can treat the result as "unknown" rather than fail
    the whole session read."""
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.rs_service_name}{settings.rs_report_status_path.format(session_id=session_id)}"
    )
    headers = {}
    request_id = current_request_id()
    if request_id is not None:
        headers["X-Request-ID"] = request_id

    try:
        token = get_gatekeeper_token(settings)
        headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=settings.rs_call_timeout_seconds)
        if not (200 <= response.status_code < 300):
            raise ReportingHTTPError(response.status_code, response.text)
        body = response.json()
        images_pending = body["images_pending"]
        if not isinstance(images_pending, bool):
            raise ValueError(f"images_pending is not a bool: {images_pending!r}")
        return images_pending
    except Exception as exc:  # noqa: BLE001 - best-effort read, never raises
        logger.info("report status fetch failed for session_id=%s: %s", session_id, exc)
        return None
