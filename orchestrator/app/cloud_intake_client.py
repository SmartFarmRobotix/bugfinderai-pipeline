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

"""The two calls the sync
worker makes after a synced report's PDF upload, both through the
Gatekeeper with the sync identity's token:

`submit_for_reassessment`: the stored edge submission to the cloud
  orchestrator's intake, marked as an edge re-assessment by the query
  parameters `origin=edge-sync&edge_report_id=<uuid>` (never a field of the
  submission, so the contract version stays as it is);
`push_image_pair`: the capped best-image pair (the pair file written at
  render time) to the Reporting service's pair route, which re-renders the
  report with the images once the intake has created its index row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, urlencode

import requests

from config import Settings
from gatekeeper import GatekeeperAuthError, get_gatekeeper_token_with_credentials


class ReassessCallError(RuntimeError):
    """Base class for the re-assessment step's call failures."""


class ReassessConnectionError(ReassessCallError):
    """The Gatekeeper, or the service behind it, was unreachable."""


class ReassessHTTPError(ReassessCallError):
    def __init__(self, status: int, body: str):
        super().__init__(f"cloud intake answered {status}: {body}")
        self.status = status
        self.body = body


class PairPushError(RuntimeError):
    """Base class for the image-pair push's failures."""


class PairPushConnectionError(PairPushError):
    """The Gatekeeper, or the Reporting service behind it, was unreachable."""


class PairPushHTTPError(PairPushError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Reporting service answered the image pair {status}: {body}")
        self.status = status
        self.body = body


@dataclass
class ReassessResult:
    status: str
    state: Optional[str]
    report_id: Optional[str]


@dataclass
class PairPushResult:
    stored: bool
    images_embedded: bool


def _bearer_headers(settings: Settings, error_type: type) -> dict:
    """The sync identity's bearer header; a failed Gatekeeper login is raised
    as `error_type` (a connection-class error, which the worker retries)."""
    try:
        token = get_gatekeeper_token_with_credentials(
            settings, settings.sync_gatekeeper_username, settings.sync_gatekeeper_password
        )
    except GatekeeperAuthError as exc:
        raise error_type(f"GateKeeper login for the sync identity failed: {exc}") from exc
    return {"Authorization": f"Bearer {token}"}


def submit_for_reassessment(
    settings: Settings,
    session_id: str,
    submission_json: dict,
    edge_report_id: str,
) -> ReassessResult:
    """POSTs `submission_json` (`{contract_version, risk_input,
    reporting_envelope}`) to the cloud orchestrator's session intake. The
    answer's `status`, `state` and `report_id` are returned as the intake
    gave them; judging them is the worker's job."""
    headers = _bearer_headers(settings, ReassessConnectionError)
    query = urlencode({"origin": "edge-sync", "edge_report_id": edge_report_id})
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.sync_cloud_orchestrator_service_name}/api/v1/sessions/?{query}"
    )
    try:
        response = requests.post(
            url,
            headers=headers,
            json=submission_json,
            timeout=settings.sync_reassess_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise ReassessConnectionError(f"cloud orchestrator/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise ReassessHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise ReassessHTTPError(response.status_code, "non-JSON response") from exc
    if not isinstance(body, dict) or not body.get("status"):
        raise ReassessHTTPError(response.status_code, "response missing 'status'")
    return ReassessResult(status=body["status"], state=body.get("state"), report_id=body.get("report_id"))


def push_image_pair(settings: Settings, session_id: str, pair: dict) -> PairPushResult:
    """POSTs the pair file's body to the Reporting service's pair route. The
    trailing slash after the session id is the Gatekeeper's registered
    pattern (the inference service's own push does the same). A 2xx answer
    that is not a JSON object counts as stored=False, images_embedded=False,
    never as delivered."""
    headers = _bearer_headers(settings, PairPushConnectionError)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.rs_service_name}{settings.rs_images_upload_path}{quote(session_id, safe='')}/"
    )
    try:
        response = requests.post(url, headers=headers, json=pair, timeout=settings.sync_call_timeout_seconds)
    except requests.RequestException as exc:
        raise PairPushConnectionError(f"ReportingService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise PairPushHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError:
        return PairPushResult(stored=False, images_embedded=False)
    if not isinstance(body, dict):
        return PairPushResult(stored=False, images_embedded=False)
    return PairPushResult(stored=body.get("stored") is True, images_embedded=body.get("images_embedded") is True)
