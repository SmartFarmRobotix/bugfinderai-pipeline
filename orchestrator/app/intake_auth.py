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
Intake authentication.

Two mechanisms, selected by `settings.deployment_profile`:

cloud/mixed: the caller must present a Gatekeeper-issued bearer
  token, verified with a timeout-bounded call to GK's `validate_token`
  endpoint — the same pattern the Reporting service uses.
edge: the caller must present the correct local shared secret in
  `X-Edge-Shared-Secret`. This branch makes zero network calls of any
  kind — in particular, never to Gatekeeper (see
  `test_intake_auth.py`'s assertion that `requests.post` is never invoked
  on this path).

Both branches run as a FastAPI dependency, so a rejected request never
reaches `intake_submission` — no session record is created.
"""

from __future__ import annotations

import hmac
import logging

import requests
from fastapi import Header, HTTPException, status

from config import DeploymentProfile, Settings, get_settings

logger = logging.getLogger("orchestrator.intake_auth")


def _validate_gatekeeper_token(settings: Settings, token: str) -> None:
    url = f"{settings.gatekeeper_base_url.rstrip('/')}{settings.gatekeeper_validate_path}"
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"token": token, "token_type": "access"},
            timeout=settings.gatekeeper_timeout_seconds,
        )
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timed out validating token with GateKeeper.",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GateKeeper is unreachable; could not validate token.",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired GateKeeper token.",
        )


def authenticate_intake(
    authorization: str | None = Header(default=None),
    x_edge_shared_secret: str | None = Header(default=None, alias="X-Edge-Shared-Secret"),
) -> None:
    settings = get_settings()

    if settings.deployment_profile is DeploymentProfile.EDGE:
        expected = settings.edge_shared_secret or ""
        presented = x_edge_shared_secret or ""
        if not x_edge_shared_secret or not hmac.compare_digest(presented, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or incorrect edge shared secret.",
            )
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    _validate_gatekeeper_token(settings, token)
