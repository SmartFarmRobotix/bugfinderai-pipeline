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
GateKeeper login client.

Mirrors the inference service's token-fetch pattern (inference/app/main.py) rather than PDM's caller-token-forwarding pattern: the
orchestrator originates its own outbound call to PDM, so it must acquire
its own token, not forward a caller's.
"""

from __future__ import annotations

import requests

from config import Settings


class GatekeeperAuthError(RuntimeError):
    """GateKeeper login failed (unreachable, non-2xx, or no token in body)."""


def get_gatekeeper_token(settings: Settings) -> str:
    return get_gatekeeper_token_with_credentials(
        settings, settings.gatekeeper_username, settings.gatekeeper_password
    )


def get_gatekeeper_token_with_credentials(settings: Settings, username: str, password: str) -> str:
    """Same login call as `get_gatekeeper_token`, but with an explicit
    credential pair rather than `settings.gatekeeper_username`/`password` --
    the sync identity is deliberately
    a distinct credential pair from this service's general-purpose
    Gatekeeper login, so it must not read `settings.gatekeeper_username`/`password`."""
    url = f"{settings.gatekeeper_base_url.rstrip('/')}{settings.gatekeeper_token_endpoint}"
    try:
        response = requests.post(
            url,
            json={"username": username, "password": password},
            timeout=settings.gatekeeper_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise GatekeeperAuthError(f"GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise GatekeeperAuthError(
            f"GateKeeper auth failed: {response.status_code} {response.text}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise GatekeeperAuthError("GateKeeper auth response was not JSON") from exc

    token = body.get("access") or body.get("access_token") or body.get("token")
    if not token:
        raise GatekeeperAuthError("GateKeeper auth response missing token")
    return token
