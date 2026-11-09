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
GateKeeper self-registration.

Registers the orchestrator's own `/api/` routes with the Gatekeeper at startup, so callers such as
the inference service's push reach it at `/api/proxy/orchestrator/...`; without it the Gatekeeper's
proxy answers 404.

Mirrors PDM's registration pattern: synchronous, best-effort,
at lifespan startup — never blocks or fails startup on a slow/unreachable
GateKeeper (matches this service's existing degraded-readiness posture,
`/ready`). Introspects `app.routes` directly (every route currently
starting with `/api/`) rather than a maintained per-route list, so a
future query endpoint is registered
automatically without a second edit here.
"""

from __future__ import annotations

import logging

import requests
from requests.exceptions import RequestException

from config import Settings

logger = logging.getLogger("orchestrator.gk_registration")


def register_apis_to_gatekeeper(settings: Settings, app) -> None:
    try:
        login_response = requests.post(
            url=settings.gatekeeper_base_url.rstrip("/") + settings.gatekeeper_token_endpoint,
            headers={"Content-Type": "application/json"},
            json={"username": settings.gatekeeper_username, "password": settings.gatekeeper_password},
            timeout=settings.gatekeeper_timeout_seconds,
        )
    except RequestException as exc:
        logger.warning("GK registration: login failed (%s); continuing startup unregistered", exc)
        return

    try:
        body = login_response.json()
        access = body["access"]
        refresh = body["refresh"]
    except (ValueError, KeyError) as exc:
        logger.warning("GK registration: could not parse login response (%s); continuing startup unregistered", exc)
        return

    base_url = f"http://{settings.service_name}:{settings.port}/"
    registered = 0
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods or not path.startswith("/api/"):
            continue
        try:
            response = requests.post(
                url=settings.gatekeeper_base_url.rstrip("/") + "/api/register_service/",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {access}"},
                json={
                    "base_url": base_url,
                    "service_name": settings.service_name,
                    "endpoint": path.lstrip("/"),
                    "methods": list(methods),
                },
                timeout=settings.gatekeeper_timeout_seconds,
            )
            if str(response.status_code)[0] == "2":
                registered += 1
            else:
                logger.warning("GK registration: %s failed with status %s", path, response.status_code)
        except RequestException as exc:
            logger.warning("GK registration: %s failed: %s", path, exc)

    logger.info("GK registration: %d/%d API routes registered under service_name=%s", registered, len(
        [r for r in app.routes if getattr(r, "path", "").startswith("/api/")]
    ), settings.service_name)

    try:
        requests.post(
            url=settings.gatekeeper_base_url.rstrip("/") + "/api/logout/",
            headers={"Content-Type": "application/json"},
            json={"refresh": refresh},
            timeout=settings.gatekeeper_timeout_seconds,
        )
    except RequestException:
        pass
