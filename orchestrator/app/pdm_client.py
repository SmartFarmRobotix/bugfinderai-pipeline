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
PDM rule-assessment call via Gatekeeper.

Forwards exactly the scalar `RiskInput` structure — never the reporting
envelope, never detection internals — to PDM's
`/api/v1/risk-assessment/` endpoint (OpenAgri-PestAndDiseaseManagement),
reached through GateKeeper's service proxy the same way every other
inter-service call in this repo is: `{GK_BASE_URL}/api/proxy/{service}{path}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from config import Settings
from contract import RiskInput
from gatekeeper import get_gatekeeper_token
from logging_utils import current_request_id


class PdmCallError(RuntimeError):
    """Base class for PDM rule-assessment call failures."""


class PdmConnectionError(PdmCallError):
    """PDM (or the GateKeeper proxy in front of it) was unreachable."""


class PdmHTTPError(PdmCallError):
    """PDM (via GateKeeper) responded with a non-2xx status."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"PDM risk-assessment call failed: {status_code} {body}")
        self.status_code = status_code
        self.body = body


@dataclass
class PdmAssessmentResult:
    level: Optional[str]
    details: Optional[str]
    provenance: Dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def raw_level(self) -> Optional[str]:
        """PDM's `level` above is already score-combined
        (its own in-PDM `max(nn_verdict, weather_dominant)` — correct for
        PDM's standalone path). `raw_level` is
        the ADDITIONAL, pre-score weather-rule-only band PDM now also
        reports (`provenance["weather_rule_level_raw"]`) — this is what
        this orchestrator's central fusion (`fusion.fuse`) must consume
        instead of `level`, or the vision score gets counted twice. `None`
        when PDM ran degraded (no weather-rule engine, score-only
        fallback) — a genuinely absent raw component, not a placeholder."""
        return self.provenance.get("weather_rule_level_raw")


def assess_via_pdm(settings: Settings, risk_input: RiskInput) -> PdmAssessmentResult:
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.pdm_service_name}{settings.pdm_risk_assessment_path}"
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
            json=risk_input.model_dump(mode="json"),
            timeout=settings.pdm_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise PdmConnectionError(f"PDM/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise PdmHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise PdmHTTPError(response.status_code, "non-JSON response") from exc

    risk_assessment = body.get("risk_assessment") or {}
    return PdmAssessmentResult(
        level=risk_assessment.get("level"),
        details=risk_assessment.get("details"),
        provenance=risk_assessment.get("provenance") or {},
        errors=body.get("errors") or [],
    )


@dataclass
class PdmForecastResult:
    """`days` = [{"date", "level" (lower-case), "rules_triggered"}] in date
    order; `reason` is PDM's named reason (e.g. "no_pest_models_configured")
    when it evaluated nothing, and then `days` is empty."""

    days: list[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


def _post_forecast_body(settings: Settings, body: Dict[str, Any]) -> PdmForecastResult:
    """One `POST /api/v1/risk-assessment/forecast/` through GateKeeper. Raises
    `PdmCallError` (or a subclass) on any failure."""
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.pdm_service_name}{settings.pdm_forecast_risk_path}"
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
            url, headers=headers, json=body, timeout=settings.pdm_call_timeout_seconds
        )
    except requests.RequestException as exc:
        raise PdmConnectionError(f"PDM/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise PdmHTTPError(response.status_code, response.text)

    try:
        answer = response.json()
    except ValueError as exc:
        raise PdmHTTPError(response.status_code, "non-JSON response") from exc

    if not isinstance(answer, dict):
        raise PdmHTTPError(response.status_code, "unexpected forecast response shape")
    return PdmForecastResult(days=answer.get("days") or [], reason=answer.get("reason"))


def assess_forecast_via_pdm(
    settings: Settings, crop: str, days: list[Dict[str, Any]]
) -> PdmForecastResult:
    """PDM's rule evaluation of daily
    forecast values (`POST /api/v1/risk-assessment/forecast/`) through
    GateKeeper. `days` are the outlook rows (`temp_max_C`, `temp_min_C`,
    `humidity_pct`, `precipitation_mm`), translated here to PDM's request
    keys. Raises `PdmCallError` (or a subclass) on any failure.
    """
    body = {
        "crop": crop,
        "days": [
            {
                "date": day["date"],
                "temp_max": day.get("temp_max_C"),
                "temp_min": day.get("temp_min_C"),
                "humidity": day.get("humidity_pct"),
                "precipitation": day.get("precipitation_mm"),
            }
            for day in days
        ],
    }
    return _post_forecast_body(settings, body)


def assess_forecast_hours_via_pdm(
    settings: Settings, crop: str, days: list[Dict[str, Any]]
) -> PdmForecastResult:
    """
    The hourly form of the same route: `days` are
    `{"date", "hours": [{"time", "temperature_2m", "relative_humidity_2m",
    "precipitation"}]}`, sent as they are; PDM evaluates each day's hours
    with the per-rule check its session assessment applies to an hourly
    window. Raises `PdmCallError` (or a subclass) on any failure.
    """
    body = {
        "crop": crop,
        "days": [{"date": day["date"], "hours": day["hours"]} for day in days],
    }
    return _post_forecast_body(settings, body)
