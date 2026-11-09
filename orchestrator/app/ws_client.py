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
WeatherService hourly-history call via Gatekeeper: the orchestrator's own outbound fetch of the same
weather window PDM already used internally for its rule assessment, so the
fuzzy engine sees the identical weather-window reference rather than an independently-derived one.

Reaches WeatherService's `POST /api/v1/history/hourly/`
(`OpenAgri-WeatherService/src/api/api_v1/endpoints/history.py`) through
GateKeeper's service proxy, the same way every other inter-service call in
this repo is (`pdm_client.py`). Request field names (`lat`/`lon`/`start`/
`end`/`variables`/`radius_km`) and the GK proxy service segment
(`weather_data`) match `OpenAgri-PestAndDiseaseManagement`'s own
`fetch_weather_data` (`app/utils/wdutils.py`) verbatim -- the only other
verified caller of this API in this codebase. The raw response body is
returned as-is, ready for `weather_adapter.ws_history_to_weather_dataframe`.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import requests

from config import Settings
from gatekeeper import get_gatekeeper_token
from logging_utils import current_request_id

REQUIRED_VARIABLES: List[str] = ["temperature_2m", "relative_humidity_2m", "precipitation"]


class WeatherServiceCallError(RuntimeError):
    """Base class for Weather-service history call failures."""


class WeatherServiceConnectionError(WeatherServiceCallError):
    """The Weather service (or the GateKeeper proxy in front of it) was unreachable."""


class WeatherServiceHTTPError(WeatherServiceCallError):
    """The Weather service (via GateKeeper) responded with a non-2xx status."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"WeatherService history call failed: {status_code} {body}")
        self.status_code = status_code
        self.body = body


def _parse_iso_date(value: str) -> datetime.date:
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise WeatherServiceCallError(f"invalid ISO-8601 date/time: {value!r}") from exc


def fetch_hourly_history(
    settings: Settings,
    latitude: float,
    longitude: float,
    time_start: str,
    time_end: str,
) -> Dict[str, Any]:
    """Fetch the hourly Weather-service history window as a raw response dict. Raises
    `WeatherServiceCallError` (or a subclass) or `gatekeeper.GatekeeperAuthError`
    on any failure -- never fabricates a result."""
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.ws_service_name}{settings.ws_history_hourly_path}"
    )
    body = {
        "lat": latitude,
        "lon": longitude,
        "start": _parse_iso_date(time_start).isoformat(),
        "end": _parse_iso_date(time_end).isoformat(),
        "variables": REQUIRED_VARIABLES,
        "radius_km": settings.ws_history_radius_km,
    }

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
            json=body,
            timeout=settings.ws_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise WeatherServiceConnectionError(f"WeatherService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise WeatherServiceHTTPError(response.status_code, response.text)

    try:
        return response.json()
    except ValueError as exc:
        raise WeatherServiceHTTPError(response.status_code, "non-JSON response") from exc


def fetch_hourly_forecast(
    settings: Settings,
    latitude: float,
    longitude: float,
    days: int = 5,
) -> Dict[str, Any]:
    """Fetch the Weather
    service's keyless hourly forecast (`POST /api/v1/forecast/hourly/`,
    `days` days ahead, wind included) as a raw response dict, the same
    shape `fetch_hourly_history` returns. Raises `WeatherServiceCallError`
    (or a subclass) or `gatekeeper.GatekeeperAuthError` on any failure,
    including a response without a usable `data` list -- never fabricates
    a result."""
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.ws_service_name}{settings.ws_forecast_hourly_path}"
    )
    body = {
        "lat": latitude,
        "lon": longitude,
        "days": days,
        "variables": REQUIRED_VARIABLES + ["wind_speed_10m"],
    }

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
            json=body,
            timeout=settings.ws_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise WeatherServiceConnectionError(f"WeatherService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise WeatherServiceHTTPError(response.status_code, response.text)

    try:
        answer = response.json()
    except ValueError as exc:
        raise WeatherServiceHTTPError(response.status_code, "non-JSON response") from exc

    rows = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or not rows:
        raise WeatherServiceHTTPError(response.status_code, "forecast response carries no hourly rows")

    return answer


_FORECAST_FIELD_MAP = {
    "temp_max": "temp_max_C",
    "temp_min": "temp_min_C",
    "humidity": "humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}


def fetch_daily_forecast(
    settings: Settings,
    latitude: float,
    longitude: float,
    days: int = 5,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """The Weather service's daily
    forecast (`POST /api/v1/forecast/daily/`, up to 5 days) through
    GateKeeper, mapped to the outlook row keys.

    Returns `{"source", "fetched_at" (UTC ISO), "location", "days": [{"date",
    "temp_max_C", "temp_min_C", "humidity_pct", "precipitation_mm",
    "wind_speed_ms"}]}`. Raises `WeatherServiceCallError` (or a subclass) or
    `gatekeeper.GatekeeperAuthError` on any failure, including a response
    without a usable `data` list -- never fabricates a forecast.
    """
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.ws_service_name}{settings.ws_forecast_daily_path}"
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
            json={"lat": latitude, "lon": longitude, "days": days},
            timeout=settings.ws_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise WeatherServiceConnectionError(f"WeatherService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise WeatherServiceHTTPError(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise WeatherServiceHTTPError(response.status_code, "non-JSON response") from exc

    rows = body.get("data") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not rows:
        raise WeatherServiceHTTPError(response.status_code, "forecast response carries no daily rows")

    mapped: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("date"):
            raise WeatherServiceHTTPError(response.status_code, "forecast row without a date")
        entry: Dict[str, Any] = {"date": str(row["date"])}
        for source_key, target_key in _FORECAST_FIELD_MAP.items():
            entry[target_key] = row.get(source_key)
        mapped.append(entry)

    fetched_at = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc)
    return {
        "source": str(body.get("source") or "unknown"),
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "location": body.get("location") or {"lat": latitude, "lon": longitude},
        "days": mapped,
    }


_HISTORY_FIELD_MAP = {
    "temperature_2m_max": "temp_max_C",
    "temperature_2m_min": "temp_min_C",
    "relative_humidity_2m_mean": "humidity_pct",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_max": "wind_speed_ms",
}
DAILY_HISTORY_VARIABLES: List[str] = list(_HISTORY_FIELD_MAP)


def fetch_daily_history(
    settings: Settings,
    latitude: float,
    longitude: float,
    start: datetime.date,
    end: datetime.date,
    variables: List[str],
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """The Weather service's real daily
    weather for `start`..`end` (`POST /api/v1/history/daily/`) through
    GateKeeper, mapped to the same outlook rows `fetch_daily_forecast`
    returns.

    Returns `{"source", "fetched_at" (UTC ISO), "location", "days": [{"date",
    "temp_max_C", "temp_min_C", "humidity_pct", "precipitation_mm",
    "wind_speed_ms"}]}`; a variable not requested, or absent from a day's
    values, is None in that row. Raises `WeatherServiceCallError` (or a
    subclass) or `gatekeeper.GatekeeperAuthError` on any failure, including
    a response without a usable `data` list -- never fabricates a day.
    """
    token = get_gatekeeper_token(settings)
    url = (
        f"{settings.gatekeeper_base_url.rstrip('/')}/api/proxy/"
        f"{settings.ws_service_name}{settings.ws_history_daily_path}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    request_id = current_request_id()
    if request_id is not None:
        headers["X-Request-ID"] = request_id

    body = {
        "lat": latitude,
        "lon": longitude,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "variables": list(variables),
        "radius_km": settings.ws_history_radius_km,
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=settings.ws_call_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise WeatherServiceConnectionError(f"WeatherService/GateKeeper unreachable at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise WeatherServiceHTTPError(response.status_code, response.text)

    try:
        answer = response.json()
    except ValueError as exc:
        raise WeatherServiceHTTPError(response.status_code, "non-JSON response") from exc

    rows = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or not rows:
        raise WeatherServiceHTTPError(response.status_code, "daily history response carries no daily rows")

    mapped: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("date"):
            raise WeatherServiceHTTPError(response.status_code, "daily history row without a date")
        values = row.get("values")
        if not isinstance(values, dict):
            values = {}
        entry: Dict[str, Any] = {"date": str(row["date"])}
        for source_key, target_key in _HISTORY_FIELD_MAP.items():
            entry[target_key] = values.get(source_key) if source_key in variables else None
        mapped.append(entry)

    fetched_at = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc)
    return {
        "source": str(answer.get("source") or "unknown"),
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "location": answer.get("location") or {"lat": latitude, "lon": longitude},
        "days": mapped,
    }
