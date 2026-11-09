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
Building the forecast outlook: the five days after the session, each evaluated on BOTH paths -- the
PDM's pest-model rules (the weather-rule level) and the fuzzy engine (the
fuzzy level). The result types and the payload shape live in
`forecast_outlook.py`.

The outlook follows the session's day: a session that ended more
than `FORECAST_WINDOW` ago gets the real HOURLY weather of the five days
after its end, from the Weather service's hourly history -- the route and
the frame the session window itself uses -- and both paths evaluate those
hours: the rules hour by hour per day (PDM's hourly form of its forecast
route), the fuzzy engine on the hourly frame, whose streak and wetness
sub-models need real hours. The day's row (maximum and minimum temperature,
mean humidity, precipitation sum) is aggregated from the same hours; the
hourly route is asked for temperature, humidity and precipitation, so the
wind cell is None. A session that ended within `FORECAST_WINDOW` gets the hourly forecast; only when that fails does
the outlook fall back to the daily forecast, with "not evaluated" in the fuzzy column.

Nothing here touches the session's stored assessment. Every failure of the
fetch or of the rules ends in `ForecastUnavailable` with a named reason: the outlook is optional, so it never blocks the report, and it never invents
a table. A fuzzy engine failure leaves the fuzzy cells empty instead: the
rules' table stands without it.
"""

from __future__ import annotations

import datetime
import logging
import statistics
from typing import Any, Callable, Dict, List, Optional

from forecast_outlook import (
    FUZZY_NOT_EVALUATED,
    ForecastOutlookData,
    ForecastResult,
    ForecastUnavailable,
    UNAVAILABLE_NO_KEY,
    UNAVAILABLE_NO_POSITION,
    UNAVAILABLE_RULES,
    UNAVAILABLE_SERVICE,
)
from fuzzy_client import assess_outlook_hours_via_fuzzy_engine
from pdm_client import assess_forecast_hours_via_pdm, assess_forecast_via_pdm
from ws_client import (
    REQUIRED_VARIABLES,
    WeatherServiceHTTPError,
    fetch_daily_forecast,
    fetch_hourly_forecast,
    fetch_hourly_history,
)

logger = logging.getLogger(__name__)

FORECAST_DAYS = 5
FORECAST_WINDOW = datetime.timedelta(hours=24)

_LEVEL_WORD = {"low": "Low", "moderate": "Moderate", "high": "High"}


def _is_missing_key(exc: WeatherServiceHTTPError) -> bool:
    """The Weather service answers 503 naming its missing OpenWeatherMap key;
    any other failure (including a bare 503 from the proxy) is 'service
    unavailable'."""
    return exc.status_code == 503 and "api key" in exc.body.lower()


def _session_end(time_end: Optional[str]) -> Optional[datetime.datetime]:
    """The session's end as an aware UTC datetime (a naive value is UTC), or
    None when it is absent or not ISO 8601."""
    if not time_end:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(time_end.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _hours_by_day(
    answer: Dict[str, Any], first_day: datetime.date, last_day: datetime.date
) -> Dict[str, List[Dict[str, Any]]]:
    """The hourly history's complete hours inside `first_day`..`last_day`,
    grouped by date in time order. An hour missing any of the three required
    variables (the archive lags a few days behind) is left out: the frame
    adapter refuses incomplete hours, and no value is filled in for it."""
    observations = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(observations, list):
        raise ValueError("hourly history response carries no 'data' list")
    days: Dict[str, List[Dict[str, Any]]] = {}
    for observation in sorted(observations, key=lambda o: str(o.get("timestamp"))):
        timestamp = str(observation.get("timestamp") or "")
        values = observation.get("values") or {}
        if not first_day.isoformat() <= timestamp[:10] <= last_day.isoformat():
            continue
        if any(values.get(name) is None for name in REQUIRED_VARIABLES):
            continue
        days.setdefault(timestamp[:10], []).append({"timestamp": timestamp, "values": values})
    return days


def _day_row(date: str, hours: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The outlook row of one day, aggregated from its hours. The history
    route is not asked for wind, so that cell is None there; the hourly
    forecast route is, so it is filled with the hours' mean when every hour
    carries it."""
    temperatures = [float(h["values"]["temperature_2m"]) for h in hours]
    humidities = [float(h["values"]["relative_humidity_2m"]) for h in hours]
    precipitation = [float(h["values"]["precipitation"]) for h in hours]
    wind_speed_ms = None
    if hours and all(h["values"].get("wind_speed_10m") is not None for h in hours):
        wind_speed_ms = round(statistics.mean(float(h["values"]["wind_speed_10m"]) for h in hours), 1)
    return {
        "date": date,
        "temp_max_C": round(max(temperatures), 1),
        "temp_min_C": round(min(temperatures), 1),
        "humidity_pct": round(statistics.mean(humidities), 1),
        "precipitation_mm": round(sum(precipitation), 1),
        "wind_speed_ms": wind_speed_ms,
    }


class _EmptyHourlyForecast(Exception):
    """Internal marker: the hourly forecast call answered but with no usable
    hour for the five days after the session's end day."""


def build_forecast_outlook(
    settings,
    latitude: Optional[float],
    longitude: Optional[float],
    crop: str,
    pest: Optional[str] = None,
    *,
    time_end: Optional[str] = None,
    now: Optional[datetime.datetime] = None,
    ws_call: Callable[..., Dict[str, Any]] = fetch_daily_forecast,
    hourly_call: Callable[..., Dict[str, Any]] = fetch_hourly_history,
    hourly_forecast_call: Callable[..., Dict[str, Any]] = fetch_hourly_forecast,
    pdm_call: Callable[..., Any] = assess_forecast_via_pdm,
    pdm_hours_call: Callable[..., Any] = assess_forecast_hours_via_pdm,
    fuzzy_call: Callable[..., Dict[str, str]] = assess_outlook_hours_via_fuzzy_engine,
) -> ForecastResult:
    """The session's five days, fetched once through the Weather service via
    GateKeeper and evaluated. A session that ended more than
    `FORECAST_WINDOW` before `now`: the hourly history of the five days after
    its end, the rules per day over that day's hours and the fuzzy engine on
    the hourly frame. Otherwise (or with no usable `time_end`): the hourly
    forecast of the five calendar days after the session's end day (or, with
    no usable `time_end`, after today), evaluated the same way. Only when
    that hourly forecast fails or comes back empty does the outlook fall
    back to the daily forecast, the rules on it, and "not evaluated" as the
    fuzzy level -- with the fallback reason recorded."""
    if latitude is None or longitude is None:
        return ForecastUnavailable(UNAVAILABLE_NO_POSITION)

    ended = _session_end(time_end)
    now_utc = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc)
    hourly = ended is not None and now_utc - ended > FORECAST_WINDOW
    hours_by_day: Optional[Dict[str, List[Dict[str, Any]]]] = None
    fallback_reason: Optional[str] = None
    try:
        if hourly:
            first_day = ended.date() + datetime.timedelta(days=1)
            last_day = first_day + datetime.timedelta(days=FORECAST_DAYS - 1)
            answer = hourly_call(
                settings,
                latitude,
                longitude,
                datetime.datetime.combine(first_day, datetime.time(0)).isoformat(),
                datetime.datetime.combine(last_day, datetime.time(23)).isoformat(),
            )
            hours_by_day = _hours_by_day(answer, first_day, last_day)
            if not hours_by_day:
                raise ValueError("hourly history carries no complete hour for the five days")
            forecast = {
                "source": str(answer.get("source") or "unknown"),
                "fetched_at": now_utc.isoformat(timespec="seconds"),
                "location": answer.get("location") or {"lat": latitude, "lon": longitude},
                "days": [_day_row(date, hours) for date, hours in sorted(hours_by_day.items())],
            }
        else:
            base_day = now_utc.date()
            first_day = base_day + datetime.timedelta(days=1)
            last_day = first_day + datetime.timedelta(days=FORECAST_DAYS - 1)
            try:
                answer = hourly_forecast_call(settings, latitude, longitude, FORECAST_DAYS)
                forecast_hours_by_day = _hours_by_day(answer, first_day, last_day)
                if not forecast_hours_by_day:
                    raise _EmptyHourlyForecast()
                hours_by_day = forecast_hours_by_day
                forecast = {
                    "source": str(answer.get("source") or "unknown"),
                    "fetched_at": now_utc.isoformat(timespec="seconds"),
                    "location": answer.get("location") or {"lat": latitude, "lon": longitude},
                    "days": [_day_row(date, hours) for date, hours in sorted(hours_by_day.items())],
                }
            except _EmptyHourlyForecast:
                fallback_reason = "hourly forecast empty"
                logger.warning("hourly forecast carried no usable hour; falling back to daily forecast")
            except Exception as exc:  # noqa: BLE001 -- any hourly-forecast failure falls back to the daily one
                fallback_reason = f"hourly forecast unavailable: {type(exc).__name__}"
                logger.warning("hourly forecast fetch failed (%s); falling back to daily forecast", exc)
            if fallback_reason is not None:
                forecast = ws_call(settings, latitude, longitude, FORECAST_DAYS)
    except WeatherServiceHTTPError as exc:
        logger.warning("forecast fetch failed: %s", exc)
        return ForecastUnavailable(UNAVAILABLE_NO_KEY if _is_missing_key(exc) else UNAVAILABLE_SERVICE)
    except Exception as exc:  # noqa: BLE001 -- GK auth, connection, shape: all "no forecast"
        logger.warning("forecast fetch failed: %s", exc)
        return ForecastUnavailable(UNAVAILABLE_SERVICE)

    rows: List[Dict[str, Any]] = forecast["days"]
    evaluate_hourly = hours_by_day is not None

    try:
        if evaluate_hourly:
            pdm_result = pdm_hours_call(
                settings,
                crop,
                [
                    {
                        "date": date,
                        "hours": [
                            {
                                "time": hour["timestamp"],
                                **{name: hour["values"][name] for name in REQUIRED_VARIABLES},
                            }
                            for hour in hours
                        ],
                    }
                    for date, hours in sorted(hours_by_day.items())
                ],
            )
        else:
            pdm_result = pdm_call(settings, crop, rows)
    except Exception as exc:  # noqa: BLE001 -- the table needs the rule levels; without them, none
        logger.warning("forecast rule evaluation failed: %s", exc)
        return ForecastUnavailable(UNAVAILABLE_RULES)
    if pdm_result.reason:
        return ForecastUnavailable(f"{UNAVAILABLE_RULES} ({pdm_result.reason})")
    levels = {entry["date"]: str(entry["level"]).lower() for entry in pdm_result.days}
    if any(row["date"] not in levels for row in rows):
        return ForecastUnavailable(f"{UNAVAILABLE_RULES} (incomplete answer)")

    fuzzy_levels: Dict[str, str] = {}
    if evaluate_hourly:
        try:
            fuzzy_levels = fuzzy_call(
                crop, [hour for _, hours in sorted(hours_by_day.items()) for hour in hours], pest
            )
        except Exception as exc:  # noqa: BLE001 -- the fuzzy level is optional per day
            logger.warning("fuzzy outlook evaluation failed: %s", exc)

    days = [
        {
            **row,
            "risk_level": _LEVEL_WORD.get(levels[row["date"]], levels[row["date"]].capitalize()),
            "fuzzy_level": fuzzy_levels.get(row["date"]) if evaluate_hourly else FUZZY_NOT_EVALUATED,
        }
        for row in rows
    ]
    return ForecastOutlookData(
        source=forecast["source"],
        fetched_at=forecast["fetched_at"],
        location=forecast["location"],
        days=days,
        days_source="history" if hourly else "forecast",
        fallback_reason=fallback_reason,
    )
