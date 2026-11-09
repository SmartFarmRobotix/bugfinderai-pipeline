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
The forecast outlook of a session report: the result types, the reasons the producer records when it
has no forecast, and the payload's `forecast_outlook` shape. Fetching and
evaluating live in `forecast_builder.py`.

The forecast is an outlook printed in the report, never combined with the
infestation score and never persisted as an assessment record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

UNAVAILABLE_NO_POSITION = "position unavailable"
UNAVAILABLE_NO_KEY = "no weather-provider key configured"
UNAVAILABLE_SERVICE = "weather service unavailable"
UNAVAILABLE_RULES = "risk rules unavailable"

FUZZY_NOT_EVALUATED = "not evaluated"

_LEVEL_ORDER = {"Low": 1, "Moderate": 2, "High": 3}


@dataclass(frozen=True)
class ForecastOutlookData:
    source: str
    fetched_at: str
    location: Dict[str, Any]
    days: List[Dict[str, Any]]
    days_source: str
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class ForecastUnavailable:
    reason: str


ForecastResult = Union[ForecastOutlookData, ForecastUnavailable]


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _day_month(iso_date: str) -> Optional[tuple]:
    """(day, month name) of an ISO date, or None when it is not one."""
    try:
        _, month, day = (int(part) for part in str(iso_date)[:10].split("-"))
        return day, _MONTHS[month - 1]
    except (ValueError, IndexError):
        return None


def _join_dates(iso_dates: List[str]) -> str:
    """'22 and 23 September' for two days of one month, '22, 23, 24
    September' for more; days in different months each carry their month
    ('30 September and 1 October')."""
    parts = [_day_month(d) or (str(d), "") for d in iso_dates]
    months = {month for _, month in parts}
    if len(months) == 1:
        labels = [str(day) for day, _ in parts]
        tail = f" {months.pop()}".rstrip()
    else:
        labels = [f"{day} {month}".strip() for day, month in parts]
        tail = ""
    joined = labels[0] if len(labels) == 1 else (
        f"{labels[0]} and {labels[1]}" if len(labels) == 2 else ", ".join(labels)
    )
    return joined + tail


def compose_outlook_summary(days: List[Dict[str, Any]]) -> str:
    """One deterministic sentence over the weather-rule levels of the days
    (the printed label is added by the renderer): one level for all days, or
    the top level with its dates and the next level for the others. A third
    level folds into 'the others'; a level no day carries is never named."""
    if not days:
        return ""
    levels = sorted(
        {d["risk_level"] for d in days},
        key=lambda level: _LEVEL_ORDER.get(level, 0),
        reverse=True,
    )
    if len(levels) == 1:
        return f"The weather rules give {levels[0]} risk on all {len(days)} days."
    top = [d["date"] for d in days if d["risk_level"] == levels[0]]
    return (
        f"The weather rules give {levels[0]} risk on {len(top)} of the {len(days)} "
        f"days ({_join_dates(top)}), {levels[1]} on the others."
    )


def forecast_outlook_payload(
    forecast: Optional[ForecastResult], recommendation: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The payload's `forecast_outlook`. `forecast` None means no forecast
    was attempted for this composition (the recommendation-only outlook the
    composer always produced). A forecast: the day rows with both levels,
    the source, the fetch time, the location, the summary and the
    recommendation. No forecast (`ForecastUnavailable`): the recommendation
    and the reason. Nothing to say -> None."""
    if forecast is None:
        return {"recommendation": recommendation} if recommendation is not None else None
    outlook: Dict[str, Any]
    if isinstance(forecast, ForecastOutlookData):
        outlook = {
            "source": forecast.source,
            "fetched_at": forecast.fetched_at,
            "location": forecast.location,
            "days": forecast.days,
            "days_source": forecast.days_source,
            "outlook_summary": compose_outlook_summary(forecast.days),
        }
        if forecast.fallback_reason is not None:
            outlook["fallback_reason"] = forecast.fallback_reason
    else:
        outlook = {"unavailable_reason": forecast.reason}
    if recommendation is not None:
        outlook = {**outlook, "recommendation": recommendation}
    return outlook
