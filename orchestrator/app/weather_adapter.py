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

"""Weather-service history -> WeatherDataFrame adapter.

Adapts the OpenAgri WeatherService's hourly-history response shape
`{"source": ..., "location": {...}, "data": [{"timestamp": ..., "values":
{"temperature_2m": ..., "relative_humidity_2m": ..., "precipitation": ...}},
...]}`, the same Open-Meteo-variable-named shape the PDM service's own weather fetch
already requests -- into the fuzzy engine's `WeatherDataFrame`
(`air_temperature`, `relative_humidity`, `precipitation`; Celsius/%/mm, the
engine's expected units per `fuzzy_mamdani.py`'s `required_weather_fields`).

WEATHER-SERVICE UNIT CONTRACT (why Celsius/%/mm is assumed, not just hoped for): the Weather service's
`OpenMeteoClient.get_hourly_history`
(`OpenAgri-WeatherService/src/external_services/openmeteo.py`) calls
Open-Meteo's archive API (`archive-api.open-meteo.com/v1/archive`) without
a `temperature_unit`, `precipitation_unit`, or `windspeed_unit` override,
so Open-Meteo's own documented defaults govern every value on the wire:
temperature in Celsius, precipitation in millimeters, and
`relative_humidity_2m` always as a percentage (Open-Meteo has no alternate
unit for it). The Weather service performs no unit conversion of its own between that
upstream call and the `/hourly/` response this adapter consumes. That is
the basis for the Celsius/%/mm assumption below -- it is a provenance
citation, not a per-request wire check, because `HourlyResponse` in the Weather service'
`src/schemas/history_data.py` carries no `units` field to check against.

Given that, unit-mismatch detection here is plausibility heuristics, not a
declared-unit comparison: a plain physical-range check per series, plus two
additional cross-checks closing gaps the plain range check misses:

1. **Fractional humidity.** A single `relative_humidity_2m` value of, say,
   `0.60` still satisfies the plain `[0, 100]` range check. If EVERY
   reading in the window is `<= 1.0`, that is flagged instead as suspected
   0-1 fractional encoding -- a real window sustaining <=1% RH throughout
   does not happen on Earth.
2. **Fahrenheit-as-Celsius coherence.** A single temperature value of, say,
   `50.0` also satisfies the plain `[-90, 60]` range check (it reads as a
   hot-but-physically-possible Celsius day). Earth's genuine >45 degC
   dry-bulb events are consistently arid (RH well under 25%), while
   everyday Fahrenheit ambient readings (roughly 45-140 degF) numerically
   alias into this same "hot Celsius" band and ordinarily co-occur with
   everyday (non-arid) humidity. So a temperature reading above 45 degC
   together with a relative humidity reading above 25% in the *same*
   observation is flagged as a suspected degF-as-degC misread.

**Documented, honest residual gap:** a Fahrenheit misread whose raw numeric
value lands at or below 45 (i.e. a real ambient reading at or below
45 degF / ~7.2 degC, e.g. "50" meaning 50 degF is caught above only because
50 > 45 -- a milder value like "40" meaning 40 degF would not be) is
indistinguishable from a genuine mild/cool Celsius reading by any check in
this module. The Weather service's wire format carries no per-request unit field, so closing
this band is not possible without unit metadata from the Weather service; it is not claimed
as covered. This module is deliberately conservative in the other
direction instead -- it fails rather than silently coercing whenever its
heuristics do trip.
"""

from __future__ import annotations

import numpy as np

from agstack_pnd.foundation.types import WeatherDataFrame

_SERIES_MAP: dict[str, tuple[str, tuple[float, float]]] = {
    "temperature_2m": ("air_temperature", (-90.0, 60.0)),
    "relative_humidity_2m": ("relative_humidity", (0.0, 100.0)),
    "precipitation": ("precipitation", (0.0, 500.0)),
}

_FRACTIONAL_HUMIDITY_CEILING = 1.0

_COHERENCE_HOT_TEMP_C = 45.0
_COHERENCE_HUMID_FLOOR_PCT = 25.0


class WeatherAdapterError(ValueError):
    """Base class for Weather-service history -> WeatherDataFrame adapter failures."""


class MissingWeatherSeriesError(WeatherAdapterError):
    """A required weather series is absent from the Weather service history window."""


class WeatherUnitMismatchError(WeatherAdapterError):
    """A weather series value is implausible given the Weather service unit contract
    (module docstring): outside its plausible physical range, an entire
    window of fractional-looking humidity, or a temperature/humidity pair
    that is not observed on Earth. The adapter's available signals given
    the wire format carries no unit metadata -- not a full unit-typed
    guarantee (see the module docstring's documented residual gap)."""


def ws_history_to_weather_dataframe(ws_response: dict) -> WeatherDataFrame:
    """Adapt a Weather-service `/hourly/` history response into a `WeatherDataFrame`.

    Raises `MissingWeatherSeriesError` if any of the three required series
    (`temperature_2m`, `relative_humidity_2m`, `precipitation`) is absent
    from the response, or from any individual observation. Raises
    `WeatherUnitMismatchError` if a present value falls outside its
    plausible physical range, if every `relative_humidity_2m` reading in
    the window is `<= 1.0` (suspected fractional encoding), or if a
    temperature/humidity pair in one observation is not physically
    observed on Earth (suspected degF-as-degC misread) -- see the module
    docstring for both heuristics and their documented residual gap. Never
    fabricates or coerces a missing/invalid value.
    """
    observations = ws_response.get("data")
    if not observations:
        raise MissingWeatherSeriesError(
            "the Weather service history response has no observations ('data' is empty or missing)"
        )

    timestamps: list[str] = []
    series: dict[str, list[float]] = {out_name: [] for out_name, _ in _SERIES_MAP.values()}

    for index, observation in enumerate(observations):
        timestamp = observation.get("timestamp")
        if not timestamp:
            raise MissingWeatherSeriesError(
                f"observation index {index} has no 'timestamp'"
            )
        values = observation.get("values") or {}
        obs_values: dict[str, float] = {}

        for source_key, (field_name, (lo, hi)) in _SERIES_MAP.items():
            if source_key not in values or values[source_key] is None:
                raise MissingWeatherSeriesError(
                    f"observation index {index} (timestamp={timestamp!r}) is "
                    f"missing required series '{source_key}'"
                )
            value = values[source_key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise WeatherUnitMismatchError(
                    f"observation index {index} (timestamp={timestamp!r}): "
                    f"'{source_key}' is {value!r}, not numeric"
                )
            value = float(value)
            if not (lo <= value <= hi):
                raise WeatherUnitMismatchError(
                    f"observation index {index} (timestamp={timestamp!r}): "
                    f"'{source_key}'={value} is outside its plausible range "
                    f"[{lo}, {hi}] -- likely wrong unit, not a fabricable value"
                )
            obs_values[field_name] = value

        if (
            obs_values["air_temperature"] > _COHERENCE_HOT_TEMP_C
            and obs_values["relative_humidity"] > _COHERENCE_HUMID_FLOOR_PCT
        ):
            raise WeatherUnitMismatchError(
                f"observation index {index} (timestamp={timestamp!r}): "
                f"air_temperature={obs_values['air_temperature']} degC together "
                f"with relative_humidity={obs_values['relative_humidity']}% is "
                "not observed on Earth (genuine >45 degC events are arid) -- "
                "likely a Fahrenheit reading stored as Celsius, not a "
                "fabricable value"
            )

        for field_name, value in obs_values.items():
            series[field_name].append(value)
        timestamps.append(timestamp)

    if series["relative_humidity"] and max(series["relative_humidity"]) <= _FRACTIONAL_HUMIDITY_CEILING:
        raise WeatherUnitMismatchError(
            f"every 'relative_humidity_2m' reading in the window is <= "
            f"{_FRACTIONAL_HUMIDITY_CEILING} -- suspected 0-1 fractional "
            "encoding, not percent (sustained real RH this low does not "
            "occur), not a fabricable value"
        )

    ts_array = np.array(timestamps, dtype="datetime64[ns]")
    fields = {name: np.array(values, dtype=np.float64) for name, values in series.items()}
    return WeatherDataFrame(timestamps=ts_array, fields=fields)
