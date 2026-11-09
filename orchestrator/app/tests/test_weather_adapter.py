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

"""Weather-service history -> WeatherDataFrame adapter."""
from __future__ import annotations

import copy
import json
import os

import pytest

from weather_adapter import (
    MissingWeatherSeriesError,
    WeatherUnitMismatchError,
    ws_history_to_weather_dataframe,
)

_FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "fuzzy_evaluation_vectors"
)


def _load_fixture(name: str) -> dict:
    with open(os.path.join(_FIXTURES_DIR, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _minimal_ws_response() -> dict:
    return {
        "source": "openmeteo",
        "location": {"lat": 45.0, "lon": 10.0},
        "data": [
            {
                "timestamp": "2026-06-01T12:00:00",
                "values": {
                    "temperature_2m": 22.0,
                    "relative_humidity_2m": 60.0,
                    "precipitation": 0.0,
                },
            },
        ],
    }


class TestFixtureAdaptation:
    def test_favorable_dry_fixture_adapts_to_expected_shape(self) -> None:
        fixture = _load_fixture("favorable_dry_low_infestation")
        wdf = ws_history_to_weather_dataframe(fixture["ws_response"])

        assert wdf.length == len(fixture["ws_response"]["data"])
        assert set(wdf.field_names()) == {
            "air_temperature",
            "relative_humidity",
            "precipitation",
        }
        first_obs = fixture["ws_response"]["data"][0]["values"]
        assert wdf["air_temperature"][0] == pytest.approx(first_obs["temperature_2m"])
        assert wdf["relative_humidity"][0] == pytest.approx(first_obs["relative_humidity_2m"])
        assert wdf["precipitation"][0] == pytest.approx(first_obs["precipitation"])

    def test_unfavorable_fixture_adapts_to_expected_shape(self) -> None:
        fixture = _load_fixture("unfavorable_cold_wet")
        wdf = ws_history_to_weather_dataframe(fixture["ws_response"])
        assert wdf.length == len(fixture["ws_response"]["data"])


class TestMalformedWindowFailures:
    def test_empty_data_raises_missing_series(self) -> None:
        response = {"source": "openmeteo", "location": {"lat": 0, "lon": 0}, "data": []}
        with pytest.raises(MissingWeatherSeriesError):
            ws_history_to_weather_dataframe(response)

    def test_missing_data_key_raises_missing_series(self) -> None:
        response = {"source": "openmeteo", "location": {"lat": 0, "lon": 0}}
        with pytest.raises(MissingWeatherSeriesError):
            ws_history_to_weather_dataframe(response)

    def test_missing_timestamp_raises(self) -> None:
        response = _minimal_ws_response()
        del response["data"][0]["timestamp"]
        with pytest.raises(MissingWeatherSeriesError):
            ws_history_to_weather_dataframe(response)

    @pytest.mark.parametrize(
        "missing_key", ["temperature_2m", "relative_humidity_2m", "precipitation"]
    )
    def test_missing_required_series_raises(self, missing_key: str) -> None:
        response = _minimal_ws_response()
        del response["data"][0]["values"][missing_key]
        with pytest.raises(MissingWeatherSeriesError, match=missing_key):
            ws_history_to_weather_dataframe(response)

    @pytest.mark.parametrize(
        "missing_key", ["temperature_2m", "relative_humidity_2m", "precipitation"]
    )
    def test_null_required_series_raises(self, missing_key: str) -> None:
        response = _minimal_ws_response()
        response["data"][0]["values"][missing_key] = None
        with pytest.raises(MissingWeatherSeriesError):
            ws_history_to_weather_dataframe(response)

    def test_non_numeric_value_raises_unit_mismatch(self) -> None:
        response = _minimal_ws_response()
        response["data"][0]["values"]["temperature_2m"] = "warm"
        with pytest.raises(WeatherUnitMismatchError):
            ws_history_to_weather_dataframe(response)

    def test_fahrenheit_temperature_raises_unit_mismatch(self) -> None:
        """98.6 F body-temp-like value is out of the plausible Celsius range --
        the adapter's only available unit-mismatch signal given the Weather service's wire
        format carries no explicit unit metadata (see weather_adapter.py)."""
        response = _minimal_ws_response()
        response["data"][0]["values"]["temperature_2m"] = 98.6
        with pytest.raises(WeatherUnitMismatchError):
            ws_history_to_weather_dataframe(response)

    def test_humidity_fraction_over_100_raises_unit_mismatch(self) -> None:
        response = _minimal_ws_response()
        response["data"][0]["values"]["relative_humidity_2m"] = 160.0
        with pytest.raises(WeatherUnitMismatchError):
            ws_history_to_weather_dataframe(response)

    def test_negative_precipitation_raises_unit_mismatch(self) -> None:
        response = _minimal_ws_response()
        response["data"][0]["values"]["precipitation"] = -1.0
        with pytest.raises(WeatherUnitMismatchError):
            ws_history_to_weather_dataframe(response)

    def test_fractional_humidity_raises_unit_mismatch(self) -> None:
        """The 0.60-fractional case: a lone value of 0.60 satisfies the
        plain [0, 100] range check, so this can only be caught once every
        reading in the window is known to be <= 1.0 (signature 1)."""
        response = _minimal_ws_response()
        response["data"][0]["values"]["relative_humidity_2m"] = 0.60
        with pytest.raises(WeatherUnitMismatchError, match="fractional"):
            ws_history_to_weather_dataframe(response)

    def test_fractional_humidity_not_flagged_when_any_reading_exceeds_one(
        self,
    ) -> None:
        """A single window with one plainly-percent reading (e.g. 45.0)
        alongside a fractional-looking one is a genuinely ambiguous/mixed
        window -- signature 1 only fires when EVERY reading is <= 1.0, per
        its documented scope, so this must NOT raise on that account."""
        response = _minimal_ws_response()
        second = copy.deepcopy(response["data"][0])
        second["timestamp"] = "2026-06-01T13:00:00"
        second["values"]["relative_humidity_2m"] = 45.0
        response["data"][0]["values"]["relative_humidity_2m"] = 0.60
        response["data"].append(second)

        wdf = ws_history_to_weather_dataframe(response)
        assert wdf.length == 2

    def test_fahrenheit_range_temperature_with_coherent_humidity_raises_unit_mismatch(
        self,
    ) -> None:
        """The 50.0 °F case: a temperature of 50.0 alone satisfies the
        plain [-90, 60] range check (reads as a hot-but-possible Celsius
        day). Paired with an everyday humidity reading (60% here, from
        `_minimal_ws_response`), the combination is not observed on Earth
        and is caught by the cross-field coherence check (signature 2)."""
        response = _minimal_ws_response()
        response["data"][0]["values"]["temperature_2m"] = 50.0
        with pytest.raises(WeatherUnitMismatchError, match="degC"):
            ws_history_to_weather_dataframe(response)

    def test_cold_fahrenheit_misread_is_not_detected_documented_gap(self) -> None:
        """Documents the module docstring's honest residual gap: a
        Fahrenheit reading at or below the 45 degC coherence threshold
        (e.g. "10" meaning 10 degF) reads as an ordinary mild/cool Celsius
        value and is NOT caught by any check here -- this asserts the gap
        stays exactly as wide as documented, not wider."""
        response = _minimal_ws_response()
        response["data"][0]["values"]["temperature_2m"] = 10.0
        response["data"][0]["values"]["relative_humidity_2m"] = 70.0
        wdf = ws_history_to_weather_dataframe(response)
        assert wdf["air_temperature"][0] == pytest.approx(10.0)

    def test_never_silently_coerces(self) -> None:
        """A malformed second observation must fail the whole window, not
        drop the bad row and proceed with a partial/fabricated result."""
        response = _minimal_ws_response()
        second = copy.deepcopy(response["data"][0])
        second["timestamp"] = "2026-06-01T13:00:00"
        del second["values"]["precipitation"]
        response["data"].append(second)

        with pytest.raises(MissingWeatherSeriesError):
            ws_history_to_weather_dataframe(response)
