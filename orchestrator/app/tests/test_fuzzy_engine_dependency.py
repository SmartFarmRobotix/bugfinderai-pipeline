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

"""Pinned agstack-pnd core dependency.

Verifies the dependency is importable in-process (no server), exposes its
pinned version, and that the [server] extra's runtime-only packages were
never installed -- i.e. this stays a pure-computation library embed,
not the FastAPI/Postgres/MCP server variant.
"""
from __future__ import annotations

import importlib.util


def test_agstack_pnd_importable_with_version() -> None:
    import agstack_pnd

    assert agstack_pnd.__version__ == "2.0.0a1"


def test_fuzzy_mamdani_engine_importable_with_version() -> None:
    from agstack_pnd.models.disease.fuzzy_mamdani import FuzzyMamdaniRisk

    assert FuzzyMamdaniRisk.metadata.version == "2.0.0"
    assert FuzzyMamdaniRisk.metadata.required_weather_fields == [
        "air_temperature",
        "relative_humidity",
        "precipitation",
    ]


def test_threat_definition_and_weather_dataframe_importable() -> None:
    from agstack_pnd.foundation.types import ThreatDefinition, WeatherDataFrame

    assert ThreatDefinition is not None
    assert WeatherDataFrame is not None


def test_server_extra_runtime_packages_not_installed() -> None:
    """The [server] extra (FastAPI/Postgres/NOAA/GeoID/MCP/Pancake) SHALL NOT
    be installed -- only agstack-pnd core (pydantic/numpy/httpx/dateutil)."""
    server_only_packages = ["asyncpg", "apscheduler", "mcp"]
    for package_name in server_only_packages:
        assert importlib.util.find_spec(package_name) is None, (
            f"'{package_name}' is installed -- it is only a transitive "
            "dependency of agstack-pnd's [server] extra, which must not be "
            "installed for the library-embed integration"
        )
