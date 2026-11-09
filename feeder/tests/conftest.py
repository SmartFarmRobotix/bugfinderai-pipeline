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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import DeploymentProfile, Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def make_settings(**overrides: Any) -> Settings:
    """
    Build a Settings instance for tests. Settings fields use `alias=`
    (pydantic-settings, no `populate_by_name`), so overrides must be keyed
    by the REPLAY_* alias, not the python attribute name.
    """
    defaults: dict[str, Any] = {
        "REPLAY_DEPLOYMENT_PROFILE": DeploymentProfile.X86_CI,
        "REPLAY_DATA_ROOT": "/tmp/unused-in-most-tests",
        "REPLAY_DEFAULT_LATITUDE": 10.0,
        "REPLAY_DEFAULT_LONGITUDE": 20.0,
        "REPLAY_INFERENCE_BASE_URL": "http://inference.test",
        "REPLAY_AUTO_PUSH": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)
