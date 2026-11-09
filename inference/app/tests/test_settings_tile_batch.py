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
The ``INFERENCE_TILE_BATCH`` domain:
an integer >= 1. Anything else refuses at settings load, naming the setting
and quoting the value -- through the environment (how a deployment sets it)
and through a direct constructor argument alike.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

SETTING = "INFERENCE_TILE_BATCH"


def test_the_default_is_four(monkeypatch):
    monkeypatch.delenv(SETTING, raising=False)
    assert Settings(_env_file=None).tile_batch == 4


@pytest.mark.parametrize("value", ["1", "4", "12", "64"])
def test_an_integer_of_at_least_one_is_accepted_from_the_environment(monkeypatch, value):
    monkeypatch.setenv(SETTING, value)
    assert Settings(_env_file=None).tile_batch == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", ""])
def test_a_value_outside_the_domain_refuses_at_settings_load_from_the_environment(monkeypatch, value):
    monkeypatch.setenv(SETTING, value)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    message = str(excinfo.value)
    assert SETTING in message
    assert f"input_value={value!r}" in message


@pytest.mark.parametrize("value", [0, -1, 1.5, "abc"])
def test_a_value_outside_the_domain_refuses_as_a_constructor_argument(monkeypatch, value):
    monkeypatch.delenv(SETTING, raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, INFERENCE_TILE_BATCH=value)
    message = str(excinfo.value)
    assert SETTING in message
    assert f"input_value={value!r}" in message
