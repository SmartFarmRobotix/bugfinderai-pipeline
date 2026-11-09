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

"""Future platform-capture position-source seam."""

from __future__ import annotations

from dataclasses import dataclass

from app.positions import PlatformCapturePositionReader, PlatformPosition


@dataclass
class FixturePositionSource:
    position: PlatformPosition | None

    def read_position(self) -> PlatformPosition | None:
        return self.position


def test_capture_reader_preserves_available_gps_position_and_provenance():
    reader = PlatformCapturePositionReader(
        FixturePositionSource(PlatformPosition(10.501, 20.501, "gps")),
        default_latitude=10.0,
        default_longitude=20.0,
    )

    assert reader.read_session_position() == PlatformPosition(10.501, 20.501, "gps")


def test_capture_reader_uses_explicit_default_provenance_when_source_is_absent():
    reader = PlatformCapturePositionReader(
        FixturePositionSource(None),
        default_latitude=10.0,
        default_longitude=20.0,
    )

    assert reader.read_session_position() == PlatformPosition(10.0, 20.0, "default")


def test_capture_reader_refuses_default_provenance_without_a_coordinate_pair():
    reader = PlatformCapturePositionReader(
        FixturePositionSource(None),
        default_latitude=None,
        default_longitude=None,
    )

    assert reader.read_session_position() is None
