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

"""Position readers at the replay seams.

Per-image positions are optional telemetry. Replay records preserve source
positions under the ``replay`` provenance tag, while a future capture producer
can provide a GPS or odometry reading through ``PlatformCapturePositionReader``.
When that reader has no source value it returns the configured session fallback
with explicit ``default`` provenance; it never fabricates a per-image track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol


PositionProvenance = Literal["gps", "odometry", "replay", "default"]


@dataclass(frozen=True)
class PlatformPosition:
    """One approximate platform position and the source that produced it."""

    latitude: float
    longitude: float
    provenance: PositionProvenance


class PlatformPositionSource(Protocol):
    """Hardware seam for a later GPS or odometry adapter."""

    def read_position(self) -> Optional[PlatformPosition]:
        """Return the latest source position, or ``None`` when unavailable."""


class PlatformCapturePositionReader:
    """Read a capture-path source, falling back only at session level."""

    def __init__(
        self,
        source: PlatformPositionSource,
        *,
        default_latitude: Optional[float],
        default_longitude: Optional[float],
    ) -> None:
        self._source = source
        if (default_latitude is None) != (default_longitude is None):
            raise ValueError("default latitude and longitude must be set together")
        self._default = (
            PlatformPosition(
                latitude=default_latitude,
                longitude=default_longitude,
                provenance="default",
            )
            if default_latitude is not None and default_longitude is not None
            else None
        )

    def read_session_position(self) -> Optional[PlatformPosition]:
        """Return a source reading, a complete default pair, or honest absence."""
        return self._source.read_position() or self._default
