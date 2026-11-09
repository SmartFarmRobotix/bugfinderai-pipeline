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
Pluggable detector interface.

The session pipeline obtains per-image detections exclusively through
``Detector.detect`` — swapping the configured implementation never changes the session contract that
consumers see, because scoring/aggregation (app/scoring.py) is computed
from ``ImageDetections`` uniformly regardless of which Detector produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class ImageRef:
    """What a Detector needs to identify and locate one session image."""

    image_id: str
    filename: str
    path: Path


@dataclass(frozen=True)
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label: str


@dataclass(frozen=True)
class ImageDetections:
    image_id: str
    detections: List[Detection]


class Detector(Protocol):
    """Sync detect(image_ref) -> ImageDetections."""

    impl_id: str
    version: str


    def detect(self, image_ref: ImageRef) -> ImageDetections: ...

    def probe(self) -> None:
        """One execution of the runtime on a zero input; raises when the
        runtime cannot run (``/healthz`` calls it on the detector's own
        executor thread). Artifact-less detectors have nothing to prove."""
        ...


def empty_detections(image_id: str) -> ImageDetections:
    return ImageDetections(image_id=image_id, detections=[])
