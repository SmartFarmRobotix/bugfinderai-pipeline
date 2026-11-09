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
NullDetector — a second, trivial Detector implementation: always reports
zero detections for every image. Exists to prove the detector interface is
genuinely pluggable and doubles as the fastest
possible smoke-test double when no ``detection_results.json`` fixture is
needed at all. Legitimately usable, not a test-only fake.
"""

from __future__ import annotations

from app.detectors.base import ImageDetections, ImageRef


class NullDetector:
    impl_id = "null-detector"
    version = "1.0.0"

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        return ImageDetections(image_id=image_ref.image_id, detections=[])

    def probe(self) -> None:
        return None
