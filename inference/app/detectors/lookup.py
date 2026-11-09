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
LookupDetector — the deterministic, input-correlated stub.

This stub is for CI/dev smoke only, never for scenario acceptance, which requires a real detector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from app.detectors.base import Detection, ImageDetections, ImageRef

_DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "detection_results.json"


class LookupDetectionValidationError(ValueError):
    """
    Raised when a ``detection_results.json`` entry contains a malformed
    score (outside ``[0,1]``) or a label absent from the configured class
    manifest — the stub VALIDATES these
    fields rather than silently substituting a default value. The stub's
    sparse/empty canonical-detections contract itself (no matching entry
    -> empty list) is unchanged; this only guards the populated path.
    """


def _load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


class LookupDetector:
    """
    Keyed by image identity (filename, then filename stem) against the
    prepared ``detection_results.json`` produced alongside the sample
    images.

    Deterministic unknown-image fallback: an image with no matching entry
    yields zero detections (empty list, per-image score 0.0). This is a
    fixed rule — not randomized, not hash-picked — so the same unknown
    image always yields the same (empty) result, and different unknown
    images never spuriously "borrow" another image's detections.
    """

    impl_id = "lookup-detector"
    version = "1.0.0"

    def __init__(self, results_path: Optional[Path] = None, class_names: Optional[Sequence[str]] = None) -> None:
        self._results = _load_results(results_path or _DEFAULT_RESULTS_PATH)
        self._class_names = set(class_names) if class_names is not None else None

    def _lookup(self, filename: str) -> Optional[Dict[str, Any]]:
        if not self._results:
            return None
        if filename in self._results:
            return self._results[filename]
        stem = Path(filename).stem
        for key, entry in self._results.items():
            if Path(key).stem == stem:
                return entry
        return None

    def _validate_box(self, box: Dict[str, Any]) -> None:
        score = float(box.get("score", 0.0))
        if not (0.0 <= score <= 1.0):
            raise LookupDetectionValidationError(
                f"malformed score {score!r} outside [0,1] in a detection_results.json entry"
            )
        label = box.get("label", "unknown")
        if self._class_names is not None and label not in self._class_names:
            raise LookupDetectionValidationError(
                f"label {label!r} is not in the configured class manifest {sorted(self._class_names)!r}"
            )

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        entry = self._lookup(image_ref.filename)
        if entry is None:
            return ImageDetections(image_id=image_ref.image_id, detections=[])

        detections = []
        for box in entry.get("boxes", []):
            self._validate_box(box)
            detections.append(
                Detection(
                    x_min=box["x_min"],
                    y_min=box["y_min"],
                    x_max=box["x_max"],
                    y_max=box["y_max"],
                    score=float(box.get("score", 0.0)),
                    label=box.get("label", "unknown"),
                )
            )
        return ImageDetections(image_id=image_ref.image_id, detections=detections)

    def probe(self) -> None:
        return None
