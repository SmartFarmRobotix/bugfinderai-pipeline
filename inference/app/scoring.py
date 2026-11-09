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
Deterministic per-image scoring and session aggregation. Identical across detector implementations —
model-agnostic by construction: it only ever looks at ``label``/``score``
on the detections a Detector implementation returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

AGGREGATION_RULE_VERSION = "topfrac-v1"

_TOP_FRACTION = 0.20


@dataclass(frozen=True)
class ScoredImage:
    image_id: str
    score: float


def per_image_score(
    detections: Iterable[object],
    target_class_set: Sequence[str],
    confidence_floor: float,
) -> float:
    """
    Per-image score = the maximum confidence among detections whose class
    is in ``target_class_set``, after excluding detections below
    ``confidence_floor``; 0.0 when no qualifying detection exists.

    ``detections`` items need only expose ``.label`` and ``.score``
    attributes (duck-typed so any Detector's box representation works).
    """
    target_classes = set(target_class_set)
    qualifying = [
        float(det.score)
        for det in detections
        if det.label in target_classes
        and det.score is not None
        and float(det.score) >= confidence_floor
    ]
    return max(qualifying) if qualifying else 0.0


def aggregate_session_score(per_image_scores: Sequence[float]) -> float:
    """
    Session score = arithmetic mean of the top ceil(20%) per-image scores
    (minimum one image). Robust worst-case: captures a hotspot cluster
    without one outlier (pure max) or many clean images (pure mean)
    distorting the signal.
    """
    if not per_image_scores:
        raise ValueError("cannot aggregate an empty session")
    top_k = max(1, math.ceil(len(per_image_scores) * _TOP_FRACTION))
    top_scores = sorted(per_image_scores, reverse=True)[:top_k]
    return sum(top_scores) / len(top_scores)


def select_best_image(scored_images: Sequence[ScoredImage]) -> str:
    """
    Best-image selection: the image with the highest per-image score, or —
    when every image scores 0.0 (clean session) — a deterministic
    representative. ``max`` returns the FIRST element achieving the
    maximum when scores tie (Python's stable-max guarantee), and images
    are always passed in session (upload/registration) order, so a clean
    session's fallback is always its first image — deterministic and
    documented here as the fallback rule.
    """
    if not scored_images:
        raise ValueError("cannot select best image from an empty session")
    return max(scored_images, key=lambda item: item.score).image_id


def top_fraction_count(n: int) -> int:
    """Exposed for provenance/debugging: how many images the aggregation used."""
    return max(1, math.ceil(n * _TOP_FRACTION)) if n else 0
