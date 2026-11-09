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

"""Unit tests for deterministic scoring/aggregation and
best-image selection."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.scoring import ScoredImage, aggregate_session_score, per_image_score, select_best_image


@dataclass(frozen=True)
class _Det:
    label: str
    score: float


class TestPerImageScore:
    def test_max_qualifying_confidence(self):
        dets = [_Det("aphid", 0.3), _Det("aphid", 0.9), _Det("aphid", 0.5)]
        assert per_image_score(dets, ["aphid"], 0.0) == 0.9

    def test_excludes_non_target_class(self):
        dets = [_Det("leaf", 0.99), _Det("aphid", 0.4)]
        assert per_image_score(dets, ["aphid"], 0.0) == 0.4

    def test_excludes_below_floor(self):
        dets = [_Det("aphid", 0.2)]
        assert per_image_score(dets, ["aphid"], 0.3) == 0.0

    def test_at_floor_included(self):
        dets = [_Det("aphid", 0.3)]
        assert per_image_score(dets, ["aphid"], 0.3) == 0.3

    def test_no_detections_zero(self):
        assert per_image_score([], ["aphid"], 0.3) == 0.0

    def test_no_qualifying_detections_zero(self):
        dets = [_Det("leaf", 0.9)]
        assert per_image_score(dets, ["aphid"], 0.0) == 0.0


class TestAggregateSessionScore:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_session_score([])

    def test_reproducible(self):
        scores = [0.1, 0.9, 0.4, 0.2, 0.05]
        assert aggregate_session_score(scores) == aggregate_session_score(scores)

    def test_single_image_is_its_own_score(self):
        assert aggregate_session_score([0.7]) == 0.7

    def test_top_20_percent_min_one(self):
        scores = [0.1, 0.2, 0.3, 0.4, 1.0]
        assert aggregate_session_score(scores) == 1.0

    def test_five_or_fewer_images_is_pure_max(self):
        for n in range(1, 6):
            clean_tail = [0.05] * (n - 1)
            assert aggregate_session_score(clean_tail + [0.83]) == 0.83

    def test_hotspot_outranks_uniform_noise(self):
        """Two images @0.9 among ten zeros beat twelve @0.2."""
        hotspot = [0.9, 0.9] + [0.0] * 10
        uniform = [0.2] * 12
        assert aggregate_session_score(hotspot) > aggregate_session_score(uniform)

    def test_pure_max_would_be_wrong_baseline_sanity(self):
        hotspot = [0.9, 0.9] + [0.0] * 10
        assert aggregate_session_score(hotspot) < max(hotspot)


class TestSelectBestImage:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_best_image([])

    def test_highest_score_wins(self):
        images = [ScoredImage("a", 0.1), ScoredImage("b", 0.9), ScoredImage("c", 0.5)]
        assert select_best_image(images) == "b"

    def test_clean_session_falls_back_to_first_in_order(self):
        images = [ScoredImage("first", 0.0), ScoredImage("second", 0.0), ScoredImage("third", 0.0)]
        assert select_best_image(images) == "first"

    def test_tie_breaks_to_first_occurrence(self):
        images = [ScoredImage("a", 0.5), ScoredImage("b", 0.5)]
        assert select_best_image(images) == "a"

    def test_single_image(self):
        assert select_best_image([ScoredImage("only", 0.0)]) == "only"
