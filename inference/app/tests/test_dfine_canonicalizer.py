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
Fixture tests for the shared D-FINE raw-to-canonical conversion.

D-FINE's export contract has no letterboxing, no ``xywh``
boxes, no re-sigmoid, and is NMS-free by construction.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pytest

from app.detectors.base import Detection
from app.detectors.dfine_common import (
    DFineCanonicalizationError,
    RawTensor,
    canonicalize_dfine,
)

CLASS_NAMES = ["aphid", "leaf"]


def _raw_tensors(
    labels: Sequence[int],
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    width: int,
    height: int,
    *,
    labels_dtype=np.int64,
    boxes_dtype=np.float32,
    scores_dtype=np.float32,
    sizes_dtype=np.int64,
) -> List[RawTensor]:
    """Natural-order raw tensors: labels, boxes, scores, orig_target_sizes."""
    return [
        RawTensor("labels", labels_dtype, np.array([labels], dtype=labels_dtype)),
        RawTensor("boxes", boxes_dtype, np.array([boxes], dtype=boxes_dtype)),
        RawTensor("scores", scores_dtype, np.array([scores], dtype=scores_dtype)),
        RawTensor(
            "orig_target_sizes", sizes_dtype, np.array([[width, height]], dtype=sizes_dtype)
        ),
    ]


class TestGoldenFixtureCorrectness:
    """
    Hand-computed on paper: a 200x100 (non-square) source image, three
    raw boxes — one fully in-bounds, one straddling the top-left corner
    (negative coordinates), one straddling the bottom-right corner
    (exceeding both bounds) — clip-then-divide applied per axis, by hand,
    proving the conversion is CORRECT, not merely internally consistent.
    """

    WIDTH, HEIGHT = 200, 100
    LABELS = [0, 1, 0]
    BOXES = [
        [10, 20, 190, 80],
        [-20, -10, 50, 40],
        [150, 90, 250, 150],
    ]
    SCORES = [0.91, 0.12, 0.55]

    def test_matches_hand_derived_expected_list(self):
        raw = _raw_tensors(self.LABELS, self.BOXES, self.SCORES, self.WIDTH, self.HEIGHT)

        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")

        expected = [
            Detection(x_min=10 / 200, y_min=20 / 100, x_max=190 / 200, y_max=80 / 100, score=0.91, label="aphid"),
            Detection(x_min=0.0, y_min=0.0, x_max=50 / 200, y_max=40 / 100, score=0.12, label="leaf"),
            Detection(x_min=150 / 200, y_min=90 / 100, x_max=1.0, y_max=1.0, score=0.55, label="aphid"),
        ]
        assert result.image_id == "img-1"
        assert len(result.detections) == 3
        for actual, exp in zip(result.detections, expected):
            assert actual.x_min == pytest.approx(exp.x_min, abs=1e-6)
            assert actual.y_min == pytest.approx(exp.y_min, abs=1e-6)
            assert actual.x_max == pytest.approx(exp.x_max, abs=1e-6)
            assert actual.y_max == pytest.approx(exp.y_max, abs=1e-6)
            assert actual.score == pytest.approx(exp.score, abs=1e-6)
            assert actual.label == exp.label


def _ort_shaped_call(labels, boxes, scores, width, height):
    """Mimics how an ORT executor assembles raw tensors: onnxruntime's
    session.run returns a list of arrays in the requested output_names
    order, each paired with its own queried dtype from the session's
    output metadata — here, output_names requested in ORT's own declared
    order (scores, labels, boxes)."""
    raw = [
        RawTensor("scores", np.float32, np.array([scores], dtype=np.float32)),
        RawTensor("labels", np.int64, np.array([labels], dtype=np.int64)),
        RawTensor("boxes", np.float32, np.array([boxes], dtype=np.float32)),
        RawTensor("orig_target_sizes", np.int64, np.array([[width, height]], dtype=np.int64)),
    ]
    return canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-x")


def _tensorrt_shaped_call(labels, boxes, scores, width, height):
    """Mimics how the Jetson executor assembles raw tensors: TensorRT
    engine.get_tensor_name(i) iterated by binding index, in the engine's
    own (unrelated) declaration order (orig_target_sizes, boxes, labels,
    scores) — a different order than the ORT call site above."""
    raw = [
        RawTensor("orig_target_sizes", np.int64, np.array([[width, height]], dtype=np.int64)),
        RawTensor("boxes", np.float32, np.array([boxes], dtype=np.float32)),
        RawTensor("labels", np.int64, np.array([labels], dtype=np.int64)),
        RawTensor("scores", np.float32, np.array([scores], dtype=np.float32)),
    ]
    return canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-x")


class TestTwoCallSitesByteIdentical:
    def test_ort_shaped_and_tensorrt_shaped_calls_agree_byte_identical(self):
        labels = [1, 0, 0]
        boxes = [[5, 5, 50, 50], [20, 10, 300, 90], [0, 0, 640, 640]]
        scores = [0.42, 0.87, 0.05]
        width, height = 320, 180

        ort_result = _ort_shaped_call(labels, boxes, scores, width, height)
        trt_result = _tensorrt_shaped_call(labels, boxes, scores, width, height)

        assert ort_result == trt_result


class TestNameDtypeBindingNotOrdinal:
    def test_reordered_raw_tensor_tuple_still_canonicalizes_correctly(self):
        """The tensors are handed in a scrambled order (scores first,
        labels last) — if the canonicalizer looked them up positionally
        instead of by name, this would silently corrupt the result."""
        raw = [
            RawTensor("scores", np.float32, np.array([[0.75]], dtype=np.float32)),
            RawTensor("orig_target_sizes", np.int64, np.array([[400, 200]], dtype=np.int64)),
            RawTensor("boxes", np.float32, np.array([[[40, 20, 360, 180]]], dtype=np.float32)),
            RawTensor("labels", np.int64, np.array([[0]], dtype=np.int64)),
        ]

        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-scrambled")

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.label == "aphid"
        assert det.score == pytest.approx(0.75)
        assert det.x_min == pytest.approx(40 / 400)
        assert det.y_min == pytest.approx(20 / 200)
        assert det.x_max == pytest.approx(360 / 400)
        assert det.y_max == pytest.approx(180 / 200)

    def test_missing_labels_tensor_raises(self):
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10)
        raw = [t for t in raw if t.name != "labels"]
        with pytest.raises(DFineCanonicalizationError, match="labels"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")

    def test_missing_boxes_tensor_raises(self):
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10)
        raw = [t for t in raw if t.name != "boxes"]
        with pytest.raises(DFineCanonicalizationError, match="boxes"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")

    def test_missing_scores_tensor_raises(self):
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10)
        raw = [t for t in raw if t.name != "scores"]
        with pytest.raises(DFineCanonicalizationError, match="scores"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")

    def test_missing_orig_target_sizes_tensor_raises(self):
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10)
        raw = [t for t in raw if t.name != "orig_target_sizes"]
        with pytest.raises(DFineCanonicalizationError, match="orig_target_sizes"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")


class TestElevenDFineProperties:
    def test_1_two_inputs_three_outputs_all_four_bound_tensors_required(self):
        """Structural: labels/boxes/scores (the 3 raw outputs) plus
        orig_target_sizes (1 of the 2 raw inputs — 'images' itself is
        preprocessing-only and never reaches the canonicalizer) are all
        required; a well-formed call with all four succeeds."""
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10)
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")
        assert len(result.detections) == 1

    def test_2_labels_must_be_integer_dtype(self):
        raw = _raw_tensors([0], [[1, 1, 2, 2]], [0.5], 10, 10, labels_dtype=np.float32)
        with pytest.raises(DFineCanonicalizationError, match="integer"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-1")

    def test_3_full_b300_shape_all_300_entries_survive(self):
        n = 300
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=n).tolist()
        boxes = rng.uniform(0, 640, size=(n, 4)).tolist()
        scores = rng.uniform(0.0, 1.0, size=n).tolist()

        raw = _raw_tensors(labels, boxes, scores, 640, 640)
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-300")

        assert len(result.detections) == 300

    def test_4_no_nms_two_heavily_overlapping_boxes_both_kept(self):
        """Two near-identical high-score boxes for the same class: a
        NMS-based pipeline would NMS-suppress one. D-FINE is NMS-free by
        construction — both survive."""
        raw = _raw_tensors(
            labels=[0, 0],
            boxes=[[100, 100, 140, 140], [101, 101, 141, 141]],
            scores=[0.95, 0.93],
            width=640,
            height=640,
        )
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-nms")
        assert len(result.detections) == 2

    def test_5_duplicate_query_box_under_two_labels_both_preserved(self):
        """The documented global-top-300 query x class-pair behavior: the
        same underlying box recurs under two different labels — both are
        distinct canonical Detection entries, not deduplicated."""
        raw = _raw_tensors(
            labels=[0, 1],
            boxes=[[50, 50, 150, 150], [50, 50, 150, 150]],
            scores=[0.6, 0.4],
            width=200,
            height=200,
        )
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-dup")

        assert len(result.detections) == 2
        assert result.detections[0].label == "aphid"
        assert result.detections[1].label == "leaf"
        assert (result.detections[0].x_min, result.detections[0].y_min) == (
            result.detections[1].x_min,
            result.detections[1].y_min,
        )
        assert result.detections[0].score != result.detections[1].score

    def test_6_dense_low_tail_scores_all_preserved_unfiltered(self):
        """A dense cluster of low scores — no floor applied here (scoring
        applies it exactly once, downstream); every entry survives."""
        n = 50
        scores = [0.01 + 0.001 * i for i in range(n)]
        labels = [0] * n
        boxes = [[1, 1, 2, 2]] * n

        raw = _raw_tensors(labels, boxes, scores, 100, 100)
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-lowtail")

        assert len(result.detections) == n
        assert [d.score for d in result.detections] == pytest.approx(scores)

    def test_7_non_square_orig_size_axes_never_swapped(self):
        """Width != height: x must divide by width, y by height — a
        swapped-axis bug would silently pass on a square fixture."""
        raw = _raw_tensors(
            labels=[0], boxes=[[30, 60, 90, 180]], scores=[0.5], width=300, height=200
        )
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-nonsquare")

        det = result.detections[0]
        assert det.x_min == pytest.approx(30 / 300)
        assert det.x_max == pytest.approx(90 / 300)
        assert det.y_min == pytest.approx(60 / 200)
        assert det.y_max == pytest.approx(180 / 200)

    def test_8_scores_at_and_near_confidence_floor_both_sides_untouched(self):
        """0.3 is a plausible confidence floor value — but the
        canonicalizer knows nothing about it; scores just below and just
        above are both preserved verbatim (floor application is scoring's
        job, not the canonicalizer's)."""
        raw = _raw_tensors(
            labels=[0, 0], boxes=[[1, 1, 2, 2], [1, 1, 2, 2]], scores=[0.29, 0.31], width=10, height=10
        )
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-floor")

        assert len(result.detections) == 2
        assert result.detections[0].score == pytest.approx(0.29)
        assert result.detections[1].score == pytest.approx(0.31)

    def test_9_canonical_scores_byte_equal_to_raw_sigmoid_no_redecode(self):
        """Raw scores are already sigmoid-decoded per the D-FINE export contract:
        the canonicalizer must pass them through byte-identical, never
        re-apply sigmoid or multiply by a separate objectness term. An
        arbitrary non-round float makes an accidental re-sigmoid or
        rescale immediately visible (sigmoid(0.7345629) != 0.7345629)."""
        raw_score = 0.7345629
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[raw_score], width=10, height=10)
        expected = float(np.float32(raw_score))

        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-sigmoid")

        assert result.detections[0].score == expected

    def test_10_exact_clip_then_divide_no_clipping_needed(self):
        """A box fully inside a non-square image: clip is a no-op, so this
        isolates the divide step's exactness."""
        raw = _raw_tensors(labels=[1], boxes=[[33, 17, 167, 83]], scores=[0.5], width=250, height=100)
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-exact")

        det = result.detections[0]
        assert det.x_min == pytest.approx(33 / 250, abs=1e-9)
        assert det.x_max == pytest.approx(167 / 250, abs=1e-9)
        assert det.y_min == pytest.approx(17 / 100, abs=1e-9)
        assert det.y_max == pytest.approx(83 / 100, abs=1e-9)

    def test_11_boxes_beyond_bounds_clipped_before_normalization(self):
        """A raw box wildly outside the source image bounds on all four
        edges — clipped to [0,width]/[0,height] BEFORE dividing, never
        rejected and never left out-of-range in the normalized output."""
        raw = _raw_tensors(labels=[0], boxes=[[-100, -50, 999, 999]], scores=[0.5], width=100, height=50)
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-oob")

        det = result.detections[0]
        assert det.x_min == pytest.approx(0.0)
        assert det.y_min == pytest.approx(0.0)
        assert det.x_max == pytest.approx(1.0)
        assert det.y_max == pytest.approx(1.0)


class TestShapeAndBoundsValidation:
    def test_zero_candidates_with_consistent_shapes_is_valid_and_empty(self):
        """[1,0]/[1,0,4] with matching candidate counts is a legitimate
        zero-detection result, not an error."""
        raw = [
            RawTensor("labels", np.int64, np.empty((1, 0), dtype=np.int64)),
            RawTensor("boxes", np.float32, np.empty((1, 0, 4), dtype=np.float32)),
            RawTensor("scores", np.float32, np.empty((1, 0), dtype=np.float32)),
            RawTensor("orig_target_sizes", np.int64, np.array([[10, 10]], dtype=np.int64)),
        ]
        result = canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="img-empty")
        assert result.detections == []

    def test_boxes_rank_2_rejected(self):
        raw = [
            RawTensor("labels", np.int64, np.empty((1, 0), dtype=np.int64)),
            RawTensor("boxes", np.float32, np.empty((1, 0), dtype=np.float32)),
            RawTensor("scores", np.float32, np.empty((1, 0), dtype=np.float32)),
            RawTensor("orig_target_sizes", np.int64, np.array([[10, 10]], dtype=np.int64)),
        ]
        with pytest.raises(DFineCanonicalizationError, match=r"\[B,N,4\]"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_boxes_last_dim_not_4_rejected(self):
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        raw[1] = RawTensor("boxes", np.float32, np.array([[[1, 1, 2]]], dtype=np.float32))
        with pytest.raises(DFineCanonicalizationError, match=r"\[B,N,4\]"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_labels_rank_1_rejected(self):
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        raw[0] = RawTensor("labels", np.int64, np.array([0], dtype=np.int64))
        with pytest.raises(DFineCanonicalizationError, match=r"labels.*\[B,N\]"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_sizes_last_dim_not_2_rejected(self):
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        raw[3] = RawTensor("orig_target_sizes", np.int64, np.array([[10, 10, 10]], dtype=np.int64))
        with pytest.raises(DFineCanonicalizationError, match=r"\[B,2\]"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_mismatched_candidate_counts_rejected(self):
        raw = _raw_tensors(labels=[0, 1], boxes=[[1, 1, 2, 2], [2, 2, 3, 3]], scores=[0.5, 0.6], width=10, height=10)
        raw[2] = RawTensor("scores", np.float32, np.array([[0.5]], dtype=np.float32))
        with pytest.raises(DFineCanonicalizationError, match="candidate counts"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_mismatched_batch_dims_rejected(self):
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        raw[3] = RawTensor(
            "orig_target_sizes", np.int64, np.array([[10, 10], [20, 20]], dtype=np.int64)
        )
        with pytest.raises(DFineCanonicalizationError, match="batch dimensions"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_batch_index_out_of_range_rejected(self):
        raw = _raw_tensors(labels=[0], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        with pytest.raises(DFineCanonicalizationError, match="batch_index"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x", batch_index=1)

    def test_negative_label_rejected_not_wrapped(self):
        """A negative raw label must raise — never silently select a class
        from the END of the manifest via Python index wrap-around."""
        raw = _raw_tensors(labels=[-1], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10)
        with pytest.raises(DFineCanonicalizationError, match="-1"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")

    def test_label_beyond_class_manifest_rejected(self):
        raw = _raw_tensors(
            labels=[len(CLASS_NAMES)], boxes=[[1, 1, 2, 2]], scores=[0.5], width=10, height=10
        )
        with pytest.raises(DFineCanonicalizationError, match="out of range"):
            canonicalize_dfine(raw, class_names=CLASS_NAMES, image_id="x")
