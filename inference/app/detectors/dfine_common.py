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
Shared D-FINE raw-to-canonical conversion.

``canonicalize_dfine`` is the ONE place that converts a D-FINE model's raw
two-input/three-output tensors into the canonical ``Detection`` list — used
identically by both ``dfine_ort.DFineORTDetector`` (dev CPU / cloud CUDA)
and ``dfine_tensorrt.DFineTensorRTDetector`` (Jetson). Neither executor
implements any decode logic of its own; each only assembles the raw
tensors and
hands them here.

Per the D-FINE export contract, this module:
  treats ``scores`` as already sigmoid-decoded (no re-sigmoid, no
    objectness multiply);
  treats ``boxes`` as already-decoded original-image-pixel ``xyxy`` (no
    format reinterpretation, e.g. no ``xywh`` assumption);
  applies no non-maximum suppression, and no additional top-k filtering:
    every raw query×class candidate is canonicalized and passed on;
  applies no confidence floor — that is scoring's job
    (``app/scoring.py``), applied exactly once, downstream of this module;
  converts pixel ``xyxy`` to the canonical normalized ``[0,1]`` ``xyxy``
    convention, exactly once, per axis: each coordinate
    is first clipped to ``[0, width]``/``[0, height]`` and then divided by
    that axis's own dimension — raw endpoints outside the image are clipped, never rejected and
    never preserved out-of-range.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from PIL import Image

from app.detectors.base import Detection, ImageDetections

LABELS_NAME = "labels"
BOXES_NAME = "boxes"
SCORES_NAME = "scores"
ORIG_TARGET_SIZES_NAME = "orig_target_sizes"

_REQUIRED_TENSOR_NAMES = (LABELS_NAME, BOXES_NAME, SCORES_NAME, ORIG_TARGET_SIZES_NAME)


class DFineCanonicalizationError(ValueError):
    """
    Raised when the raw tensors handed to ``canonicalize_dfine`` don't
    conform to the pinned D-FINE contract — a missing named tensor, a
    tensor whose queried dtype doesn't match what the contract declares
    (``labels``/``orig_target_sizes`` integer, ``boxes``/``scores``
    floating), a malformed rank/shape (labels/scores ``[B,N]``, boxes
    ``[B,N,4]``, sizes ``[B,2]``), inconsistent batch or candidate counts
    across the tensors, or a label index outside the class manifest.
    Binding is always by declared name and queried dtype, never ordinal
    position — this is what that binding step enforces.
    """


class RawTensor(NamedTuple):
    """
    One raw D-FINE tensor as a runtime executor discovers it: its declared
    name (from the ONNX/TensorRT binding metadata), its queried dtype (also
    from that metadata — never assumed from the array alone), and the
    actual array. Both executors assemble a ``Sequence[RawTensor]`` from
    their own runtime's name/dtype APIs and hand it to
    ``canonicalize_dfine`` — the sequence's order is never significant.
    """

    name: str
    dtype: np.dtype
    array: np.ndarray


def preprocess_image_dfine(image_path: Path, imgsz: int = 640) -> Tuple[np.ndarray, int, int]:
    """
    Direct resize to ``imgsz x imgsz``, RGB, ``[0,1]`` — the ONLY
    preprocessing path the D-FINE contract allows (no letterboxing, no
    mean/std normalization). Returns the ``(1,3,imgsz,imgsz)`` float32 CHW
    batch plus the source image's own ``(width, height)``, which both
    executors pass through as the raw ``orig_target_sizes`` input.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        resized = img.resize((imgsz, imgsz), Image.BILINEAR)

    chw = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    batch = np.expand_dims(chw, axis=0)
    return batch, orig_w, orig_h


def _require_dtype_kind(tensor: RawTensor, kind: str, expected_desc: str) -> None:
    if np.dtype(tensor.dtype).kind != kind:
        raise DFineCanonicalizationError(
            f"raw tensor {tensor.name!r} must be {expected_desc}, got dtype {tensor.dtype!r}"
        )


def canonicalize_dfine(
    raw_tensors: Sequence[RawTensor],
    class_names: Sequence[str],
    image_id: str,
    batch_index: int = 0,
) -> ImageDetections:
    """
    Convert one image's raw D-FINE ``labels``/``boxes``/``scores``/
    ``orig_target_sizes`` tensors into a canonical ``ImageDetections``.

    Every raw tensor is looked up in ``raw_tensors`` by its declared
    ``name`` and validated by its queried ``dtype`` — never by position in
    the sequence; a caller may pass the four tensors in any
    order.

    All raw query×class candidates are canonicalized (no top-k filtering,
    no confidence floor, no NMS) — scoring applies the floor exactly once,
    downstream (``app/scoring.py``).
    """
    by_name = {tensor.name: tensor for tensor in raw_tensors}
    missing = [name for name in _REQUIRED_TENSOR_NAMES if name not in by_name]
    if missing:
        raise DFineCanonicalizationError(
            f"missing required raw tensor(s) {missing!r}; got {sorted(by_name)!r}"
        )

    labels_t = by_name[LABELS_NAME]
    boxes_t = by_name[BOXES_NAME]
    scores_t = by_name[SCORES_NAME]
    sizes_t = by_name[ORIG_TARGET_SIZES_NAME]

    _require_dtype_kind(labels_t, "i", "an integer dtype")
    _require_dtype_kind(boxes_t, "f", "a floating dtype")
    _require_dtype_kind(scores_t, "f", "a floating dtype")
    _require_dtype_kind(sizes_t, "i", "an integer dtype")

    if labels_t.array.ndim != 2:
        raise DFineCanonicalizationError(
            f"labels must be a [B,N] rank-2 tensor, got shape {labels_t.array.shape!r}"
        )
    if scores_t.array.ndim != 2:
        raise DFineCanonicalizationError(
            f"scores must be a [B,N] rank-2 tensor, got shape {scores_t.array.shape!r}"
        )
    if boxes_t.array.ndim != 3 or boxes_t.array.shape[2] != 4:
        raise DFineCanonicalizationError(
            f"boxes must be a [B,N,4] rank-3 tensor, got shape {boxes_t.array.shape!r}"
        )
    if sizes_t.array.ndim != 2 or sizes_t.array.shape[1] != 2:
        raise DFineCanonicalizationError(
            f"orig_target_sizes must be a [B,2] rank-2 tensor, got shape {sizes_t.array.shape!r}"
        )

    batch_sizes = {
        LABELS_NAME: labels_t.array.shape[0],
        BOXES_NAME: boxes_t.array.shape[0],
        SCORES_NAME: scores_t.array.shape[0],
        ORIG_TARGET_SIZES_NAME: sizes_t.array.shape[0],
    }
    if len(set(batch_sizes.values())) != 1:
        raise DFineCanonicalizationError(
            f"inconsistent batch dimensions across raw tensors: {batch_sizes!r}"
        )
    batch = labels_t.array.shape[0]
    if not 0 <= batch_index < batch:
        raise DFineCanonicalizationError(
            f"batch_index {batch_index} out of range for batch dimension {batch}"
        )

    candidate_counts = {
        LABELS_NAME: labels_t.array.shape[1],
        BOXES_NAME: boxes_t.array.shape[1],
        SCORES_NAME: scores_t.array.shape[1],
    }
    if len(set(candidate_counts.values())) != 1:
        raise DFineCanonicalizationError(
            f"inconsistent candidate counts across labels/boxes/scores: {candidate_counts!r}"
        )

    orig_width, orig_height = (int(v) for v in sizes_t.array[batch_index])
    if orig_width <= 0 or orig_height <= 0:
        raise DFineCanonicalizationError(
            f"orig_target_sizes must be positive [width,height], got ({orig_width}, {orig_height})"
        )

    labels = labels_t.array[batch_index]
    boxes = boxes_t.array[batch_index]
    scores = scores_t.array[batch_index]

    detections: list[Detection] = []
    for i in range(labels.shape[0]):
        label_index = int(labels[i])
        if not 0 <= label_index < len(class_names):
            raise DFineCanonicalizationError(
                f"label index {label_index} out of range for class manifest of "
                f"{len(class_names)} classes (candidate {i})"
            )

        x_min, y_min, x_max, y_max = (float(v) for v in boxes[i])

        x_min = min(max(x_min, 0.0), orig_width) / orig_width
        x_max = min(max(x_max, 0.0), orig_width) / orig_width
        y_min = min(max(y_min, 0.0), orig_height) / orig_height
        y_max = min(max(y_max, 0.0), orig_height) / orig_height

        detections.append(
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                score=float(scores[i]),
                label=class_names[label_index],
            )
        )

    return ImageDetections(image_id=image_id, detections=detections)


DFINE_STOCK_COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class ClassMapResolutionError(RuntimeError):
    """
    Raised at detector construction when the class list this executor would serve cannot
    be verified against its artifact provenance record's ``class_map_hash``:
    a record carrying ``class_names`` whose hash disagrees with its own
    ``class_map_hash``, a record without ``class_names`` whose stock-list
    hash disagrees with ``class_map_hash``, or a record carrying no
    ``class_map_hash`` at all. Always names both hash values (or the
    field's absence) -- one construction-time rule, fail-closed in both
    directions, never a lenient branch that serves an unverified list.
    """


def _class_map_hash(class_names: Sequence[str]) -> str:
    """
    Canonical class-map hash: sha256 of the newline-joined list, the form recorded in the
    provenance ``class_map_hash``.
    """
    return hashlib.sha256("\n".join(class_names).encode("utf-8")).hexdigest()


def resolve_class_names(provenance: Dict[str, Any]) -> List[str]:
    """
    Resolve the class list a D-FINE executor serves from its artifact
    provenance record, at construction -- the ONE rule both executors share:

    a record carrying ``class_names`` serves those names, but only if
      the field is a well-shaped (non-empty list of non-empty,
      newline-free strings) list whose hash equals the record's own
      ``class_map_hash``;
    a record without ``class_names`` serves the stock COCO list, but
      only if the stock list's hash equals the record's ``class_map_hash``;
    a record carrying no ``class_map_hash`` at all is refused outright:
      there is nothing to verify either path against.

    Presence of the field is decided by KEY membership (``"class_names" in
    provenance``), never truthiness -- a present-but-malformed shape
    (``[]``, ``[""]``, a newline-bearing member, a bare string, a non-list
    container) must REFUSE naming what was found, not silently fall back
    to the stock list as if the field were absent.

    Either a malformed-shape or hash-disagreement finding, or the
    missing-``class_map_hash`` case, raises ``ClassMapResolutionError``
    naming what was found -- a runtime that would serve a wrong class list
    simply cannot construct.
    """
    class_map_hash = provenance.get("class_map_hash")
    if not class_map_hash:
        raise ClassMapResolutionError(
            "artifact provenance record carries no 'class_map_hash' -- a class "
            "list can never be resolved without one to verify against"
        )

    if "class_names" in provenance:
        class_names = provenance["class_names"]
        malformed = (
            not isinstance(class_names, list)
            or not class_names
            or any(
                not isinstance(member, str) or member == "" or "\n" in member
                for member in class_names
            )
        )
        if malformed:
            raise ClassMapResolutionError(
                "record's 'class_names' field is present but is not a non-empty "
                "list of non-empty, newline-free strings -- refusing to fall back "
                f"to the stock list for a malformed present value: {class_names!r}"
            )
        computed_hash = _class_map_hash(class_names)
        if computed_hash != class_map_hash:
            raise ClassMapResolutionError(
                f"record's class_names hash {computed_hash!r} does not match its own "
                f"class_map_hash {class_map_hash!r} -- refusing to serve a class list "
                "that disagrees with its own provenance record"
            )
        return list(class_names)

    stock_hash = _class_map_hash(DFINE_STOCK_COCO_CLASS_NAMES)
    if stock_hash != class_map_hash:
        raise ClassMapResolutionError(
            f"record carries no class_names; the stock COCO list's hash {stock_hash!r} "
            f"does not match the record's class_map_hash {class_map_hash!r} -- refusing "
            "to serve the stock fallback against a record that does not describe it"
        )
    return list(DFINE_STOCK_COCO_CLASS_NAMES)
