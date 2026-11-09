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
The tiled detect path shared by both D-FINE detector classes.

One frame -> exact native crops (``app/detectors/tiling.py``) -> chunks of
tiles run through the detector's own runtime with
``orig_target_sizes = [[imgsz, imgsz]]`` (boxes come back in tile pixels) ->
the runtime's RAW boxes, unclipped and unrounded, with the tile offset
added -> threshold, then the frozen union-merge fusion stage
(``app/detectors/region_merge.py``), which reads those raw coordinates in
its edge test -> one ``Detection`` per fused region. The only coordinate
transformation after fusion is the canonical mapping: the region box divided
by the frame size and clamped to [0, 1]. The label is the artifact's
``class_names[0]`` and the score the region's maximum member score. The
per-tile canonicalizer's clip-then-divide is NOT applied before fusion; only
its label-range refusal is kept.

With more than one test-time-augmentation view (``identity``, ``hflip``) each
chunk is run once per view; a mirrored view's boxes map back into tile
coordinates, and the views' detections are fused PER TILE by the same-tile
rule (IoU >= same_tile_iou -> union box, max score) before the cross-tile
stage, which therefore still sees one detection set per tile. With the
identity view alone that step is skipped and the path is the plain one.

Everything here is runtime-agnostic: a detector supplies ``run_batch``, its
own "these input tensors -> these raw output tensors" step, and this module
never touches a session, an engine or a thread. All of it runs on the one
detector-executor worker its caller is already on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.detectors.artifact_provenance import TilingSpec
from app.detectors.base import Detection, ImageDetections, ImageRef
from app.detectors.dfine_common import (
    BOXES_NAME,
    LABELS_NAME,
    SCORES_NAME,
    DFineCanonicalizationError,
    RawTensor,
)
from app.detectors.region_merge import (
    FUSION_PARAMS,
    FUSION_RULE_ID,
    collect,
    predict_regions,
    union_merge,
)
from app.detectors.tiling import grid_origins, load_frame_rgb, preprocess_tile, tile_frame

DEFAULT_TILE_BATCH = 4
DEFAULT_REGION_THRESHOLD = 0.40

SUPPORTED_VIEWS = ("identity", "hflip")

RunBatch = Callable[[np.ndarray, np.ndarray], List[RawTensor]]


@dataclass
class TiledResult:
    detections: List[Detection]
    regions: List[Dict]
    frame_size: Tuple[int, int]
    origins: List[Tuple[int, int]]
    raw_tiles: Optional[List[Dict]] = field(default=None)


def _raw_row(raw_tensors: Sequence[RawTensor], row: int) -> Tuple[List[int], List[List[float]], List[float]]:
    """One tile's runtime output, exactly as the runtime returned it: raw
    tile-pixel boxes (float, unclipped, unrounded), labels, sigmoid scores."""
    by_name = {t.name: t.array for t in raw_tensors}
    missing = [name for name in (LABELS_NAME, BOXES_NAME, SCORES_NAME) if name not in by_name]
    if missing:
        raise DFineCanonicalizationError(f"missing required raw tensor(s) {missing!r}; got {sorted(by_name)!r}")
    labels, boxes, scores = by_name[LABELS_NAME], by_name[BOXES_NAME], by_name[SCORES_NAME]
    if labels.ndim != 2 or scores.ndim != 2 or boxes.ndim != 3 or boxes.shape[2] != 4:
        raise DFineCanonicalizationError(
            "labels/scores must be [B,N] and boxes [B,N,4], got "
            f"{labels.shape!r}/{scores.shape!r}/{boxes.shape!r}"
        )
    if not labels.shape[0] == boxes.shape[0] == scores.shape[0] or not labels.shape[1] == boxes.shape[1] == scores.shape[1]:
        raise DFineCanonicalizationError(
            f"inconsistent shapes across labels/boxes/scores: {labels.shape!r}/{boxes.shape!r}/{scores.shape!r}"
        )
    if not 0 <= row < labels.shape[0]:
        raise DFineCanonicalizationError(f"batch_index {row} out of range for batch dimension {labels.shape[0]}")
    return (
        [int(v) for v in labels[row]],
        [[float(v) for v in box] for box in boxes[row]],
        [float(v) for v in scores[row]],
    )


def _require_labels_in_range(labels: Sequence[int], class_names: Sequence[str], image_id: str) -> None:
    """The label check the per-tile canonicalizer made: a raw label that does
    not index ``class_names`` (a negative one never wraps around) refuses by
    the canonicalizer's own named error."""
    for candidate, label in enumerate(labels):
        if not 0 <= label < len(class_names):
            raise DFineCanonicalizationError(
                f"label index {label} out of range for class manifest of "
                f"{len(class_names)} classes (candidate {candidate}, image {image_id})"
            )


def run_tiled(
    image_ref: ImageRef,
    *,
    tiling: TilingSpec,
    imgsz: int,
    class_names: Sequence[str],
    run_batch: RunBatch,
    batch: int,
    threshold: float,
    keep_raw: bool = False,
    views: Sequence[str] = ("identity",),
) -> TiledResult:
    if "identity" not in views or len(set(views)) != len(views) or any(v not in SUPPORTED_VIEWS for v in views):
        raise ValueError(
            f"tta views must be distinct members of {SUPPORTED_VIEWS!r} including 'identity', got {list(views)!r}"
        )
    frame = load_frame_rgb(image_ref.path)
    tiles = tile_frame(
        frame,
        tiling.tile,
        tiling.stride,
        image=f"{image_ref.filename} (image_id {image_ref.image_id})",
    )
    height, width = frame.shape[:2]
    stem = Path(image_ref.filename).stem

    tile_outputs = []
    raw_tiles: List[Dict] = []
    for start in range(0, len(tiles), batch):
        chunk = tiles[start : start + batch]
        images = np.stack([preprocess_tile(crop) for _, _, _, crop in chunk])
        orig_target_sizes = np.full((len(chunk), 2), imgsz, dtype=np.int64)
        per_view = []
        for view in views:
            view_images = images if view == "identity" else np.ascontiguousarray(images[:, :, :, ::-1])
            raw_tensors = run_batch(view_images, orig_target_sizes)
            per_view.append(raw_tensors)
            if keep_raw and view == "identity":
                identity_raw = raw_tensors
        for row, (index, x0, y0, _) in enumerate(chunk):
            tile_dets: List[Dict] = []
            for view, raw_tensors in zip(views, per_view):
                labels, raw_boxes, scores = _raw_row(raw_tensors, row)
                _require_labels_in_range(labels, class_names, image_ref.image_id)
                for box, score in zip(raw_boxes, scores):
                    if view == "hflip":
                        box = [tiling.tile - box[2], box[1], tiling.tile - box[0], box[3]]
                    tile_dets.append({"box": box, "score": score, "tile": index, "tile_xy": (x0, y0)})
            if len(views) > 1:
                tile_dets = union_merge(
                    [d for d in tile_dets if d["score"] >= threshold], tile=tiling.tile
                )
            tile_outputs.append(
                (index, x0, y0, [d["box"] for d in tile_dets], [d["score"] for d in tile_dets])
            )
            if keep_raw:
                labels, raw_boxes, scores = _raw_row(identity_raw, row)
                raw_tiles.append(
                    {
                        "tile": f"{stem}_t{index:02d}_x{x0}_y{y0}",
                        "labels": labels,
                        "boxes": raw_boxes,
                        "scores": scores,
                    }
                )

    regions = predict_regions(collect(tile_outputs), threshold, tile=tiling.tile)
    detections = [
        Detection(
            x_min=min(max(region["box"][0], 0.0), width) / width,
            y_min=min(max(region["box"][1], 0.0), height) / height,
            x_max=min(max(region["box"][2], 0.0), width) / width,
            y_max=min(max(region["box"][3], 0.0), height) / height,
            score=region["score"],
            label=class_names[0],
        )
        for region in regions
    ]
    return TiledResult(
        detections=detections,
        regions=[
            {"box": list(region["box"]), "score": region["score"], "n_members": len(region["members"])}
            for region in regions
        ],
        frame_size=(width, height),
        origins=[(x0, y0) for _, x0, y0, _ in tiles],
        raw_tiles=raw_tiles if keep_raw else None,
    )


def frame_size_of(image_path: Path) -> Tuple[int, int]:
    """(width, height) from the image header alone, without decoding pixels."""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def tiled_path_fields(
    *,
    profile: str,
    tiling: Optional[TilingSpec],
    frame_sizes: Sequence[Tuple[int, int]],
    views: Sequence[str],
    threshold: float,
    negligibility_record_sha256: Optional[str],
) -> Dict:
    """The tiled-path provenance a detector reports (session metadata and
    ``/internal/detect-debug``); the wire contract does not carry it.

    ``tiling.origins`` is the grid of the first frame; ``tiling.grids`` maps
    every distinct frame size seen (``"WxH"``) to the origins actually used
    for it. On the full-frame path ``tiling`` and the fusion fields are null.
    """
    fields: Dict = {
        "detector_profile": profile,
        "tiling": None,
        "fusion_rule_id": None,
        "fusion_params": None,
        "tta_views": list(views),
        "region_threshold": threshold,
        "negligibility_record_sha256": negligibility_record_sha256,
    }
    if tiling is None:
        return fields
    grids: Dict[str, List[List[int]]] = {}
    for width, height in frame_sizes:
        grids[f"{width}x{height}"] = [
            [x0, y0]
            for y0 in grid_origins(height, tiling.tile, tiling.stride)
            for x0 in grid_origins(width, tiling.tile, tiling.stride)
        ]
    fields["tiling"] = {
        "tile": tiling.tile,
        "stride": tiling.stride,
        "origins": next(iter(grids.values()), []),
        "grids": grids,
    }
    fields["fusion_rule_id"] = FUSION_RULE_ID
    fields["fusion_params"] = dict(FUSION_PARAMS)
    return fields


class TiledPathMixin:
    """The tiled-path surface both detector classes expose. A class mixing
    this in provides ``_run_tiled(image_ref, keep_raw)`` and carries
    ``_tiling``, ``_profile_name``, ``_tta_views``, ``_region_threshold`` and
    ``_negligibility_record_sha256``."""

    @property
    def is_tiled(self) -> bool:
        return self._tiling is not None

    def _tiled_detections(self, image_ref: ImageRef) -> ImageDetections:
        result = self._run_tiled(image_ref)
        return ImageDetections(image_id=image_ref.image_id, detections=result.detections)

    def detect_tiled_debug(self, image_ref: ImageRef) -> Dict:
        """Per-tile raw outputs (raw per-tile labels/boxes/scores, tile pixels), the
        fused regions in frame pixels, and the tiled-path provenance fields
        for this frame -- ``/internal/detect-debug?tiles=1``."""
        result = self._run_tiled(image_ref, keep_raw=True)
        return {
            "detections": [
                {
                    "label": d.label,
                    "score": d.score,
                    "x_min": d.x_min,
                    "y_min": d.y_min,
                    "x_max": d.x_max,
                    "y_max": d.y_max,
                }
                for d in result.detections
            ],
            "tiles": result.raw_tiles,
            "regions": result.regions,
            **self.tiled_path_provenance([result.frame_size]),
        }

    def tiled_path_provenance(self, frame_sizes: Sequence[Tuple[int, int]] = ()) -> Dict:
        """Session-metadata / debug provenance of the (tiled or full-frame)
        path this detector runs; ``frame_sizes`` names the frames whose
        grids are reported. Not part of the wire contract."""
        return tiled_path_fields(
            profile=self._profile_name,
            tiling=self._tiling,
            frame_sizes=frame_sizes,
            views=self._tta_views,
            threshold=self._region_threshold,
            negligibility_record_sha256=self._negligibility_record_sha256,
        )
