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
Union-merge fusion stage, identified as ``FUSION_RULE_ID = "union-merge-v1"``. The score threshold is
applied FIRST -- that order is part of the rule, and it is why a
detection below the threshold can never bridge two regions. Then two
detections join when

* from different tiles: IoU >= ``iou``, or intersection >= ``contain`` x the
  smaller area; or, boundary-cut pieces, both touch their own tile's
  shared edge (within ``edge_tol`` px) from opposite sides with >=
  ``edge_extent`` overlap of the shorter extent along that edge;
* from the same tile: only at IoU >= ``same_tile_iou``.

Joins are transitive (connected components); a region's box is the union of
its members, its score the maximum member score.

The parameters are module constants, recorded in the session metadata and
never settings. The tile size comes from the artifact's recorded tiling.

A detection here is ``{"box": [x0, y0, x1, y1] frame px, "score": float,
"tile": tile index, "tile_xy": (x0, y0) tile origin}``.
"""

from __future__ import annotations

import collections
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

FUSION_RULE_ID = "union-merge-v1"
FUSION_PARAMS: Mapping[str, float] = MappingProxyType(
    {"iou": 0.30, "contain": 0.60, "edge_tol": 4, "edge_extent": 0.50, "same_tile_iou": 0.60}
)

FUSION_TILE = 1280


def collect(tile_outputs: Sequence[Tuple[int, int, int, Sequence[Sequence[float]], Sequence[float]]]) -> List[Dict[str, Any]]:
    """``(tile_index, x0, y0, boxes [[x0,y0,x1,y1] tile px], scores)`` per
    tile -> detections in frame coordinates."""
    dets = []
    for k, x0, y0, boxes, scores in tile_outputs:
        for b, s in zip(boxes, scores):
            dets.append({"box": [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], "score": float(s),
                         "tile": k, "tile_xy": (x0, y0)})
    return dets


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def _touch(box, tile_xy, edge, tol, tile=FUSION_TILE):
    x0, y0 = tile_xy
    return {"L": abs(box[0] - x0) <= tol, "R": abs(box[2] - (x0 + tile)) <= tol,
            "T": abs(box[1] - y0) <= tol, "B": abs(box[3] - (y0 + tile)) <= tol}[edge]


def _extent_overlap(a, b, axis):
    lo, hi = (1, 3) if axis == "y" else (0, 2)
    ov = min(a[hi], b[hi]) - max(a[lo], b[lo]); shorter = min(a[hi] - a[lo], b[hi] - b[lo])
    return ov / shorter if shorter > 0 else 0.0


def _join(d1, d2, p=FUSION_PARAMS, tile=FUSION_TILE):
    b1, b2 = d1["box"], d2["box"]
    if d1["tile"] == d2["tile"]:
        return _iou(b1, b2) >= p["same_tile_iou"]
    if _iou(b1, b2) >= p["iou"] or _inter(b1, b2) >= p["contain"] * min(_area(b1), _area(b2)):
        return True
    t1, t2 = d1["tile_xy"], d2["tile_xy"]; tol = p["edge_tol"]
    if t1[1] == t2[1] and t1[0] != t2[0]:
        left, right = (d1, d2) if t1[0] < t2[0] else (d2, d1)
        if _touch(left["box"], left["tile_xy"], "R", tol, tile) and _touch(right["box"], right["tile_xy"], "L", tol, tile) \
           and _extent_overlap(b1, b2, "y") >= p["edge_extent"]:
            return True
    if t1[0] == t2[0] and t1[1] != t2[1]:
        top, bot = (d1, d2) if t1[1] < t2[1] else (d2, d1)
        if _touch(top["box"], top["tile_xy"], "B", tol, tile) and _touch(bot["box"], bot["tile_xy"], "T", tol, tile) \
           and _extent_overlap(b1, b2, "x") >= p["edge_extent"]:
            return True
    return False


def union_merge(dets, p=FUSION_PARAMS, tile=FUSION_TILE):
    n = len(dets); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if _join(dets[i], dets[j], p, tile):
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(dets[i])
    return [{"box": [min(d["box"][0] for d in g), min(d["box"][1] for d in g),
                     max(d["box"][2] for d in g), max(d["box"][3] for d in g)],
             "score": max(d["score"] for d in g), "members": g} for g in groups.values()]


def predict_regions(dets, threshold, tile=FUSION_TILE):
    """Threshold FIRST, then merge (this order is part of the rule)."""
    return union_merge([d for d in dets if d["score"] >= threshold], tile=tile)
