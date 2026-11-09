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
Native tiling grid.

An artifact whose provenance declares ``tiling: {tile, stride}`` is run on
unscaled crops of the frame. Along each axis the origins are ``0, stride,
2*stride, ...`` while ``origin + tile < extent``, followed by ``extent - tile``
when the last origin does not already reach the edge, so the last crop is
always edge-aligned. For a 4056x3040 frame with tile 1280 and
stride 1024 the rule yields X in {0, 1024, 2048, 2776}, Y in {0, 1024, 1760}:
    twelve origins, as a consequence of the rule
and not a lookup table.

Crops are exact pixel slices: no resize, no padding, no letterbox. A frame
with an extent below the tile size is refused by name (``TilingRefusal``);
the service never silently resizes or pads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


class TilingRefusal(RuntimeError):
    """A frame that does not fit the tiling grid, named by image, frame
    size (``WxH``, width first) and tile size."""


def grid_origins(extent: int, tile: int, stride: int) -> List[int]:
    """Origins along one axis of length ``extent`` (see the module docstring)."""
    if tile <= 0 or not 0 < stride <= tile:
        raise ValueError(f"tiling needs tile > 0 and 0 < stride <= tile, got tile={tile}, stride={stride}")
    if extent < tile:
        raise ValueError(f"extent {extent} is below the tile size {tile}")
    origins: List[int] = []
    origin = 0
    while origin + tile < extent:
        origins.append(origin)
        origin += stride
    edge = extent - tile
    if not origins or origins[-1] != edge:
        origins.append(edge)
    return origins


def tile_frame(
    frame_hwc_uint8: np.ndarray, tile: int, stride: int, image: str = "<unnamed>"
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Row-major (Y outer, X inner) ``(index, x0, y0, crop)`` for every grid
    tile, each ``crop`` an exact ``tile x tile`` part of the frame."""
    shape = frame_hwc_uint8.shape
    if len(shape) != 3 or shape[2] != 3:
        raise TilingRefusal(
            f"image {image!r}: a tiled artifact needs an HxWx3 RGB frame, got an array of shape {shape!r}"
        )
    height, width = shape[0], shape[1]
    if width < tile or height < tile:
        raise TilingRefusal(
            f"image {image!r}: frame {width}x{height} is smaller than one {tile}x{tile} tile -- "
            "refusing rather than resizing or padding"
        )
    tiles: List[Tuple[int, int, int, np.ndarray]] = []
    for y0 in grid_origins(height, tile, stride):
        for x0 in grid_origins(width, tile, stride):
            tiles.append((len(tiles), x0, y0, frame_hwc_uint8[y0 : y0 + tile, x0 : x0 + tile]))
    return tiles


def preprocess_tile(crop_hwc_uint8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB crop -> float32 3xHxW in [0, 1]: ``uint8 / 255`` and
    nothing else (no mean/std normalization)."""
    return np.ascontiguousarray((crop_hwc_uint8.astype(np.float32) / 255.0).transpose(2, 0, 1))


def load_frame_rgb(image_path: Path) -> np.ndarray:
    """Decode an image file to a uint8 HxWx3 RGB array."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
