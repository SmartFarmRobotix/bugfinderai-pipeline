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

"""Native tiling grid."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.detectors.tiling import (
    TilingRefusal,
    grid_origins,
    preprocess_tile,
    tile_frame,
)

PINNED_ORIGINS = [
    (0, 0), (1024, 0), (2048, 0), (2776, 0),
    (0, 1024), (1024, 1024), (2048, 1024), (2776, 1024),
    (0, 1760), (1024, 1760), (2048, 1760), (2776, 1760),
]


def _frame(width: int, height: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_the_raspicam_frame_yields_the_twelve_pinned_origins_in_order():
    frame = _frame(4056, 3040)
    tiles = tile_frame(frame, 1280, 1024, image="frame.jpg")
    assert [(x0, y0) for _, x0, y0, _ in tiles] == PINNED_ORIGINS
    assert [index for index, _, _, _ in tiles] == list(range(12))
    assert all(crop.shape == (1280, 1280, 3) for _, _, _, crop in tiles)


def test_grid_origins_axes_of_the_pinned_case():
    assert grid_origins(4056, 1280, 1024) == [0, 1024, 2048, 2776]
    assert grid_origins(3040, 1280, 1024) == [0, 1024, 1760]


@pytest.mark.parametrize(
    ("extent", "tile", "stride", "expected"),
    [
        (1280, 1280, 1024, [0]),
        (2304, 1280, 1024, [0, 1024]),
        (2305, 1280, 1024, [0, 1024, 1025]),
        (2560, 1280, 1280, [0, 1280]),
        (100, 10, 3, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90]),
    ],
)
def test_grid_origins_rule(extent, tile, stride, expected):
    origins = grid_origins(extent, tile, stride)
    assert origins == expected
    assert origins[-1] == extent - tile


def test_tiles_are_bit_exact_crops():
    frame = _frame(4056, 3040)
    for index, x0, y0, crop in tile_frame(frame, 1280, 1024, image="frame.jpg"):
        expected = frame[y0:y0 + 1280, x0:x0 + 1280]
        assert crop.dtype == np.uint8
        assert hashlib.sha256(crop.tobytes()).hexdigest() == hashlib.sha256(expected.tobytes()).hexdigest()


def test_preprocessed_tile_is_uint8_over_255_chw_without_normalization():
    frame = _frame(1280, 1280)
    (index, x0, y0, crop), = tile_frame(frame, 1280, 1024, image="frame.jpg")
    tensor = preprocess_tile(crop)
    assert tensor.dtype == np.float32
    assert tensor.shape == (3, 1280, 1280)
    assert np.array_equal(tensor, (crop.astype(np.float32) / 255.0).transpose(2, 0, 1))
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_a_frame_smaller_than_one_tile_is_refused_naming_image_frame_and_tile():
    frame = _frame(1280, 720)
    with pytest.raises(TilingRefusal) as excinfo:
        tile_frame(frame, 1280, 1024, image="small.jpg")
    message = str(excinfo.value)
    assert "small.jpg" in message
    assert "1280x720" in message
    assert "1280" in message


@pytest.mark.parametrize("shape", [(3040, 4056), (3040, 4056, 4), (3040, 4056, 1)])
def test_a_non_three_channel_array_is_refused(shape):
    frame = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(TilingRefusal, match="frame.png"):
        tile_frame(frame, 1280, 1024, image="frame.png")


def test_the_refusal_is_never_a_resize_or_a_pad():
    frame = _frame(1279, 3040)
    with pytest.raises(TilingRefusal, match="1279x3040"):
        tile_frame(frame, 1280, 1024, image="narrow.jpg")


@pytest.mark.parametrize(("tile", "stride"), [(0, 1), (-1, 1), (1280, 0), (1280, 1281), (1280, -5)])
def test_grid_origins_refuses_a_grid_the_provenance_validator_would_refuse(tile, stride):
    with pytest.raises(ValueError):
        grid_origins(4056, tile, stride)
