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

"""`image_cap.cap_image_bytes`
caps a single best-image variant to a byte bound the same way the Mixed
scenario's push already caps its pair before it leaves the device. No
fixture bytes in git -- the cells generate their own JPEGs with Pillow.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from image_cap import cap_image_bytes, cap_pair


def _large_noisy_jpeg(width: int = 4056, height: int = 3040, quality: int = 95) -> bytes:
    """A high-resolution JPEG that is genuinely several MB -- random noise
    (not a flat gradient) so JPEG's block-DCT compression cannot flatten it
    to near-nothing, mirroring a real camera capture's entropy."""
    rng = np.random.default_rng(12345)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _elongated_noisy_jpeg(width: int = 50_000, height: int = 64, quality: int = 95) -> bytes:
    """An extreme aspect-ratio JPEG -- its shorter side (64 px) sits right
    at the old, now-removed minimum-dimension floor, so this is the shape
    that exposed the bug where the capping walk gave up and returned an
    still-oversized image once a side reached that floor."""
    rng = np.random.default_rng(12345)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _small_jpeg(target_bytes: int = 50_000) -> bytes:
    """A JPEG built to land close to (and under) `target_bytes`."""
    img = Image.new("RGB", (300, 225), color=(120, 60, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    data = buf.getvalue()
    assert len(data) <= target_bytes, "fixture JPEG grew past the intended small size"
    return data


class TestCapImageBytes:
    def test_large_image_caps_to_the_bound_and_stays_a_valid_jpeg(self):
        original = _large_noisy_jpeg()
        assert len(original) > 100_000

        capped = cap_image_bytes(original, max_bytes=100_000)

        assert len(capped) <= 100_000
        decoded = Image.open(io.BytesIO(capped))
        decoded.load()
        assert decoded.format == "JPEG"

    def test_small_image_passes_through_unchanged(self):
        original = _small_jpeg()
        capped = cap_image_bytes(original, max_bytes=100_000)
        assert capped == original
        assert capped is original

    def test_elongated_image_caps_to_the_bound_with_no_floor(self):
        original = _elongated_noisy_jpeg()
        assert len(original) > 100_000

        capped = cap_image_bytes(original, max_bytes=100_000)

        assert len(capped) <= 100_000
        decoded = Image.open(io.BytesIO(capped))
        decoded.load()
        assert decoded.format == "JPEG"


class TestCapPair:
    def test_both_variants_capped_independently(self):
        clean = _large_noisy_jpeg()
        annotated = _small_jpeg()

        capped_clean, capped_annotated = cap_pair(clean, annotated, max_bytes=100_000)

        assert len(capped_clean) <= 100_000
        assert capped_annotated == annotated
