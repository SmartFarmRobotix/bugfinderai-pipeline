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

"""Caps a single best-image
variant to a byte bound, the same way the Mixed scenario's push already
caps its pair before it ever leaves the device
(`inference/app/image_compress.py`, `compress_to_cap`).
A documented copy, not an import -- the two services share no package (the same constraint `edge_report.py`'s own module
docstring names for the mirrored PDF core). Pillow is already a
dependency here transitively through fpdf2 (`requirements.txt`).
"""

from __future__ import annotations

import io
from typing import Tuple

from PIL import Image

_MIN_QUALITY = 20
_QUALITY_STEP = 15
_DOWNSCALE_FACTOR = 0.75
_SMALL_DOWNSCALE_FACTOR = 0.5
_SMALL_SIDE_THRESHOLD = 256


class ImageCapError(RuntimeError):
    """Raised when the image cannot be capped to `max_bytes` even at a
    one-pixel side -- the smallest achievable encoding is still over the
    bound, so there is nothing left to return that honors the cap."""

    def __init__(self, encoded_size: int, max_bytes: int) -> None:
        super().__init__(
            f"cannot cap image to {max_bytes} bytes: smallest encoding "
            f"(one-pixel side) is still {encoded_size} bytes"
        )
        self.encoded_size = encoded_size
        self.max_bytes = max_bytes


def cap_image_bytes(image_bytes: bytes, max_bytes: int = 100_000) -> bytes:
    """Returns `image_bytes` unchanged if already `<= max_bytes`; otherwise
    re-encodes as JPEG (RGB), walking quality from 85 down to 20 in steps
    of 15, then downscaling the longer side by 0.75 (0.5 once a side is
    under 256 px) and restarting from quality 60, repeating until the
    result is `<= max_bytes`. Always returns `<= max_bytes` -- the only
    floor is a one-pixel side; if the encoding at a one-pixel side is
    still over the bound, raises `ImageCapError` rather than ever
    returning an oversized image."""
    if len(image_bytes) <= max_bytes:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
        if quality > _MIN_QUALITY:
            quality -= _QUALITY_STEP
            continue
        if min(img.size) <= 1:
            raise ImageCapError(len(data), max_bytes)
        factor = _SMALL_DOWNSCALE_FACTOR if min(img.size) < _SMALL_SIDE_THRESHOLD else _DOWNSCALE_FACTOR
        new_size = (
            max(int(img.width * factor), 1),
            max(int(img.height * factor), 1),
        )
        img = img.resize(new_size, Image.LANCZOS)
        quality = 60


def cap_pair(clean: bytes, annotated: bytes, max_bytes: int = 100_000) -> Tuple[bytes, bytes]:
    """Caps both variants of the best-image pair to `max_bytes` each."""
    return cap_image_bytes(clean, max_bytes), cap_image_bytes(annotated, max_bytes)
