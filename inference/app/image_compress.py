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

"""Best-image pair compression to the configured size cap, smaller than the raw capture format, so
the pair stays small on a slow uplink.

Pure functions over in-memory bytes, no I/O -- independently testable
without a running service.
"""

from __future__ import annotations

import io
from typing import Tuple

from PIL import Image

_MIN_QUALITY = 20
_MIN_DIMENSION = 64
_QUALITY_STEP = 15
_DOWNSCALE_FACTOR = 0.75


def compress_to_cap(image_bytes: bytes, max_bytes: int) -> Tuple[bytes, str]:
    """
    Re-encode image_bytes as JPEG at <= max_bytes. Reduces quality first;
    once quality bottoms out, downscales and retries from a mid quality
    rather than grinding quality to zero. Always returns JPEG so the cap is
    met deterministically -- annotated overlays survive JPEG re-encoding
    (they are baked into the pixel data, not a separate layer).
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/jpeg"
        if quality > _MIN_QUALITY:
            quality -= _QUALITY_STEP
            continue
        if min(img.size) <= _MIN_DIMENSION:
            return data, "image/jpeg"
        new_size = (
            max(int(img.width * _DOWNSCALE_FACTOR), 1),
            max(int(img.height * _DOWNSCALE_FACTOR), 1),
        )
        img = img.resize(new_size, Image.LANCZOS)
        quality = 60


def compress_pair_to_cap(
    clean_bytes: bytes,
    annotated_bytes: bytes,
    max_bytes_per_image: int,
    max_bytes_combined: int,
) -> Tuple[bytes, str, bytes, str]:
    """Returns (clean_bytes, clean_content_type, annotated_bytes, annotated_content_type)."""
    clean_out, clean_ct = compress_to_cap(clean_bytes, max_bytes_per_image)
    annotated_out, annotated_ct = compress_to_cap(annotated_bytes, max_bytes_per_image)

    if len(clean_out) + len(annotated_out) > max_bytes_combined:
        per_image_share = max_bytes_combined // 2
        clean_out, clean_ct = compress_to_cap(clean_bytes, per_image_share)
        annotated_out, annotated_ct = compress_to_cap(annotated_bytes, per_image_share)

    return clean_out, clean_ct, annotated_out, annotated_ct
