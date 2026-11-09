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

"""JPEG/PNG-only enforcement at store-scan time."""

from __future__ import annotations

from app.store import is_supported_image_format, load_replay_session


def test_is_supported_image_format():
    assert is_supported_image_format("a.jpg")
    assert is_supported_image_format("a.JPEG")
    assert is_supported_image_format("a.png")
    assert not is_supported_image_format("a.tiff")
    assert not is_supported_image_format("a.gif")


def test_mixed_format_session_skips_unsupported_file_and_submits_only_jpeg_png(fixtures_dir):
    session = load_replay_session(fixtures_dir / "mixed_format", "sess-mixed")
    assert [img.filename for img in session.images] == ["a.jpg", "b.png"]
    assert session.skipped_unsupported == ["c.tiff"]
