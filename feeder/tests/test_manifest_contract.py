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
Session-folder manifest contract. Every fixture here is hand-written (not
produced by this component's own code).
"""

from __future__ import annotations

import json

import pytest

from app.manifest import (
    DuplicateImageEntryError,
    InvalidFieldError,
    MissingReferencedImageError,
    MissingRequiredKeyError,
    PathContainmentError,
    SessionIdMismatchError,
    load_session_manifest,
)


def _session_dir(fixtures_dir, fixture_name, session_id):
    return fixtures_dir / fixture_name / "sessions" / session_id


def test_minimal_conforming_manifest_is_accepted(fixtures_dir):
    m = load_session_manifest(_session_dir(fixtures_dir, "minimal_conforming", "sess-min"))
    assert m.session_id == "sess-min"
    assert m.crop == "brassica-oleracea"
    assert len(m.images) == 1
    assert m.images[0].position is None
    assert m.growth_stage is None


def test_optional_growth_stage_and_positions_pass_through(fixtures_dir):
    m = load_session_manifest(_session_dir(fixtures_dir, "optional_fields", "sess-opt"))
    assert m.growth_stage == {"representation": "enum", "stage_id": "BBCH-19"}
    assert m.images[0].position is not None
    assert m.images[0].position.latitude == 10.5
    assert m.images[0].position.longitude == 20.5
    assert m.images[1].position is None


def test_optional_session_coordinates_are_read_from_prior_session_metadata(fixtures_dir, tmp_path):
    source = _session_dir(fixtures_dir, "minimal_conforming", "sess-min")
    session_dir = tmp_path / "sess-min"
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "images" / "a.jpg").write_bytes((source / "images" / "a.jpg").read_bytes())
    raw = json.loads((source / "manifest.json").read_text())
    raw.update({"latitude": 10.5, "longitude": 20.5})
    (session_dir / "manifest.json").write_text(json.dumps(raw))

    manifest = load_session_manifest(session_dir)

    assert (manifest.latitude, manifest.longitude) == (10.5, 20.5)


def test_session_id_mismatch_is_rejected(fixtures_dir):
    with pytest.raises(SessionIdMismatchError):
        load_session_manifest(_session_dir(fixtures_dir, "bad_session_id_mismatch", "sess-real"))


def test_missing_required_key_is_rejected(fixtures_dir):
    with pytest.raises(MissingRequiredKeyError) as exc:
        load_session_manifest(_session_dir(fixtures_dir, "bad_missing_key", "sess-missing"))
    assert "crop" in exc.value.missing_keys


def test_submission_order_matches_manifest_order_exactly(fixtures_dir):
    m = load_session_manifest(_session_dir(fixtures_dir, "handwritten_conforming", "sess-001"))
    assert [img.filename for img in m.images] == ["b.png", "a.jpg"]


def test_duplicate_entry_rejects_whole_folder(fixtures_dir):
    with pytest.raises(DuplicateImageEntryError) as exc:
        load_session_manifest(_session_dir(fixtures_dir, "bad_duplicate", "sess-dup"))
    assert exc.value.filename == "a.jpg"


def test_missing_referenced_file_rejects_whole_folder(fixtures_dir):
    with pytest.raises(MissingReferencedImageError) as exc:
        load_session_manifest(_session_dir(fixtures_dir, "bad_missing_file", "sess-nofile"))
    assert exc.value.filename == "b.jpg"


def test_extra_unreferenced_file_is_skipped_not_rejected(fixtures_dir):
    m = load_session_manifest(_session_dir(fixtures_dir, "extra_file", "sess-extra"))
    assert [img.filename for img in m.images] == ["a.jpg"]
    assert m.skipped_extra_files == ["unlisted.jpg"]


def test_traversal_filename_is_rejected_before_any_file_is_read(fixtures_dir):
    with pytest.raises(PathContainmentError):
        load_session_manifest(_session_dir(fixtures_dir, "bad_traversal", "sess-trav"))


def test_symlinked_manifest_entry_escaping_images_dir_is_rejected(fixtures_dir):
    with pytest.raises(PathContainmentError):
        load_session_manifest(_session_dir(fixtures_dir, "bad_symlink_escape", "sess-symlink"))


def test_relative_filename_with_subdirectory_is_rejected(fixtures_dir, tmp_path):
    session_dir = tmp_path / "sessions" / "sess-subdir"
    images_dir = session_dir / "images" / "sub"
    images_dir.mkdir(parents=True)
    (images_dir / "x.jpg").write_bytes(b"data")
    (session_dir / "manifest.json").write_text(
        '{"schema_version":1,"session_id":"sess-subdir","crop":"c",'
        '"session_start":"t","session_end":"t","images":[{"filename":"sub/x.jpg"}]}'
    )
    with pytest.raises(PathContainmentError):
        load_session_manifest(session_dir)


def _minimal_folder_with(tmp_path, fixtures_dir, **manifest_overrides):
    import shutil

    session_dir = tmp_path / "sessions" / "sess-min"
    shutil.copytree(_session_dir(fixtures_dir, "minimal_conforming", "sess-min"), session_dir)
    raw = json.loads((session_dir / "manifest.json").read_text())
    raw.update(manifest_overrides)
    (session_dir / "manifest.json").write_text(json.dumps(raw))
    return session_dir


def test_an_optional_pest_passes_through(fixtures_dir, tmp_path):
    m = load_session_manifest(_minimal_folder_with(tmp_path, fixtures_dir, pest="BRVCBR"))
    assert m.pest == "BRVCBR"


def test_a_manifest_without_a_pest_has_none(fixtures_dir):
    assert load_session_manifest(_session_dir(fixtures_dir, "minimal_conforming", "sess-min")).pest is None


@pytest.mark.parametrize("bad", ["", "   ", 12, True, ["BRVCBR"]])
def test_an_invalid_pest_rejects_the_folder_naming_pest(fixtures_dir, tmp_path, bad):
    with pytest.raises(InvalidFieldError) as exc:
        load_session_manifest(_minimal_folder_with(tmp_path, fixtures_dir, pest=bad))
    assert exc.value.reason == "invalid_field"
    assert exc.value.field_name == "pest"
    assert "'pest'" in str(exc.value)
