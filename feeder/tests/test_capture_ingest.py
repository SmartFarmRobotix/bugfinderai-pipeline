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
The capture-folder ingest adapter (``app/ingest.py``): a capture date
directory of epoch-named folders becomes a conforming session folder,
validated through this component's own loader (``load_session_manifest`` /
``load_replay_session``), never a parallel reimplementation.

The EXIF fixtures are synthetic: an ``ImageDescription`` of ``key: value``
lines, a UNICODE-prefixed ``UserComment`` payload, and a GPS IFD that holds
only ``GPSVersionID``, with no coordinate tags.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import piexif
import pytest
from PIL import Image

from app.ingest import (
    CAPTURE_EXIF_FILENAME,
    CAPTURE_EXIF_SCHEMA,
    EmptyCaptureTreeError,
    MissingSourceImageError,
    NonContainableSessionIdError,
    NonEpochDirnameError,
    ingest_capture_date_dir,
)
from app.manifest import REQUIRED_TOP_LEVEL_KEYS, load_session_manifest
from app.store import load_replay_session, sessions_root

NON_STANDARD_DATETIME = b"2023-11-14_23-13-20"
SAMPLE_DESCRIPTION = b"Exposure: 1/250\nGain: 2\nWhiteBalance: 5000K"
VERSION_ONLY_GPS_IFD = {piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0)}

KNOWN_PER_IMAGE_KEYS = {"filename", "timestamp", "position"}

DATE_DIR_NAME = "2023-11-14"
EPOCH_A = "1700000000.07"
EPOCH_B = "1700000050.70"
EPOCH_WHOLE = "1700000000"


def _deg_to_dms_rational(value: float):
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60 * 100)
    return ((degrees, 1), (minutes, 1), (seconds, 100))


def _gps_ifd_with_coordinates(latitude: float, longitude: float) -> dict:
    return {
        piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N" if latitude >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(abs(latitude)),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if longitude >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(abs(longitude)),
    }


def _json_user_comment() -> bytes:
    payload = json.dumps({"calibration": {"batch": "example"}}).encode("utf-16-be")
    return b"UNICODE\x00" + payload


def _write_rgb_jpg(
    path: Path,
    *,
    description: Optional[bytes] = None,
    user_comment: Optional[bytes] = None,
    gps_ifd: Optional[dict] = None,
    date_time_original: Optional[bytes] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path, format="JPEG")

    zeroth: dict = {}
    exif_ifd: dict = {}
    if description is not None:
        zeroth[piexif.ImageIFD.ImageDescription] = description
    if date_time_original is not None:
        zeroth[piexif.ImageIFD.DateTime] = date_time_original
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = date_time_original
    if user_comment is not None:
        exif_ifd[piexif.ExifIFD.UserComment] = user_comment

    if not (description is None and user_comment is None and date_time_original is None and gps_ifd is None):
        exif_dict = {"0th": zeroth, "Exif": exif_ifd, "GPS": gps_ifd or {}, "1st": {}}
        piexif.insert(piexif.dump(exif_dict), str(path))


def _make_capture_folder(date_dir: Path, epoch_name: str, *, with_byproducts: bool = False, **rgb_kwargs) -> Path:
    folder = date_dir / epoch_name
    _write_rgb_jpg(folder / "rgb.jpg", **rgb_kwargs)
    if with_byproducts:
        for name in (
            "other-L.png",
            "other-R.png",
            "extra_rgb.jpg",
            "left.png",
            "right.png",
            "aligned.png",
            "center.jpg",
            "depth.png",
        ):
            (folder / name).write_bytes(b"not-the-session-image")
    return folder


def test_a_capture_tree_ingests_and_loads(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A, with_byproducts=True)
    _make_capture_folder(date_dir, EPOCH_B, with_byproducts=True)
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-tree", crop="brassica-oleracea", data_root=data_root
    )

    manifest = load_session_manifest(session_dir)
    assert {img.filename for img in manifest.images} == {f"{EPOCH_A}.jpg", f"{EPOCH_B}.jpg"}

    replay = load_replay_session(data_root, "sess-tree")
    assert len(replay.images) == 2

    source_shas = json.loads((session_dir / "source-shas.json").read_text())
    assert set(source_shas) == {f"{EPOCH_A}.jpg", f"{EPOCH_B}.jpg"}

    for filename, expected_sha in source_shas.items():
        image_bytes = (session_dir / "images" / filename).read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == expected_sha
        source_bytes = (date_dir / filename.removesuffix(".jpg") / "rgb.jpg").read_bytes()
        assert image_bytes == source_bytes


def test_only_the_rgb_jpg_enters_the_session(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A, with_byproducts=True)
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-byproducts", crop="brassica-oleracea", data_root=data_root
    )

    manifest = load_session_manifest(session_dir)
    assert [img.filename for img in manifest.images] == [f"{EPOCH_A}.jpg"]

    by_product_names = {"other-L.png", "extra_rgb.jpg", "left.png", "right.png", "aligned.png", "center.jpg", "depth.png"}
    manifest_text = (session_dir / "manifest.json").read_text()
    source_shas_text = (session_dir / "source-shas.json").read_text()
    images_on_disk = {p.name for p in (session_dir / "images").iterdir()}
    for name in by_product_names:
        assert name not in manifest_text
        assert name not in source_shas_text
        assert name not in images_on_disk


def test_folder_epochs_are_the_only_timestamps(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_WHOLE, date_time_original=NON_STANDARD_DATETIME)
    _make_capture_folder(date_dir, EPOCH_B, date_time_original=NON_STANDARD_DATETIME)
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-epochs", crop="brassica-oleracea", data_root=data_root
    )

    raw = json.loads((session_dir / "manifest.json").read_text())
    by_filename = {img["filename"]: img for img in raw["images"]}

    assert by_filename[f"{EPOCH_WHOLE}.jpg"]["timestamp"] == "2023-11-14T22:13:20+00:00"
    assert by_filename[f"{EPOCH_B}.jpg"]["timestamp"] == "2023-11-14T22:14:10.700000+00:00"
    assert raw["session_start"] == "2023-11-14T22:13:20+00:00"
    assert raw["session_end"] == "2023-11-14T22:14:10.700000+00:00"


def test_present_gps_becomes_a_gps_provenance_position(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A, gps_ifd=_gps_ifd_with_coordinates(10.501, 20.501))
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-gps", crop="brassica-oleracea", data_root=data_root
    )

    manifest = load_session_manifest(session_dir)
    position = manifest.images[0].position
    assert position is not None
    assert position.provenance == "gps"
    assert position.latitude == pytest.approx(10.501, abs=1e-3)
    assert position.longitude == pytest.approx(20.501, abs=1e-3)
    assert load_replay_session(data_root, "sess-gps").images[0].position.provenance == "gps"


def test_version_only_gps_ifd_yields_no_position(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A, gps_ifd=VERSION_ONLY_GPS_IFD)
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-no-gps", crop="brassica-oleracea", data_root=data_root
    )

    manifest = load_session_manifest(session_dir)
    assert manifest.images[0].position is None
    assert manifest.latitude is None
    assert manifest.longitude is None

    raw = json.loads((session_dir / "manifest.json").read_text())
    assert "latitude" not in raw
    assert "longitude" not in raw
    assert "position" not in raw["images"][0]


def test_no_gps_ifd_at_all_yields_no_position(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A)
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-absent-gps", crop="brassica-oleracea", data_root=data_root
    )

    manifest = load_session_manifest(session_dir)
    assert manifest.images[0].position is None


def test_capture_annotations_ride_to_the_sidecar_and_manifest_stays_frozen(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(
        date_dir,
        EPOCH_A,
        description=SAMPLE_DESCRIPTION,
        user_comment=_json_user_comment(),
        gps_ifd=_gps_ifd_with_coordinates(10.501, 20.501),
    )
    _make_capture_folder(
        date_dir,
        EPOCH_B,
        gps_ifd=VERSION_ONLY_GPS_IFD,
    )
    data_root = tmp_path / "data_root"

    session_dir = ingest_capture_date_dir(
        date_dir, session_id="sess-sidecar", crop="brassica-oleracea", data_root=data_root
    )

    sidecar = json.loads((session_dir / CAPTURE_EXIF_FILENAME).read_text())
    assert sidecar["schema"] == CAPTURE_EXIF_SCHEMA
    with_annotation = sidecar["images"][f"{EPOCH_A}.jpg"]
    assert with_annotation["description"] == {"Exposure": "1/250", "Gain": "2", "WhiteBalance": "5000K"}
    assert json.loads(with_annotation["user_comment"]) == {"calibration": {"batch": "example"}}

    without_annotation = sidecar["images"][f"{EPOCH_B}.jpg"]
    assert without_annotation["description"] == {}
    assert without_annotation["user_comment"] is None

    raw = json.loads((session_dir / "manifest.json").read_text())
    assert set(raw.keys()) == set(REQUIRED_TOP_LEVEL_KEYS)
    by_filename = {img["filename"]: img for img in raw["images"]}
    assert set(by_filename[f"{EPOCH_A}.jpg"].keys()) == KNOWN_PER_IMAGE_KEYS
    assert set(by_filename[f"{EPOCH_B}.jpg"].keys()) == {"filename", "timestamp"}

    load_session_manifest(session_dir)


def test_a_folder_without_rgb_jpg_refuses(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A)
    (date_dir / EPOCH_B).mkdir(parents=True)
    (date_dir / EPOCH_B / "depth.png").write_bytes(b"not-rgb")
    data_root = tmp_path / "data_root"

    with pytest.raises(MissingSourceImageError) as exc_info:
        ingest_capture_date_dir(date_dir, session_id="sess-missing", crop="brassica-oleracea", data_root=data_root)

    assert EPOCH_B in str(exc_info.value)
    assert not (sessions_root(data_root) / "sess-missing").exists()


@pytest.mark.parametrize("bad_name", ["notes", "1e3", "+1700000000", "nan"])
def test_a_non_epoch_dirname_refuses(tmp_path, bad_name):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A)
    (date_dir / bad_name).mkdir(parents=True)
    (date_dir / bad_name / "rgb.jpg").write_bytes(b"irrelevant")
    data_root = tmp_path / "data_root"

    with pytest.raises(NonEpochDirnameError) as exc_info:
        ingest_capture_date_dir(date_dir, session_id="sess-badname", crop="brassica-oleracea", data_root=data_root)

    assert bad_name in str(exc_info.value)
    assert not (sessions_root(data_root) / "sess-badname").exists()


def test_a_non_containable_session_id_refuses_before_any_write(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A)
    data_root = tmp_path / "data_root"

    with pytest.raises(NonContainableSessionIdError) as exc_info:
        ingest_capture_date_dir(
            date_dir, session_id="../../outside", crop="brassica-oleracea", data_root=data_root
        )

    assert "../../outside" in str(exc_info.value)
    assert not (data_root / "sessions" / "outside").exists()
    assert not (data_root.parent.parent / "outside").exists()
    assert not (data_root / "outside").exists()


def test_an_empty_tree_refuses(tmp_path):
    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    date_dir.mkdir(parents=True)
    data_root = tmp_path / "data_root"

    with pytest.raises(EmptyCaptureTreeError):
        ingest_capture_date_dir(date_dir, session_id="sess-empty", crop="brassica-oleracea", data_root=data_root)

    assert not (sessions_root(data_root) / "sess-empty").exists()


def test_cli_entry_ingests_a_valid_tree(tmp_path, capsys):
    from app.ingest import main

    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    _make_capture_folder(date_dir, EPOCH_A)
    data_root = tmp_path / "data_root"

    exit_code = main(
        [
            str(date_dir),
            "--session-id",
            "sess-cli",
            "--crop",
            "brassica-oleracea",
            "--data-root",
            str(data_root),
        ]
    )

    assert exit_code == 0
    assert "Session written" in capsys.readouterr().out
    load_session_manifest(sessions_root(data_root) / "sess-cli")


def test_cli_entry_reports_refusal_and_exits_nonzero(tmp_path, capsys):
    from app.ingest import main

    date_dir = tmp_path / "photos" / DATE_DIR_NAME
    date_dir.mkdir(parents=True)
    data_root = tmp_path / "data_root"

    exit_code = main(
        [
            str(date_dir),
            "--session-id",
            "sess-cli-empty",
            "--crop",
            "brassica-oleracea",
            "--data-root",
            str(data_root),
        ]
    )

    assert exit_code == 1
    assert "[REFUSED]" in capsys.readouterr().err
