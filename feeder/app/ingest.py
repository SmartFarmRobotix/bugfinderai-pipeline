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
Session feeder — capture-folder ingest adapter.

Reads one capture date directory, a tree of epoch-named folders that each
hold the capture ``rgb.jpg`` (and possibly other files, which are ignored),
and produces a session folder in the format this component reads
(``app/manifest.py``): ``manifest.json`` (schema 1), a ``source-shas.json``
integrity record, and an ``images/`` directory holding each folder's
``rgb.jpg`` under ``<epoch>.jpg``.

The folder epoch is the only timestamp source: EXIF ``DateTimeOriginal`` is
not consulted, since cameras write it in varying forms. GPS coordinates are
read from the EXIF GPS IFD through ``app/positions.py`` and never fabricated
when absent. The free-text EXIF annotations (the ``key: value`` lines of
``ImageDescription`` and the ``UserComment`` payload) land in a session-root
sidecar, ``capture-exif.json``, never in the manifest.

All validation happens before any write: a malformed tree (a folder without
``rgb.jpg``, a folder name that is not an epoch, a session id that is not a
bare name, an empty tree) is refused by name and leaves no partial session
folder behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import piexif

from app.manifest import load_session_manifest, resolve_contained
from app.positions import PlatformCapturePositionReader, PlatformPosition
from app.store import sessions_root

SOURCE_IMAGE_NAME = "rgb.jpg"
CAPTURE_EXIF_SCHEMA = "capture-exif-v1"
CAPTURE_EXIF_FILENAME = "capture-exif.json"

_STRICT_EPOCH_RE = re.compile(r"^[0-9]{1,12}(\.[0-9]{1,6})?$")
_DESCRIPTION_LINE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")


class IngestError(ValueError):
    """Base for every ingest refusal: always named, always before any write."""


class MissingSourceImageError(IngestError):
    def __init__(self, folder_name: str) -> None:
        self.folder_name = folder_name
        super().__init__(f"capture folder {folder_name!r} has no {SOURCE_IMAGE_NAME}")


class NonEpochDirnameError(IngestError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"capture subfolder name {name!r} is not a strict epoch "
            "(digits with an optional dot-digits fraction)"
        )


class NonContainableSessionIdError(IngestError):
    def __init__(self, session_id: str, detail: str) -> None:
        self.session_id = session_id
        super().__init__(f"session id {session_id!r} is not a bare, containable name: {detail}")


class EmptyCaptureTreeError(IngestError):
    def __init__(self, date_dir: Path) -> None:
        self.date_dir = date_dir
        super().__init__(f"capture date-directory {date_dir} contains no capture folders")


@dataclass(frozen=True)
class CaptureAnnotation:
    """The free-text EXIF annotations of one capture: the ``key: value`` lines
    of ``ImageDescription`` and the decoded ``UserComment`` payload."""

    description: dict[str, str]
    user_comment: Optional[str]


@dataclass(frozen=True)
class _PreparedImage:
    filename: str
    data: bytes
    sha256: str
    timestamp_iso: str
    epoch_sort_key: tuple[int, int]
    position: Optional[PlatformPosition]
    annotation: CaptureAnnotation


def _validate_session_id(session_id: str, sessions_root_dir: Path) -> None:
    """
    A bare, containable name: no path separators, no '.'/'..' segment, and
    its resolved sessions-root-relative path stays inside the sessions root
    (``resolve_contained`` enforces the containment, like the per-image
    filename check in ``app/manifest.py``).
    """
    if not session_id or not isinstance(session_id, str):
        raise NonContainableSessionIdError(session_id, "must be a non-empty string")
    p = Path(session_id)
    if p.is_absolute() or len(p.parts) != 1 or session_id in (".", ".."):
        raise NonContainableSessionIdError(
            session_id, "must be a bare name with no path separators or '.'/'..' segments"
        )
    candidate = sessions_root_dir / session_id
    try:
        resolve_contained(sessions_root_dir, candidate)
    except ValueError as exc:
        raise NonContainableSessionIdError(session_id, str(exc)) from exc


def _epoch_sort_key(epoch_name: str) -> tuple[int, int]:
    """Exact ordering key (integer seconds, integer microseconds), never a float."""
    int_part, _, frac_part = epoch_name.partition(".")
    micros = int((frac_part + "000000")[:6]) if frac_part else 0
    return (int(int_part), micros)


def _epoch_to_iso(epoch_name: str) -> str:
    """
    Render the folder epoch (UNIX time, integer or fractional) as UTC
    ISO-8601, built from integer seconds and integer microseconds, so the
    string is exact for every strict-epoch name.
    """
    int_part, _, frac_part = epoch_name.partition(".")
    dt = datetime.fromtimestamp(int(int_part), tz=timezone.utc)
    if frac_part:
        micros = int((frac_part + "000000")[:6])
        dt = dt.replace(microsecond=micros)
    return dt.isoformat()


def _scan_capture_folders(date_dir: Path) -> list[Path]:
    subdirs = sorted((p for p in date_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not subdirs:
        raise EmptyCaptureTreeError(date_dir)
    for p in subdirs:
        if not _STRICT_EPOCH_RE.fullmatch(p.name):
            raise NonEpochDirnameError(p.name)
    for p in subdirs:
        if not (p / SOURCE_IMAGE_NAME).is_file():
            raise MissingSourceImageError(p.name)
    return subdirs


def _decode_user_comment(raw: Optional[bytes]) -> str:
    """
    EXIF 2.3 UserComment: an 8-byte character-code prefix followed by the
    payload (``UNICODE\\x00`` for UTF-16BE text, ``ASCII\\x00\\x00\\x00`` for
    ASCII text, undefined for a best-effort decode).
    """
    if not raw:
        return ""
    prefix, payload = raw[:8], raw[8:]
    if prefix.startswith(b"UNICODE"):
        return payload.decode("utf-16-be", errors="replace")
    if prefix.startswith(b"ASCII"):
        return payload.decode("ascii", errors="replace")
    return raw.decode("ascii", errors="replace")


def _parse_description(description: str) -> dict[str, str]:
    """Every ``key: value`` line of the description, in order; other lines are ignored."""
    fields: dict[str, str] = {}
    for line in description.splitlines():
        m = _DESCRIPTION_LINE_RE.match(line)
        if m and m.group(1):
            fields[m.group(1)] = m.group(2)
    return fields


def _parse_capture_annotations(exif_dict: dict) -> CaptureAnnotation:
    zeroth = exif_dict.get("0th", {})
    raw_desc = zeroth.get(piexif.ImageIFD.ImageDescription)
    description = raw_desc.decode("ascii", errors="replace") if isinstance(raw_desc, bytes) else ""

    exif_ifd = exif_dict.get("Exif", {})
    comment = _decode_user_comment(exif_ifd.get(piexif.ExifIFD.UserComment)).strip("\x00").strip()

    return CaptureAnnotation(
        description=_parse_description(description),
        user_comment=comment or None,
    )


def _dms_to_decimal(dms, ref: Optional[bytes]) -> Optional[float]:
    if not dms or not ref:
        return None
    try:
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
    except (IndexError, TypeError, ZeroDivisionError):
        return None
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in (b"S", b"W"):
        value = -value
    return value


class _ExifGpsPositionSource:
    """
    ``app/positions.py`` source implementation over one image's loaded EXIF
    GPS IFD. An IFD without coordinate tags (for example a version-only
    IFD) has no GPSLatitude/GPSLongitude keys, so ``read_position`` returns
    ``None``, never a fabricated coordinate.
    """

    def __init__(self, gps_ifd: dict) -> None:
        self._gps_ifd = gps_ifd

    def read_position(self) -> Optional[PlatformPosition]:
        latitude = _dms_to_decimal(
            self._gps_ifd.get(piexif.GPSIFD.GPSLatitude), self._gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
        )
        longitude = _dms_to_decimal(
            self._gps_ifd.get(piexif.GPSIFD.GPSLongitude), self._gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
        )
        if latitude is None or longitude is None:
            return None
        return PlatformPosition(latitude=latitude, longitude=longitude, provenance="gps")


def _read_gps_position(exif_dict: dict) -> Optional[PlatformPosition]:
    """Per image, through ``PlatformCapturePositionReader``; no session-level default, so absence stays absence."""
    reader = PlatformCapturePositionReader(
        _ExifGpsPositionSource(exif_dict.get("GPS", {})),
        default_latitude=None,
        default_longitude=None,
    )
    return reader.read_session_position()


def _prepare_image(folder: Path) -> _PreparedImage:
    epoch_name = folder.name
    src = folder / SOURCE_IMAGE_NAME
    data = src.read_bytes()
    try:
        exif_dict = piexif.load(data)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}

    return _PreparedImage(
        filename=f"{epoch_name}.jpg",
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        timestamp_iso=_epoch_to_iso(epoch_name),
        epoch_sort_key=_epoch_sort_key(epoch_name),
        position=_read_gps_position(exif_dict),
        annotation=_parse_capture_annotations(exif_dict),
    )


def _write_session(session_dir: Path, session_id: str, crop: str, images: list[_PreparedImage]) -> None:
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True)

    for img in images:
        (images_dir / img.filename).write_bytes(img.data)

    entries = []
    for img in images:
        entry: dict = {"filename": img.filename, "timestamp": img.timestamp_iso}
        if img.position is not None:
            entry["position"] = {
                "latitude": img.position.latitude,
                "longitude": img.position.longitude,
                "provenance": img.position.provenance,
            }
        entries.append(entry)

    ordered = sorted(images, key=lambda i: i.epoch_sort_key)
    session_start = ordered[0].timestamp_iso
    session_end = ordered[-1].timestamp_iso

    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "crop": crop,
        "session_start": session_start,
        "session_end": session_end,
        "images": entries,
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    source_shas = {img.filename: img.sha256 for img in images}
    (session_dir / "source-shas.json").write_text(json.dumps(source_shas, indent=2) + "\n", encoding="utf-8")

    sidecar = {
        "schema": CAPTURE_EXIF_SCHEMA,
        "images": {
            img.filename: {
                "description": img.annotation.description,
                "user_comment": img.annotation.user_comment,
            }
            for img in images
        },
    }
    (session_dir / CAPTURE_EXIF_FILENAME).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def ingest_capture_date_dir(date_dir: Path, *, session_id: str, crop: str, data_root: Path) -> Path:
    """
    Ingest one capture date directory into ``<data_root>/sessions/<session_id>/``.

    Every check runs before any write: the session id, then the whole tree
    (folder names, then the presence of ``rgb.jpg``). An ``IngestError``
    subclass is raised, quoting the offending name, and no session directory
    (partial or otherwise) is left behind.
    """
    sessions_root_dir = sessions_root(data_root)
    _validate_session_id(session_id, sessions_root_dir)
    folders = _scan_capture_folders(date_dir)

    images = [_prepare_image(folder) for folder in folders]

    session_dir = sessions_root_dir / session_id
    _write_session(session_dir, session_id, crop, images)

    load_session_manifest(session_dir)

    return session_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date_dir", type=Path, help="Capture date directory of epoch-named folders")
    parser.add_argument("--session-id", type=str, required=True, help="Session id for the ingested folder")
    parser.add_argument("--crop", type=str, required=True, help="Crop type to record in the manifest")
    parser.add_argument("--data-root", type=Path, required=True, help="Destination data root (sessions/<id>/ layout)")
    args = parser.parse_args(argv)

    try:
        session_dir = ingest_capture_date_dir(
            args.date_dir.resolve(),
            session_id=args.session_id,
            crop=args.crop,
            data_root=args.data_root.resolve(),
        )
    except IngestError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        return 1

    print(f"Session written: {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
