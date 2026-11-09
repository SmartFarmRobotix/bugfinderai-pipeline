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
Session feeder — session-folder manifest contract.

This module is the frozen, versioned schema for the session-folder producer
interface: a ``<data_root>/sessions/<session_id>/manifest.json`` describing that session's
``images/`` subdirectory. Any conforming producer — replay (this component's
own) or a future one — can write this shape; this module only reads and
validates it, never writes it.

Rejection is deterministic and split:
  fail-closed (whole manifest rejected) for structural defects: missing
    required key, session_id mismatch, duplicate image entry, missing
    referenced file, path-containment violation;
  fail-open (one file skipped, the rest processed) for surplus files not
    referenced by any manifest entry.

Format is deliberately NOT enforced here — a
manifest-referenced file that exists but is the wrong format is a concern of
``app/store.py``, not a structural manifest defect (mirrors
inference/app/storage.py: ``list_manifest_images`` returns the
manifest-referenced files in manifest order, the caller applies
``is_supported_image_format``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "session_id",
    "crop",
    "session_start",
    "session_end",
    "images",
)


class ManifestError(ValueError):
    """Base for every session-folder rejection (fail-closed: reject the whole folder)."""

    reason: str = "manifest_error"


class MissingRequiredKeyError(ManifestError):
    reason = "missing_required_key"

    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        super().__init__(f"manifest is missing required key(s): {', '.join(missing_keys)}")


class InvalidFieldError(ManifestError):
    reason = "invalid_field"

    def __init__(self, field_name: str, detail: str) -> None:
        self.field_name = field_name
        super().__init__(f"manifest field {field_name!r} is invalid: {detail}")


class SessionIdMismatchError(ManifestError):
    reason = "session_id_mismatch"

    def __init__(self, manifest_session_id: str, dir_name: str) -> None:
        self.manifest_session_id = manifest_session_id
        self.dir_name = dir_name
        super().__init__(
            f"manifest session_id {manifest_session_id!r} does not match "
            f"enclosing directory name {dir_name!r}"
        )


class DuplicateImageEntryError(ManifestError):
    reason = "duplicate_image_entry"

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"duplicate image entry in manifest: {filename!r}")


class MissingReferencedImageError(ManifestError):
    reason = "missing_referenced_image"

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"manifest references image {filename!r} but no such file exists under images/")


class PathContainmentError(ManifestError):
    reason = "path_containment_violation"

    def __init__(self, filename: str, detail: str) -> None:
        self.filename = filename
        super().__init__(f"image entry {filename!r} violates path containment: {detail}")


@dataclass(frozen=True)
class ImagePosition:
    """A per-image position as recorded by the manifest."""

    latitude: float
    longitude: float
    provenance: str


@dataclass(frozen=True)
class ManifestImageEntry:
    filename: str
    path: Path
    position: Optional[ImagePosition]
    timestamp: Optional[str]


@dataclass(frozen=True)
class SessionManifest:
    schema_version: int
    session_id: str
    crop: str
    session_start: str
    session_end: str
    images: list[ManifestImageEntry]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    growth_stage: Optional[dict[str, Any]] = None
    images_per_plant: Optional[int] = None
    pest: Optional[str] = None
    skipped_extra_files: list[str] = field(default_factory=list)


def _validate_filename_shape(filename: str) -> None:
    """
    Reject before any file is read: a filename with a path
    separator or a '.'/'..' segment is a containment violation, not a
    normalizable path.
    """
    if not filename or not isinstance(filename, str):
        raise InvalidFieldError("images[].filename", "must be a non-empty string")
    p = Path(filename)
    if p.is_absolute() or len(p.parts) != 1 or filename in (".", ".."):
        raise PathContainmentError(filename, "filename must be a bare name with no path separators or '.'/'..' segments")
    if p.name in (".", ".."):
        raise PathContainmentError(filename, "filename must not be '.' or '..'")


def resolve_contained(root: Path, candidate: Path) -> Path:
    """
    Resolve *candidate* (realpath-style: symlinks and traversal collapsed)
    and require it to lie strictly within *root* (also resolved). Mirrors
    the inference service's own ``storage.resolve_contained_path`` (the
    inference-service-side half of the same containment check).
    """
    root_resolved = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{candidate} resolves outside {root}") from exc
    return resolved


def _parse_position(filename: str, raw: Optional[dict[str, Any]]) -> Optional[ImagePosition]:
    if raw is None:
        return None
    if not isinstance(raw, dict) or "latitude" not in raw or "longitude" not in raw:
        raise InvalidFieldError(f"images[{filename!r}].position", "must be an object with latitude and longitude")
    return ImagePosition(
        latitude=float(raw["latitude"]),
        longitude=float(raw["longitude"]),
        provenance=str(raw.get("provenance", "unknown")),
    )


def _parse_session_coordinates(raw: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Read an optional, paired coordinate from prior-session metadata."""
    latitude = raw.get("latitude")
    longitude = raw.get("longitude")
    if (latitude is None) != (longitude is None):
        raise InvalidFieldError("session coordinates", "latitude and longitude must be present together")
    if latitude is None:
        return None, None
    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise InvalidFieldError("session coordinates", "latitude and longitude must be numeric") from exc


def load_session_manifest(session_dir: Path, manifest_filename: str = "manifest.json") -> SessionManifest:
    """
    Load and validate ``<session_dir>/<manifest_filename>`` against the
    frozen schema. Raises a ``ManifestError`` subclass identifying
    the specific rejection reason on any structural defect; raises
    ``FileNotFoundError``/``json.JSONDecodeError`` if the manifest itself is
    absent or not valid JSON.
    """
    manifest_path = session_dir / manifest_filename
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise InvalidFieldError("<root>", "manifest must be a JSON object")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    if missing:
        raise MissingRequiredKeyError(missing)

    session_id = raw["session_id"]
    if not isinstance(session_id, str) or not session_id:
        raise InvalidFieldError("session_id", "must be a non-empty string")
    if session_id != session_dir.name:
        raise SessionIdMismatchError(session_id, session_dir.name)

    crop = raw["crop"]
    if not isinstance(crop, str) or not crop.strip():
        raise InvalidFieldError("crop", "must be a non-empty string")

    session_start = raw["session_start"]
    session_end = raw["session_end"]
    if not isinstance(session_start, str) or not session_start:
        raise InvalidFieldError("session_start", "must be a non-empty ISO-8601 string")
    if not isinstance(session_end, str) or not session_end:
        raise InvalidFieldError("session_end", "must be a non-empty ISO-8601 string")

    raw_images = raw["images"]
    if not isinstance(raw_images, list) or not raw_images:
        raise InvalidFieldError("images", "must be a non-empty array")

    images_dir = session_dir / "images"

    seen: set[str] = set()
    for entry in raw_images:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not filename or not isinstance(filename, str):
            raise InvalidFieldError("images[].filename", "must be a non-empty string")
        if filename in seen:
            raise DuplicateImageEntryError(filename)
        seen.add(filename)

    entries: list[ManifestImageEntry] = []
    for entry in raw_images:
        filename = entry["filename"]
        _validate_filename_shape(filename)

        candidate = images_dir / filename
        try:
            resolved = resolve_contained(images_dir, candidate)
        except ValueError as exc:
            raise PathContainmentError(filename, str(exc)) from exc

        if not resolved.exists() or not resolved.is_file():
            raise MissingReferencedImageError(filename)

        position = _parse_position(filename, entry.get("position"))
        timestamp = entry.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise InvalidFieldError(f"images[{filename!r}].timestamp", "must be a string when present")

        entries.append(
            ManifestImageEntry(filename=filename, path=resolved, position=position, timestamp=timestamp)
        )

    referenced = {e.filename for e in entries}
    skipped_extra: list[str] = []
    if images_dir.is_dir():
        for f in sorted(images_dir.iterdir(), key=lambda p: p.name):
            if f.is_file() and f.name not in referenced:
                skipped_extra.append(f.name)

    growth_stage = raw.get("growth_stage")
    if growth_stage is not None and not isinstance(growth_stage, dict):
        raise InvalidFieldError("growth_stage", "must be an object when present")

    pest = raw.get("pest")
    if pest is not None and (not isinstance(pest, str) or not pest.strip()):
        raise InvalidFieldError("pest", "must be a non-empty string when present")

    images_per_plant = raw.get("images_per_plant")
    if images_per_plant is not None:
        if not isinstance(images_per_plant, int) or isinstance(images_per_plant, bool) or images_per_plant <= 0:
            raise InvalidFieldError("images_per_plant", "must be a positive integer when present")

    latitude, longitude = _parse_session_coordinates(raw)

    return SessionManifest(
        schema_version=int(raw["schema_version"]),
        session_id=session_id,
        crop=crop,
        session_start=session_start,
        session_end=session_end,
        images=entries,
        latitude=latitude,
        longitude=longitude,
        growth_stage=growth_stage,
        images_per_plant=images_per_plant,
        pest=pest,
        skipped_extra_files=skipped_extra,
    )
