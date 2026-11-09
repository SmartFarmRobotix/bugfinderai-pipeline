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
Session feeder — local image-store layout and format enforcement.

Layout: ``<data_root>/sessions/<session_id>/images/*.{jpg,png}``
plus a sidecar ``manifest.json`` (see app/manifest.py for the frozen schema).
This module scans the store, loads a given session through the manifest
contract, and applies JPEG/PNG-only enforcement at the
store-scanning boundary — orthogonal to, and layered on top of, the
manifest's own structural validation (app/manifest.py never rejects on
format; it is a purely structural contract).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.manifest import ImagePosition, load_session_manifest, resolve_contained

logger = logging.getLogger("feeder.store")

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class SessionDirectoryEscapesDataRootError(ValueError):
    """
    Raised when a session_id's directory (realpath-style: symlinks and
    ``..`` collapsed) resolves outside the configured data root — the
    session-level half of the containment check, distinct from
    app/manifest.py's per-image-filename containment check (which only
    protects a session's own images/ directory, not the session directory's
    position relative to data_root itself).
    """

    def __init__(self, session_id: str, sdir: Path, data_root: Path) -> None:
        self.session_id = session_id
        super().__init__(f"session {session_id!r} directory {sdir} resolves outside data root {data_root}")


def is_supported_image_format(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


@dataclass(frozen=True)
class ReplayImage:
    filename: str
    path: Path
    position: Optional[ImagePosition]
    timestamp: Optional[str]


@dataclass(frozen=True)
class ReplaySession:
    """A validated, format-filtered, ready-to-submit replay session."""

    session_id: str
    session_dir: Path
    crop: str
    session_start: str
    session_end: str
    growth_stage: Optional[dict[str, Any]]
    images_per_plant: Optional[int]
    images: list[ReplayImage]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    pest: Optional[str] = None
    skipped_unsupported: list[str] = field(default_factory=list)
    skipped_extra: list[str] = field(default_factory=list)


def sessions_root(data_root: Path) -> Path:
    return data_root / "sessions"


def session_dir(data_root: Path, session_id: str) -> Path:
    return sessions_root(data_root) / session_id


def list_store_session_ids(data_root: Path) -> list[str]:
    """
    Session ids present in the store, in the order the store's own
    directory naming convention encodes (producers name sessions so that lexicographic
    order is chronological order — e.g. a date/timestamp-prefixed
    session_id). This component does not open every manifest just to
    establish "oldest"; that would defeat the point of a cheap store scan.
    """
    root = sessions_root(data_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_replay_session(data_root: Path, session_id: str, manifest_filename: str = "manifest.json") -> ReplaySession:
    """
    Load session *session_id* from the store: validate its manifest (fail
    closed on structural defects — see app/manifest.py), then apply
    format enforcement per referenced image (fail open — skip and log,
    never reject the whole folder for a format mismatch alone).

    Rejects, before the manifest is even opened, a session_id whose
    directory (following any symlink) resolves outside data_root —
    a session_id is store-controlled but its on-disk entry could still be a
    symlink planted to point elsewhere.
    """
    sdir = session_dir(data_root, session_id)
    try:
        resolve_contained(data_root, sdir)
    except ValueError as exc:
        raise SessionDirectoryEscapesDataRootError(session_id, sdir, data_root) from exc
    manifest = load_session_manifest(sdir, manifest_filename)

    accepted: list[ReplayImage] = []
    skipped_unsupported: list[str] = []
    for entry in manifest.images:
        if not is_supported_image_format(entry.filename):
            skipped_unsupported.append(entry.filename)
            logger.warning(
                "session %s: skipping unsupported-format file %r (JPEG/PNG only)",
                session_id,
                entry.filename,
            )
            continue
        accepted.append(
            ReplayImage(filename=entry.filename, path=entry.path, position=entry.position, timestamp=entry.timestamp)
        )

    for extra in manifest.skipped_extra_files:
        logger.info("session %s: skipping unreferenced file %r (not listed in manifest)", session_id, extra)

    return ReplaySession(
        session_id=manifest.session_id,
        session_dir=sdir,
        crop=manifest.crop,
        session_start=manifest.session_start,
        session_end=manifest.session_end,
        growth_stage=manifest.growth_stage,
        images_per_plant=manifest.images_per_plant,
        images=accepted,
        latitude=manifest.latitude,
        longitude=manifest.longitude,
        pest=manifest.pest,
        skipped_unsupported=skipped_unsupported,
        skipped_extra=manifest.skipped_extra_files,
    )
