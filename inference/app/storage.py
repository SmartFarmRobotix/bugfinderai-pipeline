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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import UploadFile

logger = logging.getLogger("inference.storage")

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class UnsupportedImageFormatError(ValueError):
    """Raised when an image is neither JPEG nor PNG."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"Unsupported image format for {filename!r}; only JPEG/PNG are accepted")


class PathEscapesDataRootError(ValueError):
    """Raised when a registered path resolves outside the configured data root."""

    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        super().__init__(f"path {candidate!r} resolves outside the configured data root")


class PathUnreadableError(ValueError):
    """Raised when a registered (and contained) path cannot be resolved/read."""

    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        super().__init__(f"path {candidate!r} cannot be resolved or read")


class SessionManifestError(ValueError):
    """
    Raised when a path-registered session folder's manifest.json (the
    session-folder contract) is missing or does not name files path-registration can ingest.
    Path-registration ingests exactly the manifest-referenced files in
    manifest order — never a directory scan — since any conforming
    producer, not only this repository's session feeder, may register a
    path here.
    """


def is_supported_image_format(filename: str) -> bool:
    """The format gate, applied identically to HTTP-upload and path-registration."""
    suffix = Path(filename).suffix.lower()
    return suffix in SUPPORTED_IMAGE_SUFFIXES


def resolve_contained_path(data_root: Path, candidate: Path) -> Path:
    """
    Resolve *candidate* (realpath-style: symlinks and ``..`` collapsed) and
    require it to lie strictly within *data_root* (also resolved).

    Raises ``PathEscapesDataRootError`` if the resolved path lies outside
    the root — evaluated independently of, and before, any read/existence
    check.
    Raises ``PathUnreadableError`` if the (contained) path does not exist
    or is not a directory.
    """
    root = data_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PathEscapesDataRootError(str(candidate)) from None
    if not resolved.exists() or not resolved.is_dir():
        raise PathUnreadableError(str(candidate))
    return resolved


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def session_dir(base_dir: Path, session_id: str) -> Path:
    return base_dir / "sessions" / session_id


def images_dir(base_dir: Path, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "images"


def session_metadata_path(base_dir: Path, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "metadata.json"


def image_bytes_path(base_dir: Path, session_id: str, image_id: str, ext: str) -> Path:
    return images_dir(base_dir, session_id) / f"{image_id}{ext}"


def normalize_ext(filename: str, content_type: Optional[str]) -> str:
    if content_type:
        if content_type in {"image/jpeg", "image/jpg"}:
            return ".jpg"
        if content_type == "image/png":
            return ".png"
        if content_type == "image/webp":
            return ".webp"
    lower = filename.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def write_session_metadata(base_dir: Path, session_id: str, payload: dict) -> Path:
    ensure_dir(session_dir(base_dir, session_id))
    path = session_metadata_path(base_dir, session_id)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_session_metadata(base_dir: Path, session_id: str) -> dict:
    path = session_metadata_path(base_dir, session_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def session_results_path(base_dir: Path, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "results.json"


def write_session_results(base_dir: Path, session_id: str, results: list[dict]) -> Path:
    """Persist the per-image evaluation carrier, session order preserved."""
    ensure_dir(session_dir(base_dir, session_id))
    path = session_results_path(base_dir, session_id)
    path.write_text(json.dumps(results, indent=2))
    return path


def read_session_results(base_dir: Path, session_id: str) -> list[dict]:
    path = session_results_path(base_dir, session_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def list_manifest_images(session_folder: Path) -> Tuple[list[Path], list[str]]:
    """
    Read ``<session_folder>/manifest.json`` (the session-folder contract's
    sidecar manifest) and return ``(ordered image paths, skipped extra
    filenames)``: the images in exactly the manifest's declared ``images``
    array order — never filesystem/sorted-by-name order — plus the
    filenames present under ``images/`` but not referenced by any manifest
    entry (skipped and logged, not rejected — mirrors
    feeder/app/manifest.py, the producer side of the same contract). Format validation is applied by the caller per file, uniformly with the HTTP-upload
    path.

    Raises ``SessionManifestError`` (fail-closed) if ``manifest.json`` is
    absent, is not valid JSON, has no non-empty ``images`` array, or
    references a filename that is duplicated, escapes ``images/``
    (containment), or does not exist on disk.
    """
    manifest_path = session_folder / "manifest.json"
    if not manifest_path.exists():
        raise SessionManifestError(f"session folder has no manifest.json: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionManifestError(f"manifest.json is not valid JSON: {exc}") from exc

    images_field = raw.get("images") if isinstance(raw, dict) else None
    if not isinstance(images_field, list) or not images_field:
        raise SessionManifestError(f"manifest.json is missing a non-empty 'images' array: {manifest_path}")

    images_path = session_folder / "images"
    images_path_resolved = images_path.resolve()

    ordered: list[Path] = []
    referenced: set[str] = set()
    for entry in images_field:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not filename or not isinstance(filename, str):
            raise SessionManifestError(f"manifest.json has an image entry with no filename: {manifest_path}")
        if filename in referenced:
            raise SessionManifestError(f"manifest.json lists duplicate image entry {filename!r}: {manifest_path}")
        if Path(filename).is_absolute() or len(Path(filename).parts) != 1:
            raise SessionManifestError(f"manifest image entry {filename!r} violates path containment")
        referenced.add(filename)

        candidate = images_path / filename
        resolved = candidate.resolve()
        try:
            resolved.relative_to(images_path_resolved)
        except ValueError:
            raise SessionManifestError(f"manifest image entry {filename!r} resolves outside images/") from None
        if not resolved.exists() or not resolved.is_file():
            raise SessionManifestError(
                f"manifest references image {filename!r} but no such file exists under images/"
            )
        ordered.append(resolved)

    skipped_extra: list[str] = []
    if images_path.is_dir():
        for f in sorted(images_path.iterdir(), key=lambda p: p.name):
            if f.is_file() and f.name not in referenced:
                skipped_extra.append(f.name)
                logger.info(
                    "session folder %s: skipping unreferenced file %r (not listed in manifest)",
                    session_folder,
                    f.name,
                )

    return ordered, skipped_extra


def generate_image_id(content: bytes, seed: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{seed}:".encode("utf-8"))
    hasher.update(content)
    return hasher.hexdigest()[:16]


def create_session_id(provided: Optional[str] = None) -> str:
    return provided or str(uuid.uuid4())


async def save_upload_file(
    base_dir: Path,
    session_id: str,
    upload: UploadFile,
    deterministic_seed: int,
) -> Tuple[str, Path, str]:
    ensure_dir(images_dir(base_dir, session_id))
    contents = await upload.read()
    image_id = generate_image_id(contents, deterministic_seed)
    ext = normalize_ext(upload.filename or "", upload.content_type)
    path = image_bytes_path(base_dir, session_id, image_id, ext)

    await asyncio.to_thread(path.write_bytes, contents)

    return image_id, path, upload.content_type or "image/jpeg"
