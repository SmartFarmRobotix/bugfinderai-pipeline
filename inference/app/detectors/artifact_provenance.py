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
Artifact provenance mechanics — hash verification and execution-target
boundary enforcement — shared by both D-FINE runtime executors (the
ONNX-terminal ORT detector's ``.onnx`` artifact and the engine-terminal
Jetson TensorRT detector's ``.engine`` artifact). The mechanics apply to any terminal
artifact kind — the mechanics (never trust a manifest's claimed hash
without recomputing it from the real file bytes; never load an artifact
built for a different execution target) are identical regardless of
whether the artifact is an ONNX file or a TensorRT engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ArtifactProvenanceError(RuntimeError):
    """
    Raised when an artifact's companion provenance manifest is missing or
    malformed, or when the manifest's declared ``file_sha256`` does not
    match the sha256 actually computed from the artifact file's bytes on
    disk. An artifact is never loaded on trust — its provenance manifest
    is a claim, verified against the real artifact bytes at every
    construction — this is a hard precondition, not a fallback-to-defaults
    situation.
    """


class ArtifactTargetMismatchError(ArtifactProvenanceError):
    """
    Raised when the execution target a detector was constructed for does
    not match the execution target recorded in its artifact's own
    provenance manifest. Artifacts (ONNX sessions configured for a
    provider, TensorRT engines) are execution-target-specific by
    construction — an artifact built/designated for one target is never
    silently accepted by a detector configured for another.
    """


class ArtifactImageSizeError(ArtifactProvenanceError):
    """
    Raised at detector construction when the artifact's provenance carries
    no usable ``imgsz`` (absent, zero, or not a positive integer). The image
    size is a property of the artifact, never of the code: no constructor
    carries a default, so an artifact with no recorded size cannot be served.
    """


class ArtifactTilingError(ArtifactProvenanceError):
    """
    Raised at detector construction when a provenance ``tiling`` key is
    PRESENT but its value is not ``{tile: int > 0, stride: int, 0 < stride
    <= tile}`` (or disagrees with the recorded ``imgsz``). Presence is
    decided by key, never truthiness: a malformed value is refused with the
    value quoted and is never read as "no tiling".
    """


@dataclass(frozen=True)
class TilingSpec:
    """The validated ``tiling`` provenance member: crop size and stride, px."""

    tile: int
    stride: int


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_image_size(manifest: Dict[str, Any], provenance_path: Path) -> int:
    """The artifact's recorded model input size (``imgsz``), or a refusal
    naming the field and the provenance path. Shared by both D-FINE
    detector classes."""
    if "imgsz" not in manifest:
        raise ArtifactImageSizeError(
            f"provenance manifest at {provenance_path} has no 'imgsz' field -- the model input "
            "size comes from the artifact's provenance, never from a default"
        )
    imgsz = manifest["imgsz"]
    if not _is_plain_int(imgsz) or imgsz <= 0:
        raise ArtifactImageSizeError(
            f"provenance manifest at {provenance_path} records 'imgsz' {imgsz!r} -- it must be a "
            "positive integer (the model input size); an absent or zero size cannot be served"
        )
    return imgsz


def resolve_tiling(manifest: Dict[str, Any], provenance_path: Path, imgsz: int) -> Optional[TilingSpec]:
    """
    The artifact's tiling grid, or ``None`` for the full-frame path.

    The provenance KEY ``tiling`` selects the tiled path -- membership, not
    truthiness (the ``class_names`` rule of ``resolve_class_names``): key
    absent -> ``None``; key present with any value other than an object with
    integer ``tile > 0`` and integer ``stride`` with ``0 < stride <= tile``
    -> ``ArtifactTilingError`` quoting the value and the path. The tile is
    also the model input, so a ``tile`` that differs from ``imgsz`` refuses
    too. One validator for both detector classes.
    """
    if "tiling" not in manifest:
        return None
    value = manifest["tiling"]
    well_formed = (
        isinstance(value, dict)
        and _is_plain_int(value.get("tile"))
        and _is_plain_int(value.get("stride"))
        and value["tile"] > 0
        and 0 < value["stride"] <= value["tile"]
    )
    if not well_formed:
        raise ArtifactTilingError(
            f"provenance manifest at {provenance_path} records 'tiling' {value!r} -- a present "
            "tiling value must be an object with integer tile > 0 and integer stride, "
            "0 < stride <= tile; it is never read as 'no tiling'"
        )
    if value["tile"] != imgsz:
        raise ArtifactTilingError(
            f"provenance manifest at {provenance_path} records 'tiling' {value!r} but 'imgsz' "
            f"{imgsz!r} -- the tile is the model input, so the two must be the same number"
        )
    return TilingSpec(tile=value["tile"], stride=value["stride"])


def provenance_path_for(artifact_path: Path) -> Path:
    """The default companion-manifest path for a given artifact file."""
    return artifact_path.parent / f"{artifact_path.name}.provenance.json"


def sha256_of_file(path: Path) -> str:
    """Sha256 of a file's actual on-disk bytes — never trust a manifest's
    claimed hash without recomputing this against the real artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_provenance(
    artifact_path: Path,
    provenance_path: Path,
    execution_target: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Load ``provenance_path``'s manifest, verify its claimed ``file_sha256``
    against ``artifact_path``'s actual bytes, and verify its
    ``execution_target`` matches the caller's. Returns
    ``(manifest, verified_sha256)``.

    Raises ``ArtifactProvenanceError`` (or its ``ArtifactTargetMismatchError``
    subclass) rather than ever silently accepting an unverified or
    wrong-target artifact.
    """
    if not artifact_path.exists():
        raise ArtifactProvenanceError(f"artifact file not found: {artifact_path}")
    if not provenance_path.exists():
        raise ArtifactProvenanceError(
            f"no provenance manifest at {provenance_path} for artifact {artifact_path} — "
            "every artifact file must carry conversion-chain provenance"
        )
    manifest = json.loads(provenance_path.read_text())

    declared_sha256 = manifest.get("file_sha256")
    if not declared_sha256:
        raise ArtifactProvenanceError(
            f"provenance manifest at {provenance_path} has no file_sha256 — "
            "every artifact's provenance must declare the hash it claims"
        )
    actual_sha256 = sha256_of_file(artifact_path)
    if actual_sha256 != declared_sha256:
        raise ArtifactProvenanceError(
            f"artifact file sha256 mismatch at {artifact_path}: provenance manifest "
            f"{provenance_path} declares {declared_sha256!r} but the file's actual "
            f"bytes hash to {actual_sha256!r} — refusing to load a tampered or stale artifact"
        )

    manifest_target = manifest.get("execution_target")
    if manifest_target != execution_target:
        raise ArtifactTargetMismatchError(
            f"artifact at {artifact_path} was built/designated for execution_target "
            f"{manifest_target!r} but this detector is configured for execution_target "
            f"{execution_target!r} — refusing to load an artifact across execution "
            "targets"
        )

    return manifest, actual_sha256
