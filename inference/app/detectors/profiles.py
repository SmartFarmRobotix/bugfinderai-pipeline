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
Detector profiles: a registry of
DECLARED expectations, checked against the artifact's provenance at startup.

``INFERENCE_DETECTOR_PROFILE`` names a scenario's bundle; the profile does
not configure the artifact, it states what the artifact must be, and startup
refuses -- naming the profile, the disagreeing field, the required value and
the recorded value -- when the loaded provenance disagrees. Same shape as
``designation.resolve_designation_status``. The existing ``scenario``
setting means network topology and is not reused.

| profile | variant | precision | tiling | tta_views | negligibility record |
|--------------------|---------|-----------|--------------|--------------------|----------------------|
| legacy-full-frame | any | any | not required | identity | no |
| edge-s-fp16 | s | fp16 | required | identity | no |
| cloud-l-fp32 | l | fp32 | required | identity | no |
| cloud-l-fp32-tta | l | fp32 | required | identity, hflip | no |
| edge-s-int8 | s | int8 | required | identity | required |

There is no cascade profile: a cascade is two artifacts and a hand-off rule,
and no artifact contract for that exists to gate.

The INT8 profile additionally needs a negligibility record for the exact
artifact, at ``<record dir>/<artifact sha256>.negligibility.json``: a JSON
object whose ``artifact_sha256`` equals the loaded artifact's. Presence and
sha equality are the whole gate here; what the record's criterion must be is
measured where the INT8 model runs, not designed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.detectors.artifact_provenance import (
    ArtifactProvenanceError,
    resolve_image_size,
    resolve_tiling,
)

LEGACY_FULL_FRAME = "legacy-full-frame"


class ProfileRefusal(ArtifactProvenanceError):
    """A detector profile the loaded artifact (or the deployment) does not
    satisfy: startup refuses, by name."""


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    variant: Optional[str]
    precision: Optional[str]
    requires_tiling: bool
    tta_views: Tuple[str, ...]
    requires_negligibility_record: bool


PROFILES: Dict[str, DetectorProfile] = {
    profile.name: profile
    for profile in (
        DetectorProfile(LEGACY_FULL_FRAME, None, None, False, ("identity",), False),
        DetectorProfile("edge-s-fp16", "s", "fp16", True, ("identity",), False),
        DetectorProfile("cloud-l-fp32", "l", "fp32", True, ("identity",), False),
        DetectorProfile("cloud-l-fp32-tta", "l", "fp32", True, ("identity", "hflip"), False),
        DetectorProfile("edge-s-int8", "s", "int8", True, ("identity",), True),
    )
}


def get_profile(name: str) -> DetectorProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ProfileRefusal(
            f"unknown detector profile {name!r}; available: {sorted(PROFILES)!r}"
        ) from None


def _refuse(profile: DetectorProfile, field: str, required: Any, recorded: Any, provenance_path: Path) -> None:
    raise ProfileRefusal(
        f"detector profile {profile.name!r} requires {field} {required!r} but the provenance at "
        f"{provenance_path} records {recorded!r} -- refusing to start"
    )


def _assert_string_field(profile: DetectorProfile, provenance: Dict[str, Any], field: str, required: Optional[str], path: Path) -> None:
    if required is None:
        return
    if field not in provenance:
        _refuse(profile, field, required, "<missing>", path)
    recorded = provenance[field]
    if not isinstance(recorded, str):
        _refuse(profile, field, required, recorded, path)
    if recorded != required:
        _refuse(profile, field, required, recorded, path)


def assert_profile_matches(
    profile: DetectorProfile, provenance: Dict[str, Any], provenance_path: Path = Path("<provenance>")
) -> None:
    """Refuse, naming profile, field, required and recorded value, unless the
    provenance satisfies every declared expectation: exact equality of
    ``variant`` and ``precision`` (a missing key or a non-string value is a
    disagreement too), and a present, valid ``tiling`` where the profile
    requires one. The default profile declares nothing and accepts any record.
    """
    _assert_string_field(profile, provenance, "variant", profile.variant, provenance_path)
    _assert_string_field(profile, provenance, "precision", profile.precision, provenance_path)
    if profile.requires_tiling:
        if "tiling" not in provenance:
            _refuse(profile, "tiling", "a {tile, stride} grid", "<missing>", provenance_path)
        resolve_tiling(provenance, provenance_path, resolve_image_size(provenance, provenance_path))


def negligibility_record_path(record_dir: Path, artifact_sha256: str) -> Path:
    return Path(record_dir) / f"{artifact_sha256}.negligibility.json"


def resolve_negligibility_record(
    profile: DetectorProfile, artifact_sha256: str, record_dir: Optional[Path]
) -> Optional[str]:
    """The sha256 of the negligibility record for the loaded artifact, or a
    refusal naming the profile and the path. ``None`` when the profile needs
    no record.

    ``artifact_sha256`` is the artifact's identity: the provenance record's
    ``file_sha256``, which ``load_and_verify_provenance`` has already checked
    against the bytes on disk. The record's path and the record's own
    ``artifact_sha256`` must both carry exactly that value; any other 64-hex
    string -- a source ONNX's, a checkpoint's -- refuses."""
    if not profile.requires_negligibility_record:
        return None
    if record_dir is None:
        raise ProfileRefusal(
            f"detector profile {profile.name!r} needs a negligibility record for artifact "
            f"{artifact_sha256} but no record directory is configured (INFERENCE_NEGLIGIBILITY_RECORD_DIR)"
        )
    path = negligibility_record_path(record_dir, artifact_sha256)
    if not path.exists():
        raise ProfileRefusal(
            f"detector profile {profile.name!r} needs a negligibility record for artifact "
            f"{artifact_sha256} but none exists at {path} -- refusing to start"
        )
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ProfileRefusal(
            f"detector profile {profile.name!r}: the negligibility record at {path} is unreadable "
            f"or not JSON ({exc})"
        ) from exc
    if not isinstance(record, dict):
        raise ProfileRefusal(
            f"detector profile {profile.name!r}: the negligibility record at {path} is not a JSON "
            f"object (got {type(record).__name__})"
        )
    recorded = record.get("artifact_sha256")
    if recorded != artifact_sha256:
        raise ProfileRefusal(
            f"detector profile {profile.name!r}: the negligibility record at {path} names artifact "
            f"sha256 {recorded!r}, not the loaded artifact's {artifact_sha256!r}"
        )
    return hashlib.sha256(raw).hexdigest()
