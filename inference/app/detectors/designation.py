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
Per-execution-target designation-status resolution, keyed by the loaded
artifact's own verified sha256.

Replacing
an artifact with an approved fine-tune can now make it ``designated``
without any code change, because the manifest — not the code — is what
says so.

Manifest shape (one JSON document, sha256-keyed per target)::

    {
      "schema_version": "1.0.0",
      "targets": {
        "cloud": {"artifacts": {"<sha256>": {"designation_status": "designated"}}},
        "jetson": {"artifacts": {"<sha256>": {"designation_status": "integration-only"}}}
      }
    }

An artifact whose sha256 is absent from its target's ``artifacts`` map is
never silently treated as ``integration-only`` — resolution fails loudly.
"""

from __future__ import annotations

from typing import Any, Dict

_VALID_STATUSES = ("integration-only", "designated")


class DesignationManifestError(RuntimeError):
    """
    Raised when the designation manifest has no entry for the requested
    execution target, when the loaded artifact's verified sha256 is absent
    from that target's entry, or when an entry's ``designation_status``
    isn't a recognized value. An artifact never falls back to a default
    designation — a manifest gap is a loud failure, not a silent
    ``integration-only``.
    """


def resolve_designation_status(
    artifact_sha256: str,
    execution_target: str,
    manifest: Dict[str, Any],
) -> str:
    """
    Look up ``artifact_sha256`` under ``manifest["targets"][execution_target]``
    and return its recorded ``designation_status`` ("integration-only" or
    "designated"). Raises ``DesignationManifestError`` if the target or the
    artifact's own hash isn't present, or the recorded status is invalid.

    Per-target designation uniqueness is asserted here, on EVERY resolution
    not only in tests — so a manifest that designates two artifacts for
    the same target is rejected at executor load even when the artifact
    being resolved has a valid entry of its own.
    """
    assert_per_target_designation_uniqueness(manifest)
    targets = manifest.get("targets", {})
    target_entry = targets.get(execution_target)
    if target_entry is None:
        raise DesignationManifestError(
            f"designation manifest has no entry for execution_target {execution_target!r} "
            f"(known targets: {sorted(targets)!r})"
        )

    artifacts = target_entry.get("artifacts", {})
    entry = artifacts.get(artifact_sha256)
    if entry is None:
        raise DesignationManifestError(
            f"artifact sha256 {artifact_sha256!r} is absent from the {execution_target!r} "
            "designation manifest — refusing to silently default to integration-only"
        )

    status = entry.get("designation_status")
    if status not in _VALID_STATUSES:
        raise DesignationManifestError(
            f"invalid designation_status {status!r} for artifact {artifact_sha256!r} under "
            f"execution_target {execution_target!r}; expected one of {_VALID_STATUSES!r}"
        )
    return status


def assert_per_target_designation_uniqueness(manifest: Dict[str, Any]) -> None:
    """
    Exactly one ``designated`` entry per execution target — a manifest that designates two
    artifacts for the same target simultaneously is a manifest bug, not a
    resolvable ambiguity.
    """
    for target, entry in manifest.get("targets", {}).items():
        designated = [
            sha256 for sha256, artifact in entry.get("artifacts", {}).items()
            if artifact.get("designation_status") == "designated"
        ]
        if len(designated) > 1:
            raise DesignationManifestError(
                f"execution_target {target!r} has {len(designated)} artifacts marked "
                f"designated ({sorted(designated)!r}) — exactly one designated artifact "
                "per execution target is required"
            )
