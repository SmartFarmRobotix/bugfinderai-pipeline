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
Per-target acceptance smoke.

``run_acceptance_smoke`` is the gate a scenario-acceptance run passes
through before it may proceed for the execution target and detector
implementation named in config (``app.config.Settings``): it loads the
configured artifact through the same registry/detector-construction path
production traffic uses (``app.detectors.registry.get_detector``), which
already computes the artifact's verified sha256 and resolves its
``designation_status`` from the designation manifest.

Mode: acceptance runs use the real runtime and the deterministic stub is for CI/dev smoke tests only;
these are two distinct kinds of invocation, and ``Settings.acceptance_smoke_mode`` (explicit configuration,
never inferred) says which one a given call is. In ``"acceptance"`` mode
this gate adds two checks on top of construction: the stub refuses
unconditionally, and a constructed detector's ``designation_status`` must
be exactly ``"designated"`` — never a silent pass. In ``"ci-dev"`` mode
neither check applies (the stub, and a non-designated artifact, are
exactly what CI/dev smoke testing is expected to use) — only a genuine construction failure (missing/corrupt artifact,
sha256 mismatch, artifact absent from the designation manifest) still
refuses, in both modes, because those are not acceptance-policy questions.

Wired into the running production process at ``POST
/internal/acceptance-smoke`` (``app/main.py``) — that route, not this
helper directly, is what makes the refusal gate an actual enforcement
point rather than a unit-tested function only. Covered by ``app/tests/test_acceptance_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.detectors.artifact_provenance import ArtifactProvenanceError
from app.detectors.base import Detector
from app.detectors.designation import DesignationManifestError
from app.detectors.registry import get_detector

_STUB_IMPLS = ("lookup", "null")


class AcceptanceSmokeRefusal(RuntimeError):
    """
    Raised by ``run_acceptance_smoke`` when the per-target acceptance-smoke
    gate refuses to let a scenario-acceptance run proceed. Always carries
    the specific reason — never a generic/blanket message.
    """


def run_acceptance_smoke() -> Detector:
    """
    Runs the acceptance-smoke gate for the execution target, detector
    implementation, and mode currently configured in ``app.config.Settings``
    (``INFERENCE_EXECUTION_TARGET`` / ``INFERENCE_DETECTOR_IMPL`` /
    ``INFERENCE_ACCEPTANCE_SMOKE_MODE``).

    In ``"acceptance"`` mode, returns the constructed detector on success
    (its loaded artifact's ``designation_status`` resolved to exactly
    ``"designated"``); raises ``AcceptanceSmokeRefusal`` for the stub, a
    non-designated artifact, an artifact absent from the designation
    manifest, or a hash-mismatched artifact.

    In ``"ci-dev"`` mode, the stub and a non-designated artifact both
    proceed — only a genuine construction failure (corrupt or missing artifact, hash mismatch,
    artifact absent from the designation manifest) still raises ``AcceptanceSmokeRefusal``,
    since that is not an acceptance-policy question in either mode.
    """
    from app.config import get_settings

    settings = get_settings()
    execution_target = settings.execution_target
    detector_impl = settings.detector_impl
    mode = settings.acceptance_smoke_mode

    if mode == "acceptance" and detector_impl in _STUB_IMPLS:
        raise AcceptanceSmokeRefusal(
            f"execution_target={execution_target!r}: detector_impl={detector_impl!r} is "
            "the deterministic stub — excluded from acceptance runs by construction, "
            "never loaded as a candidate artifact"
        )

    try:
        detector = get_detector(detector_impl)
    except (ArtifactProvenanceError, DesignationManifestError) as exc:
        raise AcceptanceSmokeRefusal(
            f"execution_target={execution_target!r} detector_impl={detector_impl!r} "
            f"mode={mode!r}: configured artifact failed to load: {exc}"
        ) from exc

    if mode == "acceptance":
        designation_status = getattr(detector, "designation_status", None)
        if designation_status != "designated":
            raise AcceptanceSmokeRefusal(
                f"execution_target={execution_target!r} detector_impl={detector_impl!r}: "
                f"loaded artifact's designation_status={designation_status!r}, not "
                "'designated' — an acceptance run requires that target's designated entry"
            )


    return detector
