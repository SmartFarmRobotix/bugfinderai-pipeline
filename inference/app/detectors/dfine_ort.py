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
DFineORTDetector — the shared ONNX Runtime detector implementation: dev/CI runs it on
``CPUExecutionProvider``, the cloud execution target on
``CUDAExecutionProvider``. One implementation, one shared canonicalizer
(``app/detectors/dfine_common.py``) — provider selection is deployment
configuration, never a code fork.

Fail-closed provider selection: a request configured for a provider that
``onnxruntime.get_available_providers`` does not report at construction
time raises ``ORTProviderUnavailableError`` — this detector never silently
builds a session on a different (e.g. CPU) provider than the one it was
configured for.

``onnxruntime`` is a normal (non-GPU) dependency of this service and is imported lazily here anyway, for two reasons: (1) it keeps
the provider-availability check — which must run before anything else in
``__init__`` — cheap to unit-test without needing a real ``.onnx`` model
file on disk, and (2) it mirrors the Jetson TensorRT executor's own lazy-
runtime pattern, so neither executor pays an import cost merely to be
imported by the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.detectors.artifact_provenance import (
    ArtifactImageSizeError,
    ArtifactProvenanceError,
    ArtifactTargetMismatchError,
    ArtifactTilingError,
    load_and_verify_provenance,
    provenance_path_for,
    resolve_image_size,
    resolve_tiling,
)
from app.detectors.base import ImageDetections, ImageRef
from app.detectors.designation import DesignationManifestError, resolve_designation_status
from app.detectors.dfine_common import (
    ClassMapResolutionError,
    RawTensor,
    canonicalize_dfine,
    preprocess_image_dfine,
    resolve_class_names,
)
from app.detectors.profiles import (
    LEGACY_FULL_FRAME,
    assert_profile_matches,
    get_profile,
    resolve_negligibility_record,
)
from app.detectors.tiled_path import (
    DEFAULT_REGION_THRESHOLD,
    DEFAULT_TILE_BATCH,
    TiledPathMixin,
    TiledResult,
    run_tiled,
)

__all__ = [
    "ORTProviderUnavailableError",
    "ORTUnsupportedExecutionTargetError",
    "ArtifactImageSizeError",
    "ArtifactProvenanceError",
    "ArtifactTargetMismatchError",
    "ArtifactTilingError",
    "ClassMapResolutionError",
    "DesignationManifestError",
    "DFineORTDetector",
]


class ORTProviderUnavailableError(RuntimeError):
    """
    Raised at ``DFineORTDetector`` construction when the configured
    execution provider (e.g. ``CUDAExecutionProvider`` for the cloud
    target) is not among ``onnxruntime.get_available_providers`` on this
    host. Fail closed: this is never a silent fallback to a different
    (e.g. CPU) provider — the detector is simply not constructed.
    """


class ORTUnsupportedExecutionTargetError(RuntimeError):
    """
    Raised at ``DFineORTDetector`` construction when ``execution_target ==
    "jetson"``. The runtime matrix reserves ONNX Runtime for the dev/CI and cloud execution
    targets only — the Jetson target always runs the TensorRT engine
    (``DFineTensorRTDetector``). This is a runtime-matrix violation, not a
    provider-availability problem (that is ``ORTProviderUnavailableError``,
    above), so it is checked first, before the provider check or anything
    that touches the model file.
    """


class DFineORTDetector(TiledPathMixin):
    """Detector protocol implementation backed by an ONNX Runtime session."""

    impl_id = "dfine-ort-detector"

    def __init__(
        self,
        model_path: Path,
        execution_target: str,
        provider: str,
        provenance_path: Optional[Path] = None,
        designation_manifest_path: Optional[Path] = None,
        tile_batch: int = DEFAULT_TILE_BATCH,
        region_threshold: float = DEFAULT_REGION_THRESHOLD,
        profile: str = LEGACY_FULL_FRAME,
        negligibility_record_dir: Optional[Path] = None,
    ) -> None:
        self.execution_target = execution_target
        self.model_path = Path(model_path)
        self._provider = provider
        self._tile_batch = tile_batch
        self._region_threshold = region_threshold
        self._profile = get_profile(profile)

        self._ensure_execution_target_supported()

        self._ensure_provider_available()

        self._provenance_path = Path(provenance_path) if provenance_path else provenance_path_for(self.model_path)
        self._artifact_provenance, self._verified_sha256 = load_and_verify_provenance(
            self.model_path, self._provenance_path, self.execution_target
        )
        self.artifact_sha256 = self._verified_sha256
        self.score_policy_id = self._artifact_provenance.get("score_policy_id", "")

        self._imgsz = resolve_image_size(self._artifact_provenance, self._provenance_path)
        self._tiling = resolve_tiling(self._artifact_provenance, self._provenance_path, self._imgsz)

        assert_profile_matches(self._profile, self._artifact_provenance, self._provenance_path)
        self._profile_name = self._profile.name
        self._tta_views = self._profile.tta_views
        self._negligibility_record_sha256: Optional[str] = resolve_negligibility_record(
            self._profile, self._verified_sha256, negligibility_record_dir
        )

        self._class_names = resolve_class_names(self._artifact_provenance)

        self._designation_manifest_path = (
            Path(designation_manifest_path)
            if designation_manifest_path
            else self._provenance_path.parent / "designation-manifest.json"
        )
        self.designation_status = resolve_designation_status(
            artifact_sha256=self._verified_sha256,
            execution_target=self.execution_target,
            manifest=self._load_designation_manifest(),
        )

        self._session = None

    @property
    def imgsz(self) -> int:
        """The model input size recorded in the artifact's provenance."""
        return self._imgsz

    @property
    def version(self) -> str:
        return self._verified_sha256[:12]

    def _ensure_execution_target_supported(self) -> None:
        if self.execution_target == "jetson":
            raise ORTUnsupportedExecutionTargetError(
                f"execution_target={self.execution_target!r} is not supported by "
                "DFineORTDetector — the runtime matrix runs Jetson exclusively "
                "on the TensorRT engine (DFineTensorRTDetector); ONNX Runtime is "
                "reserved for the dev/CI and cloud execution targets"
            )

    def _ensure_provider_available(self) -> None:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        if self._provider not in available:
            raise ORTProviderUnavailableError(
                f"execution provider {self._provider!r} is not available on this host "
                f"(available: {sorted(available)!r}) — refusing to silently construct a "
                "session on a different provider"
            )

    def _load_designation_manifest(self) -> Dict[str, Any]:
        if not self._designation_manifest_path.exists():
            raise DesignationManifestError(
                f"no designation manifest at {self._designation_manifest_path} — an "
                "artifact's designation_status must be resolved from a manifest, never "
                "defaulted"
            )
        return json.loads(self._designation_manifest_path.read_text())

    def contract_provenance_fields(self) -> Dict[str, Any]:
        """
        Fields for the session contract's DetectorProvenance
        model_version/model_source/model_license/conversion_chain — an
        ONNX-terminal chain, so ``engine_artifact``/``engine_sha256`` are
        None.

        ``execution_provider`` is ``self._provider``, the provider this
        detector was configured for — not a fresh read of
        ``onnxruntime``. That is honest only because ``_ensure_session``
        already refuses to serve (``ORTProviderUnavailableError``) if the
        constructed session's actual active provider ever differs from
        ``self._provider``: by the
        time a session is served, the invariant ``self._provider ==
        session.get_providers[0]`` already holds, so recording
        ``self._provider`` here records what the constructed session
        actually reported.
        """
        prov = self._artifact_provenance
        pt_prov = prov.get("pt_provenance", {})
        return {
            "model_version": pt_prov.get("model_name", "unknown"),
            "model_source": pt_prov.get("source", "unknown"),
            "model_license": pt_prov.get("license", "unknown"),
            "execution_provider": self._provider,
            "conversion_chain": {
                "execution_target": prov.get("execution_target", "unknown"),
                "pt_version": pt_prov.get("file_sha256", "unknown"),
                "pt_source": pt_prov.get("source", "unknown"),
                "pt_license": pt_prov.get("license", "unknown"),
                "onnx_artifact": prov.get("file_path", str(self.model_path)),
                "onnx_sha256": self._verified_sha256,
                "engine_artifact": None,
                "engine_sha256": None,
            },
        }

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.model_path), providers=[self._provider])
        actual_providers = session.get_providers()
        if not actual_providers or actual_providers[0] != self._provider:
            raise ORTProviderUnavailableError(
                f"requested execution provider {self._provider!r} but the constructed "
                f"session is running on {actual_providers!r} — refusing to serve inference "
                "through a session that silently fell back to a different provider than "
                "the one requested"
            )
        self._session = session
        return self._session

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        self._ensure_session()
        if self._tiling is not None:
            return self._tiled_detections(image_ref)

        images, orig_w, orig_h = preprocess_image_dfine(image_ref.path, self._imgsz)
        orig_target_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)
        raw_tensors = self._run_session(images, orig_target_sizes)
        return canonicalize_dfine(raw_tensors, class_names=self._class_names, image_id=image_ref.image_id)

    def probe(self) -> None:
        """One session run on a zero input. Goes through
        ``_ensure_session`` so a session that lost or never got the
        requested provider is refused by the same check ``detect`` uses."""
        self._ensure_session()
        images = np.zeros((1, 3, self._imgsz, self._imgsz), dtype=np.float32)
        orig_target_sizes = np.array([[self._imgsz, self._imgsz]], dtype=np.int64)
        self._run_session(images, orig_target_sizes)

    def _run_session(self, images: np.ndarray, orig_target_sizes: np.ndarray) -> List[RawTensor]:
        session = self._ensure_session()

        output_metas = session.get_outputs()
        output_names = [meta.name for meta in output_metas]
        raw_arrays = session.run(
            output_names, {"images": images, "orig_target_sizes": orig_target_sizes}
        )

        raw_tensors: List[RawTensor] = [
            RawTensor(name=meta.name, dtype=array.dtype, array=array)
            for meta, array in zip(output_metas, raw_arrays)
        ]
        raw_tensors.append(
            RawTensor(name="orig_target_sizes", dtype=orig_target_sizes.dtype, array=orig_target_sizes)
        )
        return raw_tensors

    def _run_tiled(self, image_ref: ImageRef, keep_raw: bool = False) -> TiledResult:
        return run_tiled(
            image_ref,
            tiling=self._tiling,
            imgsz=self._imgsz,
            class_names=self._class_names,
            run_batch=self._run_session,
            batch=self._tile_batch,
            threshold=self._region_threshold,
            keep_raw=keep_raw,
            views=self._tta_views,
        )
