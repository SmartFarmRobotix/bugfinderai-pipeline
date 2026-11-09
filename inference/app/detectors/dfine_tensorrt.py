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
DFineTensorRTDetector — the Jetson-only TensorRT executor. Consumes a two-input/three-output D-FINE engine.

The artifact provenance envelope (hash verification and execution-target boundary enforcement) is
shared with the ONNX-Runtime executor via ``app/detectors/artifact_provenance.py``
and the lazy tensorrt/pycuda runtime import pattern, so this module and
its config-pointed loading/provenance mechanics still unit-test cleanly on
a host with neither package installed and no GPU.

Binding is by declared tensor NAME and queried dtype only:
``_run_engine`` enumerates the engine's ``num_io_tensors`` and classifies
each by ``get_tensor_mode``/``get_tensor_name`` — never by a fixed ordinal
index — so an engine that happens to declare its five I/O tensors in any
order binds correctly.
"""

from __future__ import annotations

import json
import threading
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
    "TensorRTRuntimeUnavailable",
    "DFineEngineBindingError",
    "ArtifactImageSizeError",
    "ArtifactProvenanceError",
    "ArtifactTargetMismatchError",
    "ArtifactTilingError",
    "ClassMapResolutionError",
    "DesignationManifestError",
    "DFineTensorRTDetector",
    "provenance_path_for",
]

_INPUT_TENSOR_NAMES = {"images", "orig_target_sizes"}
_OUTPUT_TENSOR_NAMES = {"labels", "boxes", "scores"}


class TensorRTRuntimeUnavailable(RuntimeError):
    """
    Raised by ``DFineTensorRTDetector.detect`` when the tensorrt/pycuda/
    CUDA runtime is not importable on this host, or no GPU device is
    visible. Construction, config loading, and provenance mechanics never
    raise this — only the actual inference call does.
    """


class TensorRTThreadAffinityError(RuntimeError):
    """The CUDA context pycuda's
    autoinit establishes is current only on the thread that ran the
    import, so a detect arriving on any other thread would fail deep
    in the driver (`invalid device context - no currently active
    context`). This typed refusal fires
    BEFORE any runtime use instead — naming the threads involved — and
    fails closed when the owner was never recorded."""


class DFineEngineBindingError(RuntimeError):
    """
    Raised when the loaded engine's discovered input/output tensor names
    don't match the pinned D-FINE two-input (``images``,
    ``orig_target_sizes``) / three-output (``labels``, ``boxes``,
    ``scores``) contract — an engine built for a different I/O signature
    is never silently bound by position.
    """


class DFineEngineExecutionError(RuntimeError):
    """
    Raised when the engine's inference enqueue itself fails
    (``execute_async_v3`` returning False), or when an output tensor's
    shape is still unresolved (a negative dimension) after both input
    shapes were set. A failed enqueue must fail closed — never fall
    through to reading back uninitialized device buffers as if they were
    detections.
    """


class DFineTensorRTDetector(TiledPathMixin):
    """Detector protocol implementation backed by a D-FINE TensorRT engine."""

    impl_id = "dfine-tensorrt-detector"

    def __init__(
        self,
        engine_path: Path,
        execution_target: str,
        provenance_path: Optional[Path] = None,
        designation_manifest_path: Optional[Path] = None,
        tile_batch: int = DEFAULT_TILE_BATCH,
        region_threshold: float = DEFAULT_REGION_THRESHOLD,
        profile: str = LEGACY_FULL_FRAME,
        negligibility_record_dir: Optional[Path] = None,
    ) -> None:
        self.execution_target = execution_target
        self.engine_path = Path(engine_path)
        self._tile_batch = tile_batch
        self._region_threshold = region_threshold
        self._profile = get_profile(profile)

        self._provenance_path = Path(provenance_path) if provenance_path else provenance_path_for(self.engine_path)
        self._engine_provenance, self._verified_sha256 = load_and_verify_provenance(
            self.engine_path, self._provenance_path, self.execution_target
        )
        self.artifact_sha256 = self._verified_sha256
        self.score_policy_id = self._engine_provenance.get("score_policy_id", "")

        self._imgsz = resolve_image_size(self._engine_provenance, self._provenance_path)
        self._tiling = resolve_tiling(self._engine_provenance, self._provenance_path, self._imgsz)

        assert_profile_matches(self._profile, self._engine_provenance, self._provenance_path)
        self._profile_name = self._profile.name
        self._tta_views = self._profile.tta_views
        self._negligibility_record_sha256: Optional[str] = resolve_negligibility_record(
            self._profile, self._verified_sha256, negligibility_record_dir
        )

        self._class_names = resolve_class_names(self._engine_provenance)

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

        self._runtime = None
        self._runtime_owner_ident: Optional[int] = None

    @property
    def imgsz(self) -> int:
        """The model input size recorded in the artifact's provenance."""
        return self._imgsz

    @property
    def version(self) -> str:
        return self._verified_sha256[:12]

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
        model_version/model_source/model_license/conversion_chain —
        engine-terminal chain, so ``engine_artifact``/``engine_sha256``
        are populated (unlike the ORT executor's ONNX-terminal chain).
        """
        prov = self._engine_provenance
        pt_prov = prov.get("pt_provenance", {})
        return {
            "model_version": pt_prov.get("model_name", "unknown"),
            "model_source": pt_prov.get("source", "unknown"),
            "model_license": pt_prov.get("license", "unknown"),
            "conversion_chain": {
                "execution_target": prov.get("execution_target", "unknown"),
                "pt_version": pt_prov.get("file_sha256", "unknown"),
                "pt_source": pt_prov.get("source", "unknown"),
                "pt_license": pt_prov.get("license", "unknown"),
                "onnx_artifact": prov.get("source_onnx_provenance_file", "unknown"),
                "onnx_sha256": prov.get("source_onnx_sha256", "unknown"),
                "engine_artifact": prov.get("file_path", str(self.engine_path)),
                "engine_sha256": self._verified_sha256,
            },
        }

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise TensorRTRuntimeUnavailable(
                "tensorrt/pycuda runtime not available on this host — TensorRT engine "
                "execution requires the CUDA/TensorRT runtime and a GPU (the Jetson "
                "Orin's own runtime). Config-pointed loading and provenance mechanics do "
                "not need it and are covered by this repo's unit tests."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise TensorRTRuntimeUnavailable(f"failed to deserialize TensorRT engine at {self.engine_path}")
        context = engine.create_execution_context()
        self._runtime = (trt, cuda, engine, context)
        self._runtime_owner_ident = threading.get_ident()
        return self._runtime

    def _run_engine(
        self,
        trt,
        cuda,
        engine,
        context,
        images: np.ndarray,
        orig_target_sizes: np.ndarray,
    ) -> List[RawTensor]:
        """
        Enumerate the engine's tensors by name (never ordinal index), set
        the two named inputs, execute, and read the three named outputs
        back — returning them (plus the ``orig_target_sizes`` input tensor
        itself) as ``RawTensor``s ready for the shared canonicalizer.
        """
        input_arrays = {
            "images": np.ascontiguousarray(images.astype(np.float32)),
            "orig_target_sizes": np.ascontiguousarray(orig_target_sizes.astype(np.int64)),
        }
        device_mem: Dict[str, Any] = {}
        host_outputs: Dict[str, np.ndarray] = {}
        stream = cuda.Stream()

        seen_inputs: List[str] = []
        output_names: List[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name not in input_arrays:
                    raise DFineEngineBindingError(
                        f"engine declares unexpected input tensor {name!r}; expected one "
                        f"of {sorted(input_arrays)!r}"
                    )
                seen_inputs.append(name)
            else:
                output_names.append(name)

        if set(seen_inputs) != _INPUT_TENSOR_NAMES:
            raise DFineEngineBindingError(
                f"engine declares input tensors {sorted(seen_inputs)!r}; expected exactly "
                f"{sorted(_INPUT_TENSOR_NAMES)!r} (D-FINE two-input contract)"
            )
        if set(output_names) != _OUTPUT_TENSOR_NAMES:
            raise DFineEngineBindingError(
                f"engine declares output tensors {sorted(output_names)!r}; expected exactly "
                f"{sorted(_OUTPUT_TENSOR_NAMES)!r} (D-FINE three-output contract)"
            )

        for name in seen_inputs:
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            host_array = input_arrays[name].astype(np_dtype, copy=False)
            context.set_input_shape(name, host_array.shape)
            d_mem = cuda.mem_alloc(host_array.nbytes)
            context.set_tensor_address(name, int(d_mem))
            cuda.memcpy_htod_async(d_mem, host_array, stream)
            device_mem[name] = d_mem

        for name in output_names:
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = tuple(context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise DFineEngineExecutionError(
                    f"output tensor {name!r} shape {shape!r} still has unresolved "
                    "dimensions after both input shapes were set — refusing to "
                    "allocate/read an indeterminate buffer"
                )
            host_array = np.empty(shape, dtype=np_dtype)
            d_mem = cuda.mem_alloc(host_array.nbytes)
            context.set_tensor_address(name, int(d_mem))
            device_mem[name] = d_mem
            host_outputs[name] = host_array

        if not context.execute_async_v3(stream_handle=stream.handle):
            raise DFineEngineExecutionError(
                "execute_async_v3 reported failure — failing closed rather than "
                "reading back uninitialized output buffers as detections"
            )
        for name in output_names:
            cuda.memcpy_dtoh_async(host_outputs[name], device_mem[name], stream)
        stream.synchronize()

        raw_tensors = [
            RawTensor(name=name, dtype=host_outputs[name].dtype, array=host_outputs[name])
            for name in output_names
        ]
        raw_tensors.append(
            RawTensor(
                name="orig_target_sizes",
                dtype=input_arrays["orig_target_sizes"].dtype,
                array=input_arrays["orig_target_sizes"],
            )
        )
        return raw_tensors

    def _refuse_cross_thread_use(self) -> None:
        if self._runtime is not None:
            current_ident = threading.get_ident()
            if self._runtime_owner_ident is None:
                raise TensorRTThreadAffinityError(
                    "TensorRT runtime is established but no owning thread was recorded -- "
                    "refusing before any device call (fail-closed)"
                )
            if current_ident != self._runtime_owner_ident:
                raise TensorRTThreadAffinityError(
                    f"TensorRT runtime (and its CUDA context) is owned by thread "
                    f"{self._runtime_owner_ident}, but detect() was called on thread "
                    f"{current_ident} -- the context is thread-affine; refusing before any "
                    "device call. Route detector work through the dedicated detector executor."
                )

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        self._refuse_cross_thread_use()
        trt, cuda, engine, context = self._ensure_runtime()
        if self._tiling is not None:
            return self._tiled_detections(image_ref)

        images, orig_w, orig_h = preprocess_image_dfine(image_ref.path, self._imgsz)
        orig_target_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

        raw_tensors = self._run_engine(trt, cuda, engine, context, images, orig_target_sizes)
        return canonicalize_dfine(raw_tensors, class_names=self._class_names, image_id=image_ref.image_id)

    def probe(self) -> None:
        """One engine execution on a zero batch of one. Refuses a
        thread other than the runtime's owner before any device call, like
        ``detect``."""
        self._refuse_cross_thread_use()
        trt, cuda, engine, context = self._ensure_runtime()
        images = np.zeros((1, 3, self._imgsz, self._imgsz), dtype=np.float32)
        orig_target_sizes = np.array([[self._imgsz, self._imgsz]], dtype=np.int64)
        self._run_engine(trt, cuda, engine, context, images, orig_target_sizes)

    def _engine_max_batch(self, engine) -> int:
        """The largest batch the loaded engine accepts, read from the engine
        itself -- never assumed. An engine built from the dynamic-batch export
        reports -1 for the batch dimension and carries its bounds in the
        input's optimization profile (min, opt, max); an engine with a static
        batch dimension reports that number."""
        static = tuple(engine.get_tensor_shape("images"))
        if static and static[0] > 0:
            return int(static[0])
        _, _, max_shape = engine.get_tensor_profile_shape("images", 0)
        return int(max_shape[0])

    def detect_tiled_debug(self, image_ref: ImageRef) -> Dict[str, Any]:
        self._refuse_cross_thread_use()
        return super().detect_tiled_debug(image_ref)

    def _run_tiled(self, image_ref: ImageRef, keep_raw: bool = False) -> TiledResult:
        trt, cuda, engine, context = self._ensure_runtime()

        def run_batch(images: np.ndarray, orig_target_sizes: np.ndarray) -> List[RawTensor]:
            return self._run_engine(trt, cuda, engine, context, images, orig_target_sizes)

        return run_tiled(
            image_ref,
            tiling=self._tiling,
            imgsz=self._imgsz,
            class_names=self._class_names,
            run_batch=run_batch,
            batch=max(1, min(self._tile_batch, self._engine_max_batch(engine))),
            threshold=self._region_threshold,
            keep_raw=keep_raw,
            views=self._tta_views,
        )
