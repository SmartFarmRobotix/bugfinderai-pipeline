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

"""Detector implementation registry, selected by config."""

from __future__ import annotations

from typing import Callable, Dict

from app.detectors.base import Detector
from app.detectors.lookup import LookupDetector
from app.detectors.null import NullDetector


def _ort_detector_factory() -> Detector:
    from app.config import get_settings
    from app.detectors.dfine_ort import DFineORTDetector

    settings = get_settings()
    return DFineORTDetector(
        model_path=settings.onnx_model_path,
        execution_target=settings.execution_target,
        provider=settings.onnx_execution_provider,
        provenance_path=settings.onnx_provenance_path,
        tile_batch=settings.tile_batch,
        region_threshold=settings.region_threshold,
        profile=settings.detector_profile,
        negligibility_record_dir=settings.negligibility_record_dir,
    )


def _tensorrt_detector_factory() -> Detector:
    from app.config import get_settings
    from app.detectors.dfine_tensorrt import DFineTensorRTDetector

    settings = get_settings()
    return DFineTensorRTDetector(
        engine_path=settings.engine_path,
        execution_target=settings.execution_target,
        provenance_path=settings.engine_provenance_path,
        tile_batch=settings.tile_batch,
        region_threshold=settings.region_threshold,
        profile=settings.detector_profile,
        negligibility_record_dir=settings.negligibility_record_dir,
    )


_REGISTRY: Dict[str, Callable[[], Detector]] = {
    "lookup": LookupDetector,
    "null": NullDetector,
    "ort": _ort_detector_factory,
    "tensorrt": _tensorrt_detector_factory,
}


_ARTIFACT_LESS = frozenset({"lookup", "null"})


def get_detector(impl_name: str) -> Detector:
    try:
        factory = _REGISTRY[impl_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown detector implementation {impl_name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from exc
    if impl_name in _ARTIFACT_LESS:
        from app.config import get_settings
        from app.detectors.profiles import LEGACY_FULL_FRAME, ProfileRefusal

        profile = get_settings().detector_profile
        if profile != LEGACY_FULL_FRAME:
            raise ProfileRefusal(
                f"detector profile {profile!r} declares expectations of an artifact's provenance, "
                f"but detector implementation {impl_name!r} has no artifact -- refusing to start"
            )
    return factory()
