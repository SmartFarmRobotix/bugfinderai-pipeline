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
Versioned session-level inference contract.

This module is the single source of truth for the contract's shape. It is
owned by the inference service and MIRRORED BYTE-FOR-BYTE (copied, not
imported) into its consumer, orchestrator/app/contract.py.

Because it is mirrored, this module MUST NOT import anything outside the
standard library and pydantic — no service-specific settings, ORM models, or
relative imports. Keep it self-contained so the copies stay identical.

contract_version bump rules
----------------------------
CONTRACT_VERSION is a semver string "MAJOR.MINOR.PATCH":
  MAJOR: breaking change — a required field removed/renamed, or an
    existing field's meaning/type changes incompatibly. Consumers on an
    older MAJOR MUST reject the submission (see check_contract_version).
  MINOR: additive, backward-compatible change — a new optional field, a
    new enum member that old consumers can ignore. Older consumers accept
    with a warning.
  PATCH: no schema change (docstring/comment only).
"""

from __future__ import annotations

from typing import List, Literal, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "1.5.0"

EPPO_CODE_PATTERN = r"^[A-Z0-9]{1,8}$"

_SUPPORTED_MAJOR = int(CONTRACT_VERSION.split(".", 1)[0])


class GrowthStage(BaseModel):
    """
    Growth-stage representation, per the versioned risk-input contract.

    ``representation`` is a deployment-level configuration choice fixed for
    a given contract_version's deployment — never mixed per-session:
      "bbch_range": growth stage is a BBCH general/extended-scale numeric
        range (``bbch_min``, ``bbch_max``, integers 0-99 inclusive).
      "enum": growth stage is an enumerated crop-growth-stage identifier
        (``stage_id``), drawn from a documented, versioned enumeration
        owned by the deployment (not by this contract module).

    Exactly one of the two forms is present for a non-unknown instance;
    when the operator/campaign did not supply a growth stage, ``unknown``
    is set True and the value fields for the chosen representation are
    left absent — the field is never silently omitted from the payload,
    only its value is marked explicitly unknown.
    """

    model_config = ConfigDict(extra="forbid")

    representation: Literal["bbch_range", "enum"]
    unknown: bool = False
    bbch_min: Optional[int] = Field(default=None, ge=0, le=99)
    bbch_max: Optional[int] = Field(default=None, ge=0, le=99)
    stage_id: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_form_or_unknown(self) -> "GrowthStage":
        if self.representation == "bbch_range":
            if self.stage_id is not None:
                raise ValueError(
                    "stage_id must not be set when representation='bbch_range'"
                )
            if self.unknown:
                if self.bbch_min is not None or self.bbch_max is not None:
                    raise ValueError(
                        "bbch_min/bbch_max must be absent when unknown=True"
                    )
            else:
                if self.bbch_min is None or self.bbch_max is None:
                    raise ValueError(
                        "bbch_min and bbch_max are both required when "
                        "representation='bbch_range' and unknown=False"
                    )
                if self.bbch_min > self.bbch_max:
                    raise ValueError("bbch_min must be <= bbch_max")
        else:
            if self.bbch_min is not None or self.bbch_max is not None:
                raise ValueError(
                    "bbch_min/bbch_max must not be set when representation='enum'"
                )
            if self.unknown:
                if self.stage_id is not None:
                    raise ValueError("stage_id must be absent when unknown=True")
            elif not self.stage_id:
                raise ValueError(
                    "stage_id is required when representation='enum' and unknown=False"
                )
        return self

    @classmethod
    def unknown_for(cls, representation: Literal["bbch_range", "enum"]) -> "GrowthStage":
        """The explicit-unknown marker for a deployment's chosen representation."""
        return cls(representation=representation, unknown=True)


class RiskInput(BaseModel):
    """
    Scalar-only risk-engine payload. Deliberately does NOT contain
    best_image_id, bounding boxes, per-image labels, or per-image
    confidences — ``extra="forbid"`` makes any such field a validation
    error rather than a silent pass-through.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    infestation_score: float = Field(..., ge=0.0, le=1.0)
    crop: str
    growth_stage: GrowthStage
    time_start: str
    time_end: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    position_accuracy_m: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("session_id", "crop", "time_start", "time_end")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value


class ConversionChainProvenance(BaseModel):
    """
    Full checkpoint -> ONNX [-> engine] conversion-chain provenance for a real
    detector's session.
    Absent for non-real detectors (stub/lookup/null) — see
    DetectorProvenance.conversion_chain.

    The engine sub-fields (``engine_artifact``/``engine_sha256``) are
    OPTIONAL:
    present for a Jetson TensorRT session (engine-terminal chain), absent
    for an ONNX Runtime dev/cloud session.
    """

    model_config = ConfigDict(extra="forbid")

    execution_target: Literal["cloud", "jetson"]
    pt_version: str
    pt_source: str
    pt_license: str
    onnx_artifact: str
    onnx_sha256: str
    engine_artifact: Optional[str] = None
    engine_sha256: Optional[str] = None

    @model_validator(mode="after")
    def _engine_fields_paired(self) -> "ConversionChainProvenance":
        if (self.engine_artifact is None) != (self.engine_sha256 is None):
            raise ValueError(
                "engine_artifact and engine_sha256 must be present together or absent "
                f"together, got engine_artifact={self.engine_artifact!r}, "
                f"engine_sha256={self.engine_sha256!r}"
            )
        return self


class DetectorProvenance(BaseModel):
    """Which detector implementation produced a session's results."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    detector_impl_id: str
    detector_version: str
    designation_status: Literal["stub", "integration-only", "designated"]
    aggregation_rule_version: str
    target_class_set: List[str]
    confidence_floor: float = Field(..., ge=0.0, le=1.0)
    model_version: Optional[str] = None
    model_source: Optional[str] = None
    model_license: Optional[str] = None
    conversion_chain: Optional[ConversionChainProvenance] = None
    execution_provider: Optional[str] = None


class ReportingEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    best_image_id: str
    image_count: int = Field(..., ge=0)
    time_start: str
    time_end: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    position_accuracy_m: Optional[float] = Field(default=None, ge=0.0)
    parcel_id: Optional[str] = None
    pest: Optional[str] = Field(default=None, pattern=EPPO_CODE_PATTERN)
    provenance: DetectorProvenance


class SessionSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Optional[str] = None
    risk_input: RiskInput
    reporting_envelope: ReportingEnvelope


class ApproxPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float
    longitude: float


PositionProvenance = Literal["gps", "odometry", "replay", "default"]


class PerImageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    position: Optional[ApproxPosition] = None
    position_provenance: Optional[PositionProvenance] = None

    @model_validator(mode="after")
    def _position_and_provenance_are_paired(self) -> "PerImageResult":
        if (self.position is None) != (self.position_provenance is None):
            raise ValueError("position and position_provenance must be present together")
        return self


class SessionResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(default=CONTRACT_VERSION)
    session_id: str
    results: List[PerImageResult]


class VersionCheck(NamedTuple):
    accepted: bool
    warning: Optional[str]


def contract_major_version(version: str) -> int:
    return int(version.split(".", 1)[0])


def check_contract_version(version: Optional[str]) -> VersionCheck:
    """
    Consumer-side version gate: accept + warn on missing version, reject
    unknown major, silently accept a known/current major.
    """
    if version is None or not version.strip():
        return VersionCheck(
            True,
            f"session submission missing contract_version; assuming {CONTRACT_VERSION}",
        )
    try:
        major = contract_major_version(version)
    except (ValueError, IndexError):
        return VersionCheck(False, f"unparseable contract_version: {version!r}")
    if major != _SUPPORTED_MAJOR:
        return VersionCheck(
            False,
            f"unsupported contract_version major {major} "
            f"(this consumer supports major {_SUPPORTED_MAJOR})",
        )
    return VersionCheck(True, None)
