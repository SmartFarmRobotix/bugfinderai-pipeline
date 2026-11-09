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
Session feeder — Configuration module.

Runtime settings for the session-feeder: the Jetson-resident, x86-parity
component that replays locally-stored prior-session images as new inference
sessions. One code path, one Settings model — x86 (CI/dev) and the Jetson
differ only by which values are configured, never by a
code branch.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("feeder.config")

DEFAULT_UPLOAD_STALL_SECONDS = 60
DEFAULT_RESPONSE_WAIT_SECONDS = 300


class DeploymentProfile(str, Enum):
    """Which deployment this feeder instance is running as."""

    CLOUD = "cloud"
    MIXED = "mixed"
    EDGE = "edge"
    X86_CI = "x86_ci"
    AUTO = "auto"


class IngestionMode(str, Enum):
    HTTP_UPLOAD = "http_upload"
    PATH = "path"


_MODE_BY_PROFILE: dict[DeploymentProfile, IngestionMode] = {
    DeploymentProfile.CLOUD: IngestionMode.HTTP_UPLOAD,
    DeploymentProfile.X86_CI: IngestionMode.HTTP_UPLOAD,
    DeploymentProfile.MIXED: IngestionMode.PATH,
    DeploymentProfile.EDGE: IngestionMode.PATH,
}


class Settings(BaseSettings):
    """Session-feeder runtime configuration."""

    deployment_profile: DeploymentProfile = Field(
        default=DeploymentProfile.X86_CI,
        alias="REPLAY_DEPLOYMENT_PROFILE",
        description="cloud | mixed | edge | x86_ci | auto — selects the ingestion mode (auto: chosen per session by the link probe)",
    )
    log_level: str = Field(default="info", alias="REPLAY_LOG_LEVEL")

    data_root: Path = Field(
        default=Path("/app/data"),
        alias="REPLAY_DATA_ROOT",
        description="Root containing sessions/<session_id>/images/*.{jpg,png} + manifest.json",
    )
    manifest_filename: str = Field(default="manifest.json", alias="REPLAY_MANIFEST_FILENAME")
    marker_filename: str = Field(
        default=".replayed.json",
        alias="REPLAY_MARKER_FILENAME",
        description="Local progress marker tracking which stored sessions have been replayed",
    )
    session_id_override: Optional[str] = Field(
        default=None,
        alias="REPLAY_SESSION_ID",
        description="Explicit session id override for targeted replay",
    )

    default_latitude: Optional[float] = Field(default=None, alias="REPLAY_DEFAULT_LATITUDE")
    default_longitude: Optional[float] = Field(default=None, alias="REPLAY_DEFAULT_LONGITUDE")

    inference_base_url: str = Field(default="http://inference:8010", alias="REPLAY_INFERENCE_BASE_URL")
    inference_api_key: str = Field(default="", alias="REPLAY_INFERENCE_API_KEY")
    inference_session_endpoint: str = Field(default="/api/v1/sessions", alias="REPLAY_INFERENCE_SESSION_ENDPOINT")
    inference_path_endpoint: str = Field(default="/api/v1/sessions/path", alias="REPLAY_INFERENCE_PATH_ENDPOINT")
    inference_push_endpoint_template: str = Field(
        default="/api/v1/sessions/{session_id}/push", alias="REPLAY_INFERENCE_PUSH_ENDPOINT"
    )
    inference_visible_data_root: Optional[Path] = Field(
        default=None, alias="REPLAY_INFERENCE_VISIBLE_DATA_ROOT"
    )

    cloud_inference_base_url: Optional[str] = Field(default=None, alias="REPLAY_CLOUD_INFERENCE_BASE_URL")
    cloud_inference_api_key: Optional[str] = Field(
        default=None,
        alias="REPLAY_CLOUD_INFERENCE_API_KEY",
        description="Defaults to inference_api_key when unset (effective_cloud_inference_api_key).",
    )
    probe_samples: int = Field(default=3, alias="REPLAY_PROBE_SAMPLES")
    probe_payload_bytes: int = Field(
        default=32768,
        alias="REPLAY_PROBE_PAYLOAD_BYTES",
        description=(
            "Small enough that a slow-but-alive link measures instead of failing -- "
            "at the default per-sample bound, links down to single-digit kbps complete a "
            "sample well inside the bound."
        ),
    )
    probe_min_sample_seconds: float = Field(
        default=1.0,
        alias="REPLAY_PROBE_MIN_SAMPLE_SECONDS",
        description=(
            "The probe first measures the round trip with a 1-byte upload, then doubles the "
            "payload from probe_payload_bytes until one upload lasts at least this many "
            "seconds (or the payload reaches probe_max_payload_bytes), so the transfer, not "
            "the round trip, dominates each sample; each sample is then bytes over elapsed "
            "minus the round trip. 0 switches the round-trip upload, the ramp and the "
            "correction off: probe_samples uploads of probe_payload_bytes, bytes over elapsed."
        ),
    )
    probe_max_payload_bytes: int = Field(
        default=1048576,
        alias="REPLAY_PROBE_MAX_PAYLOAD_BYTES",
        description=(
            "The ramp never uploads more than this. It is bound by the inference service's "
            "probe sink cap (INFERENCE_PROBE_SINK_MAX_BYTES, default 1048576): a ramp step "
            "the sink refuses as too large (HTTP 413) settles the size one step below."
        ),
    )
    probe_timeout_seconds: float = Field(
        default=45.0,
        alias="REPLAY_PROBE_TIMEOUT_SECONDS",
        description=(
            "A per-operation STALL bound (connect / single read / single write), not a "
            "total per-sample deadline: a sample fails only when the transport cannot make "
            "progress within the bound, never merely because the transfer is slow."
        ),
    )

    tls_verify: bool = Field(default=True, alias="REPLAY_TLS_VERIFY")
    upload_stall_seconds: Optional[float] = Field(
        default=None,
        alias="REPLAY_UPLOAD_STALL_SECONDS",
        description="How long a single transport write may sit without progress before the upload is declared dead.",
    )
    response_wait_seconds: Optional[float] = Field(
        default=None,
        alias="REPLAY_RESPONSE_WAIT_SECONDS",
        description="How long to await the service's answer after the body is sent (sized for inline CPU detection).",
    )
    request_timeout_seconds: Optional[float] = Field(
        default=None,
        alias="REPLAY_REQUEST_TIMEOUT_SECONDS",
        description="Deprecated: fallback for both new bounds when they are unset. A startup log line names the deprecation when this is set.",
    )
    auto_push: bool = Field(default=True, alias="REPLAY_AUTO_PUSH")
    parcel_id: Optional[str] = Field(default="parcel-001", alias="REPLAY_PARCEL_ID")

    result_file: Optional[Path] = Field(
        default=None,
        alias="REPLAY_RESULT_FILE",
        description="If set, write a JSON summary of the real push API response here",
    )
    accounting_file: Optional[Path] = Field(
        default=None,
        alias="REPLAY_ACCOUNTING_FILE",
        description=(
            "If set, additionally mirror this run's network-accounting record here "
            "-- for direct-python invocations with no shared container "
            "volume to read app/accounting.py's own <data_root>/accounting/ store from."
        ),
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    @field_validator("inference_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("cloud_inference_base_url")
    @classmethod
    def _strip_cloud_trailing_slash(cls, v: Optional[str]) -> Optional[str]:
        return v.rstrip("/") if v else v

    @model_validator(mode="after")
    def _default_coordinates_are_paired(self) -> "Settings":
        if (self.default_latitude is None) != (self.default_longitude is None):
            raise ValueError(
                "REPLAY_DEFAULT_LATITUDE and REPLAY_DEFAULT_LONGITUDE must be set together"
            )
        return self

    @model_validator(mode="after")
    def _probe_sizing_is_valid(self) -> "Settings":
        if self.probe_payload_bytes <= 0:
            raise ValueError(
                f"REPLAY_PROBE_PAYLOAD_BYTES must be positive, got {self.probe_payload_bytes}"
            )
        if self.probe_min_sample_seconds < 0:
            raise ValueError(
                f"REPLAY_PROBE_MIN_SAMPLE_SECONDS must not be negative, got {self.probe_min_sample_seconds}"
            )
        if self.probe_max_payload_bytes < self.probe_payload_bytes:
            raise ValueError(
                f"REPLAY_PROBE_MAX_PAYLOAD_BYTES ({self.probe_max_payload_bytes}) must not be below "
                f"REPLAY_PROBE_PAYLOAD_BYTES ({self.probe_payload_bytes})"
            )
        return self

    @model_validator(mode="after")
    def _log_request_timeout_deprecation(self) -> "Settings":
        if self.request_timeout_seconds is not None:
            logger.warning(
                "REPLAY_REQUEST_TIMEOUT_SECONDS is deprecated -- use REPLAY_UPLOAD_STALL_SECONDS "
                "and REPLAY_RESPONSE_WAIT_SECONDS instead. It remains honored as a fallback for "
                "whichever of the two new bounds is left unset."
            )
        return self

    @property
    def effective_upload_stall_seconds(self) -> float:
        if self.upload_stall_seconds is not None:
            return self.upload_stall_seconds
        if self.request_timeout_seconds is not None:
            return self.request_timeout_seconds
        return DEFAULT_UPLOAD_STALL_SECONDS

    @property
    def effective_response_wait_seconds(self) -> float:
        if self.response_wait_seconds is not None:
            return self.response_wait_seconds
        if self.request_timeout_seconds is not None:
            return self.request_timeout_seconds
        return DEFAULT_RESPONSE_WAIT_SECONDS

    @model_validator(mode="after")
    def _auto_requires_cloud_target(self) -> "Settings":
        if self.deployment_profile is DeploymentProfile.AUTO and not self.cloud_inference_base_url:
            raise ValueError(
                "REPLAY_DEPLOYMENT_PROFILE=auto requires REPLAY_CLOUD_INFERENCE_BASE_URL to be set "
                "(auto mode's dual target -- fail-closed configuration)"
            )
        return self

    @property
    def ingestion_mode(self) -> IngestionMode:
        if self.deployment_profile not in _MODE_BY_PROFILE:
            raise ValueError(
                f"No ingestion mode for deployment_profile={self.deployment_profile.value!r} -- "
                "AUTO must be resolved to an effective profile before ingestion_mode is read"
            )
        return _MODE_BY_PROFILE[self.deployment_profile]

    @property
    def effective_cloud_inference_api_key(self) -> str:
        return self.cloud_inference_api_key or self.inference_api_key

    @property
    def cloud_auth_headers(self) -> dict[str, str]:
        key = self.effective_cloud_inference_api_key
        return {"X-API-Key": key} if key else {}

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.inference_api_key} if self.inference_api_key else {}

    def session_url(self) -> str:
        return f"{self.inference_base_url}{self.inference_session_endpoint}"

    def path_url(self) -> str:
        return f"{self.inference_base_url}{self.inference_path_endpoint}"

    def push_url(self, session_id: str) -> str:
        path = self.inference_push_endpoint_template.replace("{session_id}", session_id)
        return f"{self.inference_base_url}{path}"

    def visible_path_for(self, local_session_dir: Path) -> Path:
        """
        Translate a local session-folder path into the path the inference service should
        register/resolve, for path-registration mode (identity unless
        REPLAY_INFERENCE_VISIBLE_DATA_ROOT is configured — see that field).
        """
        if self.inference_visible_data_root is None:
            return local_session_dir
        rel = local_session_dir.resolve().relative_to(self.data_root.resolve())
        return self.inference_visible_data_root / rel

    def describe(self) -> str:
        try:
            ingestion_mode_desc = self.ingestion_mode.value
        except ValueError:
            ingestion_mode_desc = "<unresolved -- auto>"
        return "\n".join(
            [
                f"  deployment_profile: {self.deployment_profile.value}",
                f"  ingestion_mode    : {ingestion_mode_desc}",
                f"  data_root         : {self.data_root}",
                f"  inference_base_url: {self.inference_base_url}",
                f"  api_key configured: {'yes' if self.inference_api_key else 'no'}",
                f"  auto_push         : {self.auto_push}",
                f"  default location  : {self.default_latitude}, {self.default_longitude}",
                f"  parcel_id         : {self.parcel_id}",
            ]
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance (cached after first call)."""
    return Settings()
