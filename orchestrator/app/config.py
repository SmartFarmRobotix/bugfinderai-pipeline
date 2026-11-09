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

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fusion import FUSION_RULE_VERSION as _DEFAULT_FUSION_RULE_VERSION
from fusion import UnknownFusionRuleError, select_fusion_rule

PLACEHOLDER_SECRET_VALUES = {
    "change-me",
    "some-key",
    "dev_jwt_signing_key",
    "smoketest-jwt-key-change-in-production",
}


class DeploymentProfile(str, Enum):
    """Which auth mechanism the intake endpoint uses: cloud/mixed front the intake with Gatekeeper-validated
    tokens; edge authenticates with a local shared secret and makes no
    Gatekeeper calls at all."""

    CLOUD = "cloud"
    MIXED = "mixed"
    EDGE = "edge"


class Settings(BaseSettings):
    app_name: str = Field(default="SessionOrchestrator", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="ORCH_SERVICE_HOST")
    port: int = Field(default=8020, alias="ORCH_SERVICE_PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    service_name: str = Field(default="orchestrator", alias="ORCH_SERVICE_NAME")

    database_url: str = Field(
        default="sqlite:////code/app/data/orchestrator.db",
        alias="ORCH_DATABASE_URL",
    )

    max_outbound_retries: int = Field(default=3, ge=1, alias="ORCH_MAX_OUTBOUND_RETRIES")

    deployment_profile: DeploymentProfile = Field(
        default=DeploymentProfile.CLOUD, alias="ORCH_DEPLOYMENT_PROFILE"
    )

    gatekeeper_base_url: str = Field(default="http://gatekeeper:8001", alias="ORCH_GATEKEEPER_BASE_URL")
    gatekeeper_username: Optional[str] = Field(default=None, alias="ORCH_GATEKEEPER_USERNAME")
    gatekeeper_password: Optional[str] = Field(default=None, alias="ORCH_GATEKEEPER_PASSWORD")
    gatekeeper_token_endpoint: str = Field(
        default="/api/login/", alias="ORCH_GATEKEEPER_TOKEN_ENDPOINT"
    )
    gatekeeper_timeout_seconds: int = Field(
        default=10, alias="ORCH_GATEKEEPER_TIMEOUT_SECONDS"
    )
    gatekeeper_validate_path: str = Field(
        default="/api/validate_token/", alias="ORCH_GATEKEEPER_VALIDATE_PATH"
    )

    edge_shared_secret: Optional[str] = Field(default=None, alias="ORCH_EDGE_SHARED_SECRET")

    edge_report_dir: str = Field(default="/code/app/data/reports", alias="ORCH_EDGE_REPORT_DIR")
    edge_inference_base_url: str = Field(
        default="http://inference:8010", alias="ORCH_EDGE_INFERENCE_BASE_URL"
    )
    edge_inference_api_key: Optional[str] = Field(default=None, alias="ORCH_EDGE_INFERENCE_API_KEY")
    edge_inference_call_timeout_seconds: int = Field(
        default=15, alias="ORCH_EDGE_INFERENCE_CALL_TIMEOUT_SECONDS"
    )

    sync_queue_dir: str = Field(default="/code/app/sync_queue", alias="ORCH_SYNC_QUEUE_DIR")

    sync_gatekeeper_username: Optional[str] = Field(default=None, alias="ORCH_SYNC_GATEKEEPER_USERNAME")
    sync_gatekeeper_password: Optional[str] = Field(default=None, alias="ORCH_SYNC_GATEKEEPER_PASSWORD")

    rs_edge_report_ingest_path: str = Field(
        default="/api/v1/openagri-report/pest-session-report/edge-sync/",
        alias="ORCH_RS_EDGE_REPORT_INGEST_PATH",
    )
    sync_call_timeout_seconds: int = Field(default=30, alias="ORCH_SYNC_CALL_TIMEOUT_SECONDS")

    sync_max_transient_failures: int = Field(default=5, alias="ORCH_SYNC_MAX_TRANSIENT_FAILURES", ge=1)

    sync_reassess_enabled: bool = Field(default=True, alias="ORCH_SYNC_REASSESS_ENABLED")
    sync_cloud_orchestrator_service_name: str = Field(
        default="orchestrator", alias="ORCH_SYNC_CLOUD_ORCHESTRATOR_SERVICE_NAME"
    )
    rs_images_upload_path: str = Field(
        default="/api/v1/openagri-report/pest-session-images/", alias="ORCH_RS_IMAGES_UPLOAD_PATH"
    )
    sync_reassess_timeout_seconds: int = Field(default=120, alias="ORCH_SYNC_REASSESS_TIMEOUT_SECONDS", ge=1)

    edge_disk_budget_pct: float = Field(default=80.0, alias="EDGE_DISK_BUDGET_PCT")
    edge_disk_headroom_min_pct: float = Field(default=10.0, alias="EDGE_DISK_HEADROOM_MIN_PCT")

    edge_housekeeping_interval_sec: float = Field(default=300.0, alias="EDGE_HOUSEKEEPING_INTERVAL_SEC")

    @model_validator(mode="after")
    def _validate_profile_credentials(self) -> "Settings":
        if self.deployment_profile is DeploymentProfile.EDGE:
            if not self.edge_shared_secret:
                raise ValueError(
                    "ORCH_EDGE_SHARED_SECRET must be set when ORCH_DEPLOYMENT_PROFILE=edge"
                )
            if self.edge_shared_secret.strip().lower() in PLACEHOLDER_SECRET_VALUES:
                raise ValueError(
                    f"ORCH_EDGE_SHARED_SECRET is set to a known placeholder secret "
                    f"({self.edge_shared_secret!r}); refusing to start. Set it to a "
                    "real, non-placeholder value."
                )
        else:
            if not self.gatekeeper_username or not self.gatekeeper_password:
                raise ValueError(
                    "ORCH_GATEKEEPER_USERNAME/ORCH_GATEKEEPER_PASSWORD must both be "
                    "set when ORCH_DEPLOYMENT_PROFILE is cloud or mixed"
                )
        return self

    pdm_service_name: str = Field(default="pdm", alias="ORCH_PDM_SERVICE_NAME")
    pdm_risk_assessment_path: str = Field(
        default="/api/v1/risk-assessment/", alias="ORCH_PDM_RISK_ASSESSMENT_PATH"
    )
    pdm_call_timeout_seconds: int = Field(
        default=15, alias="ORCH_PDM_CALL_TIMEOUT_SECONDS"
    )
    pdm_forecast_risk_path: str = Field(
        default="/api/v1/risk-assessment/forecast/", alias="ORCH_PDM_FORECAST_RISK_PATH"
    )

    ws_service_name: str = Field(default="weather_data", alias="ORCH_WS_SERVICE_NAME")
    ws_history_hourly_path: str = Field(
        default="/api/v1/history/hourly/", alias="ORCH_WS_HISTORY_HOURLY_PATH"
    )
    ws_history_radius_km: float = Field(default=10.0, alias="ORCH_WS_HISTORY_RADIUS_KM")
    ws_forecast_daily_path: str = Field(
        default="/api/v1/forecast/daily/", alias="ORCH_WS_FORECAST_DAILY_PATH"
    )
    ws_history_daily_path: str = Field(
        default="/api/v1/history/daily/", alias="ORCH_WS_HISTORY_DAILY_PATH"
    )
    ws_forecast_hourly_path: str = Field(
        default="/api/v1/forecast/hourly/", alias="ORCH_WS_FORECAST_HOURLY_PATH"
    )
    ws_call_timeout_seconds: int = Field(
        default=15, alias="ORCH_WS_CALL_TIMEOUT_SECONDS"
    )

    rs_service_name: str = Field(default="reporting", alias="ORCH_RS_SERVICE_NAME")
    rs_pest_session_report_path: str = Field(
        default="/api/v1/openagri-report/pest-session-report/",
        alias="ORCH_RS_PEST_SESSION_REPORT_PATH",
    )
    rs_report_status_path: str = Field(
        default="/api/v1/openagri-report/pest-session-report/by-session/{session_id}/status",
        alias="ORCH_RS_REPORT_STATUS_PATH",
    )
    rs_call_timeout_seconds: int = Field(
        default=15, alias="ORCH_RS_CALL_TIMEOUT_SECONDS"
    )

    fusion_rule_version: str = Field(
        default=_DEFAULT_FUSION_RULE_VERSION, alias="ORCH_FUSION_RULE_VERSION"
    )

    report_site_label: str = Field(
        default="Field site", alias="ORCH_REPORT_SITE_LABEL"
    )

    @field_validator("fusion_rule_version")
    @classmethod
    def _validate_fusion_rule_version(cls, value: str) -> str:
        try:
            select_fusion_rule(value)
        except UnknownFusionRuleError as exc:
            raise ValueError(str(exc)) from exc
        return value

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
