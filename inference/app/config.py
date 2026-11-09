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

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="InferenceService", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="INFERENCE_SERVICE_HOST")
    port: int = Field(default=8010, alias="INFERENCE_SERVICE_PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    storage_dir: Path = Field(default=Path("/code/app/data"), alias="INFERENCE_STORAGE_DIR")

    data_root: Path = Field(default=Path("/code/app/data"), alias="INFERENCE_DATA_ROOT")

    deterministic_seed: int = Field(default=42, alias="INFERENCE_DETERMINISTIC_SEED")

    detector_impl: str = Field(default="lookup", alias="INFERENCE_DETECTOR_IMPL")

    acceptance_smoke_mode: Literal["acceptance", "ci-dev"] = Field(
        default="acceptance", alias="INFERENCE_ACCEPTANCE_SMOKE_MODE"
    )


    engine_path: Path = Field(default=Path("/code/app/data/engine/model.engine"), alias="INFERENCE_ENGINE_PATH")
    engine_provenance_path: Optional[Path] = Field(default=None, alias="INFERENCE_ENGINE_PROVENANCE_PATH")
    execution_target: Literal["cloud", "jetson"] = Field(default="cloud", alias="INFERENCE_EXECUTION_TARGET")

    onnx_model_path: Path = Field(default=Path("/code/app/data/onnx/model.onnx"), alias="INFERENCE_ONNX_MODEL_PATH")
    onnx_provenance_path: Optional[Path] = Field(default=None, alias="INFERENCE_ONNX_PROVENANCE_PATH")
    onnx_execution_provider: Literal["CPUExecutionProvider", "CUDAExecutionProvider"] = Field(
        default="CPUExecutionProvider", alias="INFERENCE_ONNX_EXECUTION_PROVIDER"
    )

    tile_batch: int = Field(default=4, ge=1, alias="INFERENCE_TILE_BATCH")
    region_threshold: float = Field(default=0.40, ge=0.0, le=1.0, alias="INFERENCE_REGION_THRESHOLD")

    detector_profile: Literal["legacy-full-frame", "edge-s-fp16", "cloud-l-fp32", "cloud-l-fp32-tta", "edge-s-int8"] = Field(
        default="legacy-full-frame", alias="INFERENCE_DETECTOR_PROFILE"
    )
    negligibility_record_dir: Path = Field(
        default=Path("/code/app/data/negligibility"), alias="INFERENCE_NEGLIGIBILITY_RECORD_DIR"
    )

    target_class_set: List[str] = Field(
        default_factory=lambda: ["brassica_seedling"],
        alias="INFERENCE_TARGET_CLASS_SET",
    )
    confidence_floor: float = Field(default=0.3, ge=0.0, le=1.0, alias="INFERENCE_CONFIDENCE_FLOOR")

    default_crop: str = Field(default="brassica-oleracea", alias="INFERENCE_DEFAULT_CROP")
    default_growth_stage_representation: Literal["bbch_range", "enum"] = Field(
        default="enum", alias="INFERENCE_DEFAULT_GROWTH_STAGE_REPRESENTATION"
    )

    external_api_key: str = Field(
        default="",
        alias="INFERENCE_EXTERNAL_API_KEY",
        description=(
            "Pre-shared API key for external clients (e.g. the session feeder). "
            "Sent as X-API-Key header. Empty = disabled."
        ),
    )

    probe_sink_max_bytes: int = Field(default=1_048_576, alias="INFERENCE_PROBE_SINK_MAX_BYTES")

    using_gatekeeper: bool = Field(default=False, alias="INFERENCE_USING_GATEKEEPER")
    gatekeeper_base_url: str = Field(default="http://gatekeeper:8001", alias="INFERENCE_GATEKEEPER_BASE_URL")
    gatekeeper_username: str = Field(default="", alias="INFERENCE_GATEKEEPER_USERNAME")
    gatekeeper_password: str = Field(default="", alias="INFERENCE_GATEKEEPER_PASSWORD")
    gatekeeper_token_endpoint: str = Field(
        default="/api/login/", alias="INFERENCE_GATEKEEPER_TOKEN_ENDPOINT"
    )
    gatekeeper_register_endpoint: str = Field(
        default="/api/register_service/", alias="INFERENCE_GATEKEEPER_REGISTER_ENDPOINT"
    )
    gatekeeper_timeout_seconds: int = Field(default=10, alias="INFERENCE_GATEKEEPER_TIMEOUT_SECONDS")
    service_name: str = Field(default="inference", alias="INFERENCE_SERVICE_NAME")

    self_base_url: Optional[str] = Field(default=None, alias="INFERENCE_SELF_BASE_URL")

    orch_push_enabled: bool = Field(default=False, alias="INFERENCE_ORCH_PUSH_ENABLED")
    orch_push_path: str = Field(
        default="/api/v1/sessions/",
        alias="INFERENCE_ORCH_PUSH_PATH",
        description="Relative orchestrator path to receive session submissions",
    )
    orch_edge_shared_secret: Optional[str] = Field(default=None, alias="INFERENCE_ORCH_EDGE_SHARED_SECRET")

    best_image_push_enabled: bool = Field(default=False, alias="INFERENCE_BEST_IMAGE_PUSH_ENABLED")
    rs_service_name: str = Field(default="reporting", alias="INFERENCE_RS_SERVICE_NAME")
    rs_upload_path: str = Field(
        default="/api/v1/openagri-report/pest-session-images/",
        alias="INFERENCE_RS_UPLOAD_PATH",
    )
    best_image_push_max_retries: int = Field(default=3, alias="BEST_IMAGE_PUSH_MAX_RETRIES")
    best_image_push_backoff_base_sec: float = Field(default=30, alias="BEST_IMAGE_PUSH_BACKOFF_BASE_SEC")
    best_image_push_deadline_sec: float = Field(default=600, alias="BEST_IMAGE_PUSH_DEADLINE_SEC")
    best_image_push_continuation_interval_sec: float = Field(
        default=300, alias="BEST_IMAGE_PUSH_CONTINUATION_INTERVAL_SEC"
    )
    best_image_max_bytes_per_image: int = Field(default=100_000, alias="BEST_IMAGE_MAX_BYTES_PER_IMAGE")
    best_image_max_bytes_combined: int = Field(default=200_000, alias="BEST_IMAGE_MAX_BYTES_COMBINED")

    scenario: str = Field(default="cloud", alias="INFERENCE_SCENARIO")
    run_label: Optional[str] = Field(default=None, alias="INFERENCE_RUN_LABEL")
    byte_accounting_tolerance_pct: float = Field(default=15.0, alias="BYTE_ACCOUNTING_TOLERANCE_PCT")

    sync_queue_dir: Path = Field(default=Path("/code/app/sync_queue"), alias="INFERENCE_SYNC_QUEUE_DIR")

    edge_disk_budget_pct: float = Field(default=80.0, alias="EDGE_DISK_BUDGET_PCT")
    edge_disk_headroom_min_pct: float = Field(default=10.0, alias="EDGE_DISK_HEADROOM_MIN_PCT")

    edge_housekeeping_interval_sec: float = Field(default=300.0, alias="EDGE_HOUSEKEEPING_INTERVAL_SEC")

    health_proof_max_age_seconds: float = Field(default=60.0, alias="INFERENCE_HEALTH_PROOF_MAX_AGE_SECONDS")
    health_probe_timeout_seconds: float = Field(default=20.0, alias="INFERENCE_HEALTH_PROBE_TIMEOUT_SECONDS")
    health_exit_after_failures: int = Field(default=3, alias="INFERENCE_HEALTH_EXIT_AFTER_FAILURES", ge=0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
