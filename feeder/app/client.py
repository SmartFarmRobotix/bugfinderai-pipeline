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
Session feeder — HTTP client module.

Submits a replay session to the inference service using one of two ingestion modes, selected
once from deployment configuration and never
negotiated per request:

  ``http_upload``: POST image bytes as multipart/form-data to
    ``/api/v1/sessions`` (cloud, x86 CI-smoke).
  ``path``: POST a JSON path-registration request to
    ``/api/v1/sessions/path`` — zero-copy, valid only where this component
    and the inference service share a filesystem (mixed, edge).

Per-image position/provenance passthrough: a present manifest
position is forwarded, tagged provenance ``replay``, via the ``metadata``
field both of the inference service's intake endpoints already accept as an arbitrary JSON object. An absent position is never
invented — the key is simply omitted for that image.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app import accounting
from app.config import IngestionMode, Settings
from app.store import ReplaySession

logger = logging.getLogger("feeder.client")


@dataclass
class SessionSubmitResult:
    """Parsed response from either session-creation endpoint."""

    session_id: str = ""
    best_image_id: str = ""
    infestation_score: float = 0.0
    status_code: int = 0
    raw_body: str = ""
    ingestion_mode: str = ""
    success: bool = False
    error: str = ""


@dataclass
class PushResult:
    """Parsed response from ``POST /api/v1/sessions/{session_id}/push``."""

    pushed: bool = False
    target_url: str = ""
    status_code: int = 0
    response_body: str = ""
    success: bool = False
    error: str = ""
    report_id: str = ""
    pdm_session_id: str = ""


def _pest_fields(session: ReplaySession) -> dict[str, Any]:
    """`pest` next to `crop`, only when the manifest carried one -- a
    session without a pest submits exactly as it did before the field."""
    return {"pest": session.pest} if session.pest is not None else {}


def _growth_stage_fields(session: ReplaySession) -> dict[str, Any]:
    gs = session.growth_stage or {}
    return {
        "growth_stage_representation": gs.get("representation"),
        "growth_stage_bbch_min": gs.get("bbch_min"),
        "growth_stage_bbch_max": gs.get("bbch_max"),
        "growth_stage_stage_id": gs.get("stage_id"),
    }


def _per_image_positions_metadata(session: ReplaySession) -> Optional[dict[str, Any]]:
    """
    Build the ``metadata.per_image_positions`` passthrough carrier: one entry per image that has a manifest-supplied position,
    forwarded unchanged and always tagged provenance ``replay`` from this
    component's own sourcing perspective (independent of whatever
    provenance value the manifest itself recorded). Returns
    ``None`` (omit the key entirely) when no image in the session has a
    position — never a fabricated/default value.
    """
    positions: dict[str, Any] = {}
    for img in session.images:
        if img.position is not None:
            positions[img.filename] = {
                "latitude": img.position.latitude,
                "longitude": img.position.longitude,
                "provenance": "replay",
            }
    if not positions:
        return None
    return {"per_image_positions": positions}


def _session_coordinate_fields(session: ReplaySession, settings: Settings) -> dict[str, float]:
    """Prefer deployment overrides, then replayed session metadata, else omit."""
    latitude = settings.default_latitude if settings.default_latitude is not None else session.latitude
    longitude = settings.default_longitude if settings.default_longitude is not None else session.longitude
    if latitude is None or longitude is None:
        return {}
    return {"latitude": latitude, "longitude": longitude}


class SessionFeederClient:
    """Synchronous HTTP client that talks to the inference service."""

    def __init__(self, settings: Settings, transport: Optional[httpx.BaseTransport] = None) -> None:
        self._settings = settings
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.effective_response_wait_seconds),
                write=float(settings.effective_upload_stall_seconds),
                pool=10.0,
            ),
            verify=settings.tls_verify,
            follow_redirects=True,
            transport=transport,
        )
        self._headers: dict[str, str] = dict(settings.auth_headers)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SessionFeederClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def healthcheck(self) -> bool:
        url = f"{self._settings.inference_base_url}/healthz"
        try:
            resp = self._http.get(url, headers=self._headers)
            if resp.status_code == 200:
                return resp.json().get("status") == "ready"
            return False
        except httpx.HTTPError as exc:
            logger.error("inference service healthcheck failed: %s", exc)
            return False

    def wait_for_inference_service(self, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
        for attempt in range(1, max_retries + 1):
            if self.healthcheck():
                return True
            if attempt < max_retries:
                time.sleep(retry_delay)
        return False

    def submit_session(self, session: ReplaySession) -> SessionSubmitResult:
        """
        Submit *session* using exactly the ingestion mode this feeder is
        configured for — never chosen at request time.
        """
        mode = self._settings.ingestion_mode
        if mode is IngestionMode.HTTP_UPLOAD:
            return self._upload_session(session)
        return self._register_path_session(session)

    def _upload_session(self, session: ReplaySession) -> SessionSubmitResult:
        result = SessionSubmitResult(ingestion_mode=IngestionMode.HTTP_UPLOAD.value)

        if not session.images:
            result.error = "No images to upload"
            logger.error(result.error)
            return result

        file_parts: list[tuple[str, tuple[str, bytes, str]]] = []
        for img in session.images:
            try:
                img_bytes = img.path.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s — skipping", img.path, exc)
                continue
            content_type = "image/png" if img.path.suffix.lower() == ".png" else "image/jpeg"
            file_parts.append(("files", (img.filename, img_bytes, content_type)))

        if not file_parts:
            result.error = "All image files unreadable"
            logger.error(result.error)
            return result

        metadata = _per_image_positions_metadata(session)
        form_data: dict[str, Any] = {
            "session_id": session.session_id,
            "time_start": session.session_start,
            "time_end": session.session_end,
            "crop": session.crop,
        }
        form_data.update(_pest_fields(session))
        form_data.update(
            {key: str(value) for key, value in _session_coordinate_fields(session, self._settings).items()}
        )
        if self._settings.parcel_id:
            form_data["parcel_id"] = self._settings.parcel_id
        for key, value in _growth_stage_fields(session).items():
            if value is not None:
                form_data[key] = value
        if metadata is not None:
            form_data["metadata"] = json.dumps(metadata)

        accountant = accounting.ByteAccountant(self._settings, session.session_id)
        request_bytes = accounting.multipart_bytes(form_data, file_parts)

        try:
            resp = self._http.post(
                self._settings.session_url(), headers=self._headers, data=form_data, files=file_parts
            )
        except httpx.WriteTimeout as exc:
            accountant.record(endpoint=self._settings.session_url(), direction="up", kind="session_upload", byte_count=request_bytes)
            accountant.finalize()
            result.error = (
                f"upload-stalled: no upload progress within "
                f"{self._settings.effective_upload_stall_seconds}s ({exc})"
            )
            logger.error(result.error)
            return result
        except httpx.ReadTimeout as exc:
            accountant.record(endpoint=self._settings.session_url(), direction="up", kind="session_upload", byte_count=request_bytes)
            accountant.finalize()
            result.error = (
                f"response-timeout: no response within "
                f"{self._settings.effective_response_wait_seconds}s ({exc})"
            )
            logger.error(result.error)
            return result
        except httpx.HTTPError as exc:
            accountant.record(endpoint=self._settings.session_url(), direction="up", kind="session_upload", byte_count=request_bytes)
            accountant.finalize()
            result.error = f"HTTP request failed: {exc}"
            logger.error(result.error)
            return result

        accountant.record(endpoint=self._settings.session_url(), direction="up", kind="session_upload", byte_count=request_bytes)
        accountant.record(
            endpoint=self._settings.session_url(), direction="down", kind="session_upload", byte_count=len(resp.content)
        )
        accountant.finalize()

        return self._parse_session_response(result, resp)

    def _register_path_session(self, session: ReplaySession) -> SessionSubmitResult:
        result = SessionSubmitResult(ingestion_mode=IngestionMode.PATH.value)

        if not session.images:
            result.error = "No images to register"
            logger.error(result.error)
            return result

        visible_path = self._settings.visible_path_for(session.session_dir)
        body: dict[str, Any] = {
            "session_id": session.session_id,
            "path": str(visible_path),
            "parcel_id": self._settings.parcel_id,
            "time_start": session.session_start,
            "time_end": session.session_end,
            "crop": session.crop,
            **_pest_fields(session),
            **_growth_stage_fields(session),
        }
        body.update(_session_coordinate_fields(session, self._settings))
        metadata = _per_image_positions_metadata(session)
        if metadata is not None:
            body["metadata"] = metadata

        accountant = accounting.ByteAccountant(self._settings, session.session_id)
        request_bytes = accounting.bytes_of(body)

        try:
            resp = self._http.post(self._settings.path_url(), headers=self._headers, json=body)
        except httpx.HTTPError as exc:
            accountant.record(endpoint=self._settings.path_url(), direction="up", kind="session_registration_local", byte_count=request_bytes)
            accountant.finalize()
            result.error = f"HTTP request failed: {exc}"
            logger.error(result.error)
            return result

        accountant.record(endpoint=self._settings.path_url(), direction="up", kind="session_registration_local", byte_count=request_bytes)
        accountant.record(
            endpoint=self._settings.path_url(), direction="down", kind="session_registration_local", byte_count=len(resp.content)
        )
        accountant.finalize()

        return self._parse_session_response(result, resp)

    @staticmethod
    def _parse_session_response(result: SessionSubmitResult, resp: httpx.Response) -> SessionSubmitResult:
        result.status_code = resp.status_code
        result.raw_body = resp.text

        if resp.status_code >= 400:
            result.error = f"the inference service returned HTTP {resp.status_code}: {resp.text[:500]}"
            logger.error(result.error)
            return result

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            result.error = f"Invalid JSON response: {exc}"
            logger.error(result.error)
            return result

        result.session_id = data.get("session_id", "")
        result.best_image_id = data.get("best_image_id", "")
        result.infestation_score = float(data.get("infestation_score", 0.0))
        result.success = True
        return result

    def push_session(self, session_id: str) -> PushResult:
        result = PushResult()

        if not session_id:
            result.error = "No session_id provided"
            logger.error(result.error)
            return result

        url = self._settings.push_url(session_id)
        accountant = accounting.ByteAccountant(self._settings, session_id)
        try:
            resp = self._http.post(url, headers=self._headers)
        except httpx.HTTPError as exc:
            accountant.record(endpoint=url, direction="up", kind="push_trigger", byte_count=0)
            accountant.finalize()
            result.error = f"HTTP request failed: {exc}"
            logger.error(result.error)
            return result

        accountant.record(endpoint=url, direction="up", kind="push_trigger", byte_count=0)
        accountant.record(endpoint=url, direction="down", kind="push_trigger", byte_count=len(resp.content))
        accountant.finalize()

        result.status_code = resp.status_code
        if resp.status_code >= 400:
            result.error = f"Push returned HTTP {resp.status_code}: {resp.text[:500]}"
            logger.error(result.error)
            return result

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            result.error = f"Invalid JSON response: {exc}"
            logger.error(result.error)
            return result

        result.pushed = data.get("pushed", False)
        result.target_url = data.get("target_url", "")
        result.response_body = data.get("response_body", "")
        result.success = True

        if result.response_body:
            try:
                upstream = json.loads(result.response_body)
                result.report_id = upstream.get("report_id") or ""
                result.pdm_session_id = upstream.get("session_id") or ""
            except (json.JSONDecodeError, ValueError):
                pass

        return result
