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

import asyncio
import base64
import concurrent.futures
import contextlib
import functools
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional

from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("inference.auth")

from app import accounting
from app.config import Settings, get_settings
from app.contract import (
    CONTRACT_VERSION,
    EPPO_CODE_PATTERN,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionResults,
    SessionSubmission,
)
from app.detectors.acceptance_smoke import AcceptanceSmokeRefusal, run_acceptance_smoke
from app.detectors.base import Detector, ImageRef
from app.detectors.profiles import LEGACY_FULL_FRAME
from app.detectors.registry import get_detector
from app.detectors.tiled_path import frame_size_of, tiled_path_fields
from app.detectors.tiling import TilingRefusal
from app.health import HealthState
from app.image_compress import compress_pair_to_cap
from app.schemas import SessionCreateResponse
from app.scoring import (
    AGGREGATION_RULE_VERSION,
    ScoredImage,
    aggregate_session_score,
    per_image_score,
    select_best_image,
)
from app.storage import (
    PathEscapesDataRootError,
    PathUnreadableError,
    SessionManifestError,
    create_session_id,
    generate_image_id,
    is_supported_image_format,
    list_manifest_images,
    read_session_metadata,
    read_session_results,
    resolve_contained_path,
    save_upload_file,
    session_dir,
    write_session_metadata,
    write_session_results,
)
from app.connection_audit import install as _install_connection_audit
from app.disk_lifecycle import (
    DiskAdmissionRefused,
    HousekeepingResult,
    check_preflight,
    resume_leftover_tombstones,
    run_housekeeping,
)

_install_connection_audit(get_settings().service_name)

resume_leftover_tombstones(get_settings())

_ANNOTATED_IMAGES_DIR = Path(__file__).parent / "annotated_images"
_PER_IMAGE_POSITION_PROVENANCES = frozenset({"gps", "odometry", "replay", "default"})

_DETECTOR_CACHE: Dict[str, Detector] = {}

DETECTOR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")


def _get_detector(impl_name: str) -> Detector:
    if impl_name not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[impl_name] = get_detector(impl_name)
    return _DETECTOR_CACHE[impl_name]


HEALTH_STATE = HealthState()

_EXIT_DELAY_SECONDS = 0.5
_EXIT_STATUS = 3


def _run_tracked(fn: Callable[..., Any], *args: Any) -> Any:
    """Run one detector call with the in-flight count held; runs on the
    executor's thread so a call still queued behind another does not count."""
    HEALTH_STATE.enter()
    try:
        return fn(*args)
    finally:
        HEALTH_STATE.leave()


def _detect_recorded(detector: Detector, image_ref: ImageRef) -> Any:
    result = detector.detect(image_ref)
    HEALTH_STATE.record_detection_ok()
    return result


def _probe_detector(impl_name: str) -> None:
    """The probe body, run on DETECTOR_EXECUTOR through ``_run_tracked`` like
    a detection, so a probe in progress counts as in flight and a concurrent
    /healthz does not queue a second one: the TensorRT detector's runtime
    belongs to that one thread, so the probe never runs anywhere else.
    Building the detector here also turns a construction failure into a
    failed probe."""
    _get_detector(impl_name).probe()


class SessionMetadataResponse(BaseModel):
    session_id: str
    best_image_id: str
    infestation_score: float
    images: list[dict[str, Any]]
    crop: Optional[str] = None
    pest: Optional[str] = None
    growth_stage: Optional[dict[str, Any]] = None
    provenance: Optional[dict[str, Any]] = None
    contract_version: Optional[str] = None
    ingestion_mode: Optional[str] = None
    parcel_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    position_provenance: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    images_pending: Optional[bool] = None
    best_image_push_attempts: Optional[int] = None
    best_image_artifact_hash: Optional[str] = None


class PushResponse(BaseModel):
    pushed: bool
    target_url: Optional[str] = None
    status_code: Optional[int] = None
    response_body: Optional[str] = None


class PathRegistrationRequest(BaseModel):
    """Body for the zero-copy path-registration intake mode."""

    session_id: Optional[str] = None
    path: str
    parcel_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    crop: Optional[str] = None
    pest: Optional[str] = Field(default=None, pattern=EPPO_CODE_PATTERN)
    growth_stage_representation: Optional[Literal["bbch_range", "enum"]] = None
    growth_stage_bbch_min: Optional[int] = None
    growth_stage_bbch_max: Optional[int] = None
    growth_stage_stage_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


_REGISTRATION_BACKOFF_INITIAL_SECONDS = 5
_REGISTRATION_BACKOFF_MAX_SECONDS = 300


async def _register_service_with_retry(settings: Settings) -> None:
    """Retry GateKeeper registration with bounded exponential backoff.

    Runs as a background task so a GateKeeper that is unreachable or slow to
    boot (it loses the startup race against GateKeeper's Django boot in
    practice) never prevents this service from starting or serving /healthz.
    Retries are unbounded in count; only the backoff interval is bounded.
    """
    delay = _REGISTRATION_BACKOFF_INITIAL_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            await asyncio.to_thread(_register_service, settings)
            logger.info("GateKeeper registration succeeded on attempt %d", attempt)
            return
        except Exception as exc:  # noqa: BLE001 - any failure just means "retry later"
            logger.warning(
                "GateKeeper registration attempt %d failed, retrying in %ds: %s",
                attempt, delay, exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _REGISTRATION_BACKOFF_MAX_SECONDS)


def _run_off_loop(func, *args):
    """Runs a blocking `func(*args)` on a fresh daemon thread and returns
    an awaitable for its result. Deliberately not `asyncio.to_thread`:
    that schedules onto the event loop's *default* executor, which
    `asyncio.run`'s (and therefore uvicorn's) shutdown sequence joins
    before the process can exit -- so a still-in-flight purge would
    still delay shutdown for its full remaining duration even though the
    loop itself is free to keep servicing requests meanwhile. A daemon
    thread is never joined by either path: cancelling the task awaiting
    this future cancels it immediately (a plain `asyncio.Future`, unlike
    `to_thread`'s wrapped executor future, transitions to cancelled
    regardless of whether the underlying work has finished), and an
    abandoned daemon thread does not block interpreter exit either."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def _target() -> None:
        try:
            result = func(*args)
        except BaseException as exc:  # noqa: BLE001 - relayed to the awaiting coroutine
            callback, callback_arg = _try_set_exception, exc
        else:
            callback, callback_arg = _try_set_result, result
        try:
            loop.call_soon_threadsafe(callback, future, callback_arg)
        except RuntimeError:
            pass

    threading.Thread(target=_target, daemon=True, name="housekeeping-purge").start()
    return future


def _try_set_result(future: asyncio.Future, result: Any) -> None:
    if not future.done():
        future.set_result(result)


def _try_set_exception(future: asyncio.Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


async def _periodic_housekeeping_loop() -> None:
    """Periodic housekeeping: an interval-scheduled
    in-process worker that runs `run_housekeeping` on a fixed cadence
    (`EDGE_HOUSEKEEPING_INTERVAL_SEC`) independent of session activity
    entirely. `_kick_housekeeping` (below) only fires as a side effect of
    an admitted session start, so it never runs at all during a lull and
    misses a budget crossing caused by the session that just finished;
    this loop is what actually keeps the pass active through the
    80-90%-used band with zero session activity. The
    session-start kick stays as an additional fast-reaction trigger, not a
    substitute for this loop. Runs for the lifetime of the process;
    cancelled cleanly on shutdown (`lifespan`, below). A single failed
    pass is logged and never kills the loop -- this loop has no other
    caller to observe or retry a failure, so it must swallow broad
    exceptions itself to keep running indefinitely."""
    settings = get_settings()
    while True:
        await asyncio.sleep(settings.edge_housekeeping_interval_sec)
        try:
            result: HousekeepingResult = await _run_off_loop(run_housekeeping, get_settings())
            if result.budget_triggered:
                logger.info(
                    "Periodic housekeeping pass: purged=%d resumed_tombstones=%d errors=%d",
                    len(result.purged_sessions), len(result.resumed_tombstones), len(result.errors),
                )
        except Exception:  # noqa: BLE001 - the periodic loop must never die on one bad pass
            logger.exception("Periodic housekeeping pass failed")


_PERIODIC_TASK_SHUTDOWN_TIMEOUT_SECONDS = 2.0


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Register with GateKeeper on startup (replaces deprecated @app.on_event)."""
    settings = get_settings()
    if settings.detector_profile != LEGACY_FULL_FRAME:
        _get_detector(settings.detector_impl)
    if settings.using_gatekeeper:
        task = asyncio.create_task(_register_service_with_retry(settings))
        application.state.gatekeeper_registration_task = task
    periodic_task = asyncio.create_task(_periodic_housekeeping_loop())
    try:
        yield
    finally:
        periodic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            try:
                await asyncio.wait_for(periodic_task, timeout=_PERIODIC_TASK_SHUTDOWN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    "periodic housekeeping task still running %.1fs after cancellation "
                    "(a purge is likely still finishing on its worker thread) -- "
                    "proceeding with shutdown without waiting further",
                    _PERIODIC_TASK_SHUTDOWN_TIMEOUT_SECONDS,
                )


app = FastAPI(
    title="InferenceService",
    version="0.2.0",
    description="Session-level inference service (versioned contract, pluggable detector).",
    lifespan=lifespan,
)

@app.exception_handler(TilingRefusal)
async def tiling_refusal_handler(request: Request, exc: TilingRefusal) -> JSONResponse:
    """a frame that does not fit the tiling grid
    refuses the WHOLE session finalization (``_finalize_session`` runs every
    image in one pass and has no per-image failure record), so the refusal
    surfaces here with its own text -- naming the image, the frame size and
    the tile size -- as a client error, never a retry-inviting server error.
    Nothing was written for any image of the session."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


reusable_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/login/", auto_error=False)


@app.get("/healthz", response_model=None)
async def healthz(settings: Settings = Depends(get_settings)) -> Any:
    """Answers 200 only with a proof that the detector runtime works, in
    order: a detector call in flight, a detection or probe that succeeded
    within ``INFERENCE_HEALTH_PROOF_MAX_AGE_SECONDS``, a probe run now on
    the detector's own executor thread. A failed or timed-out probe answers
    503 with the reason; after ``INFERENCE_HEALTH_EXIT_AFTER_FAILURES``
    failures in a row the process exits (status 3) once the answer has
    left, so the container's restart policy recreates it."""
    proof = HEALTH_STATE.proof(settings.health_proof_max_age_seconds)
    if proof is not None:
        return {"status": "ready", "proof": proof}

    loop = asyncio.get_running_loop()
    timeout = settings.health_probe_timeout_seconds
    try:
        await asyncio.wait_for(
            loop.run_in_executor(DETECTOR_EXECUTOR, _run_tracked, _probe_detector, settings.detector_impl),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        reason = f"the detector probe did not finish within {timeout:g} s"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    else:
        HEALTH_STATE.record_probe_ok()
        return {"status": "ready", "proof": "probe"}

    failures = HEALTH_STATE.record_probe_failure(reason)
    logger.warning("healthz: detector probe failed (%d in a row): %s", failures, reason)
    limit = settings.health_exit_after_failures
    if limit > 0 and failures >= limit:
        logger.error(
            "healthz: %d consecutive failed probes (limit %d), last reason: %s -- exiting with status %d "
            "so the container is restarted",
            failures, limit, reason, _EXIT_STATUS,
        )
        loop.call_later(_EXIT_DELAY_SECONDS, os._exit, _EXIT_STATUS)
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "reason": reason, "consecutive_failures": failures},
    )


@app.post("/internal/acceptance-smoke", response_model=None)
def run_acceptance_smoke_endpoint() -> dict:
    """This service's own acceptance-smoke
    entrypoint — the running production process, hit at this route, IS the
    actual acceptance flow. Loads the currently-configured
    (``INFERENCE_EXECUTION_TARGET``/``INFERENCE_DETECTOR_IMPL``) artifact
    through the real ``registry.get_detector`` path; in the currently
    configured ``INFERENCE_ACCEPTANCE_SMOKE_MODE`` ("acceptance" or
    "ci-dev"),
    refuses (409, citing the specific reason) unless it resolves to that
    target's designated entry, or — "ci-dev" — proceeds regardless (for the stub or a non-designated artifact).
    Covered by ``app/tests/test_acceptance_smoke.py``. No auth dependency, matching
    ``/internal/housekeeping/run``: internal-only network.
    """
    from app.config import get_settings

    try:
        detector = run_acceptance_smoke()
    except AcceptanceSmokeRefusal as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    designation_status = getattr(detector, "designation_status", None)
    return {
        "status": "designated" if designation_status == "designated" else "proceeded",
        "mode": get_settings().acceptance_smoke_mode,
        "execution_target": getattr(detector, "execution_target", None),
        "designation_status": designation_status,
        "detector_impl_id": getattr(detector, "impl_id", None),
    }


@app.post("/internal/detect-debug", response_model=None)
async def detect_debug_endpoint(
    file: UploadFile = File(..., description="One image, run through the configured detector"),
    tiles: int = Query(
        0,
        ge=0,
        le=1,
        description=(
            "1: for a tiled detector, also return the per-tile raw outputs in the parity "
            "candidate format, the fused regions in frame pixels, and the tiled-path provenance"
        ),
    ),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Raw detections of one picture. The session API
    (/api/v1/sessions) never surfaces raw detections — only the aggregated
    per-image score (app/scoring.py) — so there was previously no way to
    assert a specific box/label/score against a live deployed pipeline.
    This route runs one image through the currently-configured detector
    (the same registry.get_detector/_get_detector path production
    sessions use, no separate construction) and returns its raw canonical
    Detection list plus the detector's own identity fields. No auth
    dependency, matching /internal/housekeeping/run and
    /internal/acceptance-smoke: internal-only network. Debug/verification
    surface only — never consulted by session scoring itself. ``tiles=1`` on a tiled detector
    answers with the per-tile raw outputs (``tiles``: one entry
    per grid tile named ``<stem>_t<kk>_x<x0>_y<y0>`` with the runtime's full
    labels/boxes/scores in tile pixels; ``regions`` in frame pixels) plus the
    tiled-path provenance fields. Without ``tiles=1`` the response carries the canonical detection
    list only.
    """
    detector = _get_detector(settings.detector_impl)
    if tiles and not getattr(detector, "is_tiled", False):
        raise HTTPException(
            status_code=409,
            detail="tiles=1 needs a tiled detector; the configured detector runs the full-frame path",
        )
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        image_ref = ImageRef(image_id="detect-debug", filename=file.filename or "image", path=tmp_path)
        if tiles:
            tiled = await asyncio.get_running_loop().run_in_executor(
                DETECTOR_EXECUTOR, _run_tracked, detector.detect_tiled_debug, image_ref
            )
            return {
                "detector_impl_id": detector.impl_id,
                "designation_status": getattr(detector, "designation_status", None),
                "execution_target": getattr(detector, "execution_target", None),
                **tiled,
            }
        result = await asyncio.get_running_loop().run_in_executor(
            DETECTOR_EXECUTOR, _run_tracked, _detect_recorded, detector, image_ref
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "detector_impl_id": detector.impl_id,
        "designation_status": getattr(detector, "designation_status", None),
        "execution_target": getattr(detector, "execution_target", None),
        "detections": [
            {
                "label": d.label,
                "score": d.score,
                "x_min": d.x_min,
                "y_min": d.y_min,
                "x_max": d.x_max,
                "y_max": d.y_max,
            }
            for d in result.detections
        ],
    }


def _kick_housekeeping(settings: Settings) -> None:
    """The automatic trigger for
    the housekeeping pass. Scheduled via FastAPI BackgroundTasks
    from every admitted session start (create_session /
    create_session_from_path, below) -- the same non-blocking idiom
    push_session already uses for the best-image push. A
    BackgroundTasks callback runs strictly after the response has been
    built and is being sent, so this can never add latency to session
    creation, delay admission, or block capture/inference -- it is
    "session-start triggered" in timing only, not inline in the
    request/response path (housekeeping never runs inline with a session).
    Admission itself stays governed ONLY by EDGE_DISK_HEADROOM_MIN_PCT via
    check_preflight -- this kick cannot affect whether a session is
    admitted, only whether a housekeeping pass happens to run concurrently
    with it. Cheap to call unconditionally: run_housekeeping no-ops past
    its initial disk_usage check unless usage is already at or above
    EDGE_DISK_BUDGET_PCT."""
    result: HousekeepingResult = run_housekeeping(settings)
    if result.budget_triggered:
        logger.info(
            "Automatic housekeeping kick (session start): purged=%d resumed_tombstones=%d errors=%d",
            len(result.purged_sessions), len(result.resumed_tombstones), len(result.errors),
        )


@app.post("/internal/housekeeping/run", response_model=None)
def run_housekeeping_endpoint(settings: Settings = Depends(get_settings)) -> dict:
    """Manual, on-demand trigger for the housekeeping pass -- deliberately independent of session
    capture and inference, invoked out of band
    (operator, cron, or a test) for an on-demand pass in addition to the
    automatic session-start kick (`_kick_housekeeping`, above) that fires
    on every admitted session without an operator calling this endpoint.
    No auth dependency: this endpoint only exists on the edge profile's
    internal-only network (`internal: true`), unreachable from
    anywhere Gatekeeper would otherwise gate."""
    result: HousekeepingResult = run_housekeeping(settings)
    return {
        "resumed_tombstones": result.resumed_tombstones,
        "purged_sessions": result.purged_sessions,
        "errors": result.errors,
        "budget_triggered": result.budget_triggered,
    }


def _resolve_growth_stage(
    settings: Settings,
    representation: Optional[str],
    bbch_min: Optional[int],
    bbch_max: Optional[int],
    stage_id: Optional[str],
) -> GrowthStage:
    """Growth stage is operator/campaign-supplied session metadata; the
    operational default (no value supplied) is the explicit "unknown"
    marker in the deployment's configured representation.
    """
    rep = representation or settings.default_growth_stage_representation
    if rep == "bbch_range":
        if bbch_min is not None and bbch_max is not None:
            return GrowthStage(representation="bbch_range", bbch_min=bbch_min, bbch_max=bbch_max)
        return GrowthStage.unknown_for("bbch_range")
    if stage_id:
        return GrowthStage(representation="enum", stage_id=stage_id)
    return GrowthStage.unknown_for("enum")


def _content_type_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


def _find_image_record(base_dir: Path, image_id: str) -> Optional[Dict[str, Any]]:
    """
    Look up an image's stored record (filename + on-disk path) by scanning
    session metadata. Metadata-driven (not a directory glob) so it works
    identically for HTTP-uploaded images (stored under base_dir) and
    path-registered images.
    """
    sessions_root = base_dir / "sessions"
    if not sessions_root.exists():
        return None
    for sess_path in sessions_root.iterdir():
        if not sess_path.is_dir():
            continue
        meta = read_session_metadata(base_dir, sess_path.name)
        for img in meta.get("images", []):
            if img.get("image_id") == image_id:
                return img
    return None


def find_image_path(base_dir: Path, image_id: str) -> Optional[Path]:
    record = _find_image_record(base_dir, image_id)
    if not record:
        return None
    path = Path(record["path"])
    return path if path.exists() else None


def _find_session_for_image_id(base_dir: Path, image_id: str) -> Optional[tuple[str, dict]]:
    """
    (session_id, metadata) of the session holding ``image_id``. Image ids
    are content hashes, so one id can appear in several sessions; the one
    whose best image it is and that persisted detections wins, else the
    first session holding it.
    """
    sessions_root = base_dir / "sessions"
    if not sessions_root.exists():
        return None
    first: Optional[tuple[str, dict]] = None
    for sess_path in sessions_root.iterdir():
        if not sess_path.is_dir():
            continue
        meta = read_session_metadata(base_dir, sess_path.name)
        if not any(img.get("image_id") == image_id for img in meta.get("images", [])):
            continue
        if meta.get("best_image_id") == image_id and meta.get("best_image_detections") is not None:
            return sess_path.name, meta
        if first is None:
            first = (sess_path.name, meta)
    return first


def _original_filename_for_image_id(base_dir: Path, image_id: str) -> Optional[str]:
    record = _find_image_record(base_dir, image_id)
    return record.get("filename") if record else None


def _per_image_position_payload(
    record: Dict[str, Any], parsed_metadata: Optional[Dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Extract one optional source position from capture/replay metadata.

    The replay/capture seam identifies images by their manifest filename. A
    malformed or absent entry is not converted into a synthetic position;
    malformed provided telemetry is rejected so it cannot be misrepresented
    in a completed session's diagnostics carrier.
    """
    supplied = (parsed_metadata or {}).get("per_image_positions")
    if supplied is None:
        return None, None
    if not isinstance(supplied, dict):
        raise HTTPException(status_code=400, detail="metadata.per_image_positions must be an object")

    raw_position = supplied.get(record["filename"])
    if raw_position is None:
        return None, None
    if not isinstance(raw_position, dict):
        raise HTTPException(status_code=400, detail="per-image position must be an object")
    try:
        latitude = float(raw_position["latitude"])
        longitude = float(raw_position["longitude"])
        provenance = raw_position["provenance"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="per-image position requires latitude, longitude, and provenance",
        ) from exc
    if provenance not in _PER_IMAGE_POSITION_PROVENANCES:
        raise HTTPException(status_code=400, detail="per-image position provenance is not supported")
    return {"latitude": latitude, "longitude": longitude}, provenance


def _session_coordinates_from_positions(
    per_image_results: list[dict[str, Any]],
    fallback_latitude: Optional[float],
    fallback_longitude: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Centroid available per-image telemetry, otherwise retain explicit default."""
    positions = [entry["position"] for entry in per_image_results if entry["position"] is not None]
    if positions:
        return (
            sum(position["latitude"] for position in positions) / len(positions),
            sum(position["longitude"] for position in positions) / len(positions),
            "centroid",
        )
    if fallback_latitude is None or fallback_longitude is None:
        return fallback_latitude, fallback_longitude, None
    return fallback_latitude, fallback_longitude, "default"


def _tiled_path_metadata(settings: Settings, detector: Detector, accepted_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """detector_profile, tiling (tile, stride, the origins actually used, or
    null), fusion_rule_id, fusion_params, tta_views, region_threshold and
    negligibility_record_sha256 for the session metadata file. A detector
    with no tiled-path accessor (the stub implementations) runs the
    full-frame path under whatever profile is configured."""
    accessor = getattr(detector, "tiled_path_provenance", None)
    if accessor is None:
        return tiled_path_fields(
            profile=settings.detector_profile,
            tiling=None,
            frame_sizes=(),
            views=("identity",),
            threshold=settings.region_threshold,
            negligibility_record_sha256=None,
        )
    frame_sizes = (
        [frame_size_of(Path(record["path"])) for record in accepted_records]
        if getattr(detector, "is_tiled", False)
        else []
    )
    return accessor(frame_sizes)


def _finalize_session(
    settings: Settings,
    session_id: str,
    accepted_records: List[Dict[str, Any]],
    rejected_files: List[str],
    *,
    parcel_id: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    time_start: Optional[str],
    time_end: Optional[str],
    crop: Optional[str],
    pest: Optional[str],
    growth_stage_representation: Optional[str],
    growth_stage_bbch_min: Optional[int],
    growth_stage_bbch_max: Optional[int],
    growth_stage_stage_id: Optional[str],
    parsed_metadata: Optional[Dict[str, Any]],
    ingestion_mode: str,
) -> SessionCreateResponse:
    if not accepted_records:
        raise HTTPException(
            status_code=400,
            detail=(
                "Session has zero processable images "
                f"(rejected: {rejected_files})." if rejected_files
                else "Session has zero processable images."
            ),
        )

    detector = _get_detector(settings.detector_impl)

    scored_images: List[ScoredImage] = []
    detections_by_image: Dict[str, list] = {}
    for record in accepted_records:
        image_ref = ImageRef(
            image_id=record["image_id"],
            filename=record["filename"],
            path=Path(record["path"]),
        )
        result = _run_tracked(_detect_recorded, detector, image_ref)
        score = per_image_score(result.detections, settings.target_class_set, settings.confidence_floor)
        scored_images.append(ScoredImage(image_id=record["image_id"], score=score))
        detections_by_image[record["image_id"]] = [
            {
                "label": d.label,
                "score": d.score,
                "x_min": d.x_min,
                "y_min": d.y_min,
                "x_max": d.x_max,
                "y_max": d.y_max,
            }
            for d in result.detections
        ]

    best_image_id = select_best_image(scored_images)
    session_score = aggregate_session_score([s.score for s in scored_images])

    crop_value = crop or settings.default_crop
    growth_stage = _resolve_growth_stage(
        settings, growth_stage_representation, growth_stage_bbch_min, growth_stage_bbch_max, growth_stage_stage_id
    )

    provenance = {
        "detector_impl_id": detector.impl_id,
        "detector_version": detector.version,
        "designation_status": getattr(detector, "designation_status", "stub"),
        "aggregation_rule_version": AGGREGATION_RULE_VERSION,
        "target_class_set": list(settings.target_class_set),
        "confidence_floor": settings.confidence_floor,
    }
    if hasattr(detector, "contract_provenance_fields"):
        provenance.update(detector.contract_provenance_fields())

    per_image_payload: list[dict[str, Any]] = []
    for scored_image, record in zip(scored_images, accepted_records):
        position, position_provenance = _per_image_position_payload(record, parsed_metadata)
        per_image_payload.append(
            {
                "image_id": scored_image.image_id,
                "score": scored_image.score,
                "position": position,
                "position_provenance": position_provenance,
            }
        )

    session_latitude, session_longitude, position_provenance = _session_coordinates_from_positions(
        per_image_payload, latitude, longitude
    )

    metadata_payload: Dict[str, Any] = {
        "session_id": session_id,
        "parcel_id": parcel_id,
        "latitude": session_latitude,
        "longitude": session_longitude,
        "position_provenance": position_provenance,
        "time_start": time_start,
        "time_end": time_end,
        "metadata": parsed_metadata,
        "images": accepted_records,
        "rejected_files": rejected_files,
        "ingestion_mode": ingestion_mode,
        "best_image_id": best_image_id,
        "best_image_detections": detections_by_image.get(best_image_id, []),
        "infestation_score": session_score,
        "crop": crop_value,
        "growth_stage": growth_stage.model_dump(),
        "provenance": provenance,
        "contract_version": CONTRACT_VERSION,
    }
    if pest is not None:
        metadata_payload["pest"] = pest
    metadata_payload.update(_tiled_path_metadata(settings, detector, accepted_records))
    write_session_metadata(settings.storage_dir, session_id, metadata_payload)

    write_session_results(settings.storage_dir, session_id, per_image_payload)

    return SessionCreateResponse(
        session_id=session_id,
        best_image_id=best_image_id,
        infestation_score=session_score,
    )


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, body
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {exc.reason}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {exc}",
        ) from exc


def _get_gatekeeper_token(settings: Settings, *, _meter: Optional[List[tuple[str, int]]] = None) -> str:
    """
    ``_meter``, when given a list, appends this call's measured
    ``("up"|"down", byte_count)`` pairs to it: the JWT
    auth round-trip crosses the edge/cloud boundary in mixed mode, so it is counted. Optional and
    additive so every existing caller/test that doesn't pass it keeps
    today's exact behavior -- including tests that mock this whole
    function out (a mock never touches ``_meter`` regardless).
    """
    token_url = f"{settings.gatekeeper_base_url}{settings.gatekeeper_token_endpoint}"
    payload = {"username": settings.gatekeeper_username, "password": settings.gatekeeper_password}
    status_code, body = _post_json(
        token_url,
        payload,
        {},
        settings.gatekeeper_timeout_seconds,
    )
    if _meter is not None:
        _meter.append(("up", accounting.bytes_of(payload)))
        _meter.append(("down", accounting.bytes_of(body)))
    if status_code >= 400:
        raise HTTPException(status_code=502, detail=f"GateKeeper auth failed: {body}")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="GateKeeper auth returned invalid JSON") from exc

    token = data.get("access") or data.get("access_token") or data.get("token")
    if not token:
        raise HTTPException(status_code=502, detail="GateKeeper auth response missing token")
    return token


def _validate_gatekeeper_token(settings: Settings, token: str) -> None:
    status_code, body = _post_json(
        f"{settings.gatekeeper_base_url}/api/validate_token/",
        {"token": token, "token_type": "access"},
        {},
        settings.gatekeeper_timeout_seconds,
    )
    if status_code >= 400:
        raise HTTPException(
            status_code=403,
            detail=f"GateKeeper token validation failed: {body}",
        )


def _register_service(settings: Settings) -> None:
    if not settings.using_gatekeeper:
        return

    token = _get_gatekeeper_token(settings)
    base_url = settings.self_base_url or f"http://{settings.service_name}:{settings.port}/"
    endpoints = [
        {"endpoint": "healthz", "methods": ["GET"]},
        {"endpoint": "api/v1/sessions", "methods": ["POST"]},
        {"endpoint": "api/v1/probe/sink", "methods": ["POST"]},
        {"endpoint": "api/v1/sessions/path", "methods": ["POST"]},
        {"endpoint": "api/v1/sessions/{session_id}", "methods": ["GET"]},
        {"endpoint": "api/v1/sessions/{session_id}/results", "methods": ["GET"]},
        {"endpoint": "api/v1/sessions/{session_id}/push", "methods": ["POST"]},
        {"endpoint": "api/v1/sessions/{session_id}/accounting", "methods": ["GET"]},
        {"endpoint": "image/{image_id}/bytes", "methods": ["GET"]},
        {"endpoint": "image/{image_id}/annotated", "methods": ["GET"]},
        {"endpoint": "image/{image_id}/clean", "methods": ["GET"]},
    ]

    for entry in endpoints:
        status_code, body = _post_json(
            f"{settings.gatekeeper_base_url}{settings.gatekeeper_register_endpoint}",
            {
                "base_url": base_url,
                "service_name": settings.service_name,
                "endpoint": entry["endpoint"],
                "methods": entry["methods"],
            },
            {"Authorization": f"Bearer {token}"},
            settings.gatekeeper_timeout_seconds,
        )
        if status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"GateKeeper registration failed: {body}",
            )


def gatekeeper_auth(
    token: Optional[str] = Depends(reusable_oauth2),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """
    Authenticate an inbound request using one of two methods:

    1. **Pre-shared API key** (``X-API-Key`` header) — intended for external
       clients such as the Jetson-resident session-feeder. This lets the
       robot talk directly to InferenceService over HTTPS without
       OpenAgri-GateKeeper being exposed to the Internet.

    2. **GateKeeper JWT** (``Authorization: Bearer <token>``) — used by
       internal services communicating through the cloud Docker network.

    Evaluation order:
      If ``INFERENCE_EXTERNAL_API_KEY`` is configured and the request
        carries a matching ``X-API-Key`` header → **accept** (skip GK
        validation entirely).
      Otherwise, fall through to the existing GK token validation logic.
      If neither GK mode nor an API key is configured, auth is a no-op
        (development / Edge mode).

    Note: ``reusable_oauth2`` uses ``auto_error=False`` so that requests
    without a Bearer token are **not** rejected before this function runs.
    This is essential — the robot sends only ``X-API-Key``, no Bearer token.
    """
    if settings.external_api_key:
        if x_api_key:
            if x_api_key == settings.external_api_key:
                logger.debug("Request authenticated via X-API-Key")
                return None
            else:
                logger.warning("Invalid X-API-Key presented")
                raise HTTPException(
                    status_code=403,
                    detail="Invalid API key",
                )

    if not settings.using_gatekeeper:
        if not settings.external_api_key:
            return None
        if not x_api_key and not token:
            raise HTTPException(status_code=403, detail="Not authenticated")
        return None
    if not token:
        raise HTTPException(status_code=403, detail="Not authenticated")
    _validate_gatekeeper_token(settings, token)
    return token


async def _read_capped_body(request: Request, max_bytes: int) -> bytes:
    """
    Read a request body while refusing loudly (naming the cap) as soon as
    it is known to exceed ``max_bytes`` -- from the declared
    ``Content-Length`` when present, or from the streamed byte count
    otherwise. Never persisted; the caller (the probe sink) counts and
    discards.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        if declared is not None and declared > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"probe payload of {declared} bytes exceeds INFERENCE_PROBE_SINK_MAX_BYTES={max_bytes}",
            )

    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"probe payload exceeds INFERENCE_PROBE_SINK_MAX_BYTES={max_bytes}",
            )
    return bytes(chunks)


@app.post("/api/v1/probe/sink")
async def probe_sink(
    request: Request,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> dict[str, int]:
    """The link-probe's upload target.
    Authenticated exactly like the feeder's session endpoints
    (``gatekeeper_auth`` -- the pre-shared ``X-API-Key`` path); reads and
    counts a size-capped request body and returns the count; stores
    nothing (no write under ``settings.storage_dir`` or anywhere else;
    it is a side-effect-free measurement target).
    """
    body = await _read_capped_body(request, settings.probe_sink_max_bytes)
    return {"received_bytes": len(body)}


@app.post("/api/v1/sessions", response_model=SessionCreateResponse)
async def create_session(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more images"),
    session_id: Optional[str] = Form(default=None),
    parcel_id: Optional[str] = Form(default=None),
    latitude: Optional[float] = Form(default=None),
    longitude: Optional[float] = Form(default=None),
    time_start: Optional[str] = Form(default=None),
    time_end: Optional[str] = Form(default=None),
    crop: Optional[str] = Form(default=None),
    pest: Optional[str] = Form(default=None, pattern=EPPO_CODE_PATTERN),
    growth_stage_representation: Optional[str] = Form(default=None),
    growth_stage_bbch_min: Optional[int] = Form(default=None),
    growth_stage_bbch_max: Optional[int] = Form(default=None),
    growth_stage_stage_id: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(
        default=None,
        description="Optional JSON string for arbitrary session metadata",
    ),
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> SessionCreateResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    try:
        check_preflight(settings)
    except DiskAdmissionRefused as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc

    background_tasks.add_task(_kick_housekeeping, settings)

    session_id = create_session_id(session_id)

    accepted_records: list[dict[str, Any]] = []
    rejected_files: list[str] = []

    for upload in files:
        filename = upload.filename or ""
        if not is_supported_image_format(filename):
            rejected_files.append(filename)
            await upload.read()
            continue
        image_id, path, content_type = await save_upload_file(
            settings.storage_dir,
            session_id,
            upload,
            settings.deterministic_seed,
        )
        accepted_records.append(
            {
                "image_id": image_id,
                "filename": filename,
                "content_type": content_type,
                "path": str(path),
            }
        )

    parsed_metadata: Optional[dict[str, Any]] = None
    if metadata:
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
        if parsed is not None and not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="metadata must be a JSON object")
        parsed_metadata = parsed

    return await asyncio.get_running_loop().run_in_executor(
        DETECTOR_EXECUTOR,
        functools.partial(
            _finalize_session,
            settings,
            session_id,
            accepted_records,
            rejected_files,
            parcel_id=parcel_id,
        latitude=latitude,
        longitude=longitude,
        time_start=time_start,
        time_end=time_end,
        crop=crop,
        pest=pest,
        growth_stage_representation=growth_stage_representation,
        growth_stage_bbch_min=growth_stage_bbch_min,
        growth_stage_bbch_max=growth_stage_bbch_max,
        growth_stage_stage_id=growth_stage_stage_id,
            parsed_metadata=parsed_metadata,
            ingestion_mode="http_upload",
        ),
    )


@app.post("/api/v1/sessions/path", response_model=SessionCreateResponse)
def create_session_from_path(
    body: PathRegistrationRequest,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> SessionCreateResponse:
    """
    Zero-copy path-registration intake — valid only on
    deployments where the caller and this service share a filesystem
    (mixed/edge).
    """
    try:
        check_preflight(settings)
    except DiskAdmissionRefused as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc

    background_tasks.add_task(_kick_housekeeping, settings)

    try:
        resolved = resolve_contained_path(settings.data_root, Path(body.path))
    except PathEscapesDataRootError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"registered path escapes the configured data root: {exc}",
        ) from exc
    except PathUnreadableError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"registered path cannot be resolved or read: {exc}",
        ) from exc

    try:
        manifest_images, _skipped_extra = list_manifest_images(resolved)
    except SessionManifestError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"registered path does not conform to the session-folder manifest contract: {exc}",
        ) from exc

    session_id = create_session_id(body.session_id)

    accepted_records: list[dict[str, Any]] = []
    rejected_files: list[str] = []

    for src_path in manifest_images:
        if not is_supported_image_format(src_path.name):
            rejected_files.append(src_path.name)
            continue
        content = src_path.read_bytes()
        image_id = generate_image_id(content, settings.deterministic_seed)
        accepted_records.append(
            {
                "image_id": image_id,
                "filename": src_path.name,
                "content_type": _content_type_for_suffix(src_path.suffix),
                "path": str(src_path),
            }
        )

    return DETECTOR_EXECUTOR.submit(
        _finalize_session,
        settings,
        session_id,
        accepted_records,
        rejected_files,
        parcel_id=body.parcel_id,
        latitude=body.latitude,
        longitude=body.longitude,
        time_start=body.time_start,
        time_end=body.time_end,
        crop=body.crop,
        pest=body.pest,
        growth_stage_representation=body.growth_stage_representation,
        growth_stage_bbch_min=body.growth_stage_bbch_min,
        growth_stage_bbch_max=body.growth_stage_bbch_max,
        growth_stage_stage_id=body.growth_stage_stage_id,
        parsed_metadata=body.metadata,
        ingestion_mode="path_registration",
    ).result()


@app.get("/api/v1/sessions/{session_id}", response_model=SessionMetadataResponse)
def get_session(
    session_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> SessionMetadataResponse:
    session_path = session_dir(settings.storage_dir, session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    payload = read_session_metadata(settings.storage_dir, session_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Session metadata not found")

    return SessionMetadataResponse(
        session_id=payload.get("session_id", session_id),
        best_image_id=payload.get("best_image_id", ""),
        infestation_score=payload.get("infestation_score", 0.0),
        images=payload.get("images", []),
        crop=payload.get("crop"),
        pest=payload.get("pest"),
        growth_stage=payload.get("growth_stage"),
        provenance=payload.get("provenance"),
        contract_version=payload.get("contract_version"),
        ingestion_mode=payload.get("ingestion_mode"),
        parcel_id=payload.get("parcel_id"),
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        position_provenance=payload.get("position_provenance"),
        time_start=payload.get("time_start"),
        time_end=payload.get("time_end"),
        metadata=payload.get("metadata"),
        images_pending=payload.get("images_pending"),
        best_image_push_attempts=payload.get("best_image_push_attempts"),
        best_image_artifact_hash=payload.get("best_image_artifact_hash"),
    )


@app.get("/api/v1/sessions/{session_id}/results", response_model=SessionResults)
def get_session_results(
    session_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> SessionResults:
    """Per-image evaluation carrier — not consumed by risk engines."""
    session_path = session_dir(settings.storage_dir, session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    results = read_session_results(settings.storage_dir, session_id)
    return SessionResults(session_id=session_id, results=results)


@app.get("/api/v1/sessions/{session_id}/accounting")
def get_session_accounting(
    session_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """The persisted per-session accounting record (session id,
    scenario, bytes up/down, MB, and the individual measured transfers) --
    404 until this session has made at
    least one accounted outbound call (session submission or best-image
    push).
    """
    record = accounting.load_record(settings.storage_dir, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="No accounting record for this session")
    return record


def _build_submission_from_metadata(payload: dict, session_id: str) -> SessionSubmission:
    """
    Build the versioned SessionSubmission from stored session metadata.

    Timestamps are required for submission. Coordinates are optional
    telemetry: their absence is preserved in the contract rather than
    replaced with a legacy deployment default.
    """
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    time_start = payload.get("time_start")
    time_end = payload.get("time_end")

    missing = []
    if not time_start:
        missing.append("time_start")
    if not time_end:
        missing.append("time_end")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Session is missing required field(s) for the contract submission: "
                f"{', '.join(missing)}. Re-create the session supplying them, "
                "or use the override body to provide the submission explicitly."
            ),
        )

    growth_stage_data = payload.get("growth_stage") or GrowthStage.unknown_for("enum").model_dump()
    provenance_data = payload.get("provenance") or {
        "detector_impl_id": "unknown",
        "detector_version": "unknown",
        "designation_status": "stub",
        "aggregation_rule_version": AGGREGATION_RULE_VERSION,
        "target_class_set": [],
        "confidence_floor": 0.0,
    }

    try:
        risk_input = RiskInput(
            session_id=payload.get("session_id", session_id),
            infestation_score=payload.get("infestation_score", 0.0),
            crop=payload.get("crop") or "unknown",
            growth_stage=GrowthStage(**growth_stage_data),
            time_start=time_start,
            time_end=time_end,
            latitude=latitude,
            longitude=longitude,
        )
        envelope = ReportingEnvelope(
            session_id=payload.get("session_id", session_id),
            best_image_id=payload.get("best_image_id", ""),
            image_count=len(payload.get("images", [])),
            time_start=time_start,
            time_end=time_end,
            latitude=latitude,
            longitude=longitude,
            parcel_id=payload.get("parcel_id"),
            pest=payload.get("pest"),
            provenance=DetectorProvenance(**provenance_data),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid session data for submission: {exc}") from exc

    return SessionSubmission(
        contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope
    )


def _record_meter(settings: Settings, session_id: str, meter: List[tuple[str, int]], kind: str) -> None:
    for direction, byte_count in meter:
        accounting.record_transfer(
            settings.storage_dir,
            session_id,
            scenario=settings.scenario,
            component="inf",
            endpoint=f"{settings.gatekeeper_base_url}{settings.gatekeeper_token_endpoint}",
            direction=direction,
            kind=kind,
            byte_count=byte_count,
            crosses_boundary=settings.scenario == "mixed",
            profile=settings.run_label,
        )


def _update_session_metadata_fields(settings: Settings, session_id: str, **fields: Any) -> None:
    payload = read_session_metadata(settings.storage_dir, session_id)
    if not payload:
        return
    payload.update(fields)
    write_session_metadata(settings.storage_dir, session_id, payload)


def _push_best_image_pair_once(
    settings: Settings,
    session_id: str,
    image_id: str,
    clean_bytes: bytes,
    clean_content_type: str,
    annotated_bytes: bytes,
    annotated_content_type: str,
    artifact_hash: str,
) -> tuple[bool, str]:
    """
    One push attempt to the Reporting service's upload endpoint. Returns (success, detail);
    success means the Reporting service answered 2xx AND its report now carries both variants.
    """
    upload_payload = {
        "contract_version": CONTRACT_VERSION,
        "image_id": image_id,
        "artifact_hash": artifact_hash,
        "clean_content_type": clean_content_type,
        "clean_b64": base64.b64encode(clean_bytes).decode("ascii"),
        "annotated_content_type": annotated_content_type,
        "annotated_b64": base64.b64encode(annotated_bytes).decode("ascii"),
    }
    _meter: List[tuple[str, int]] = []
    try:
        token = _get_gatekeeper_token(settings, _meter=_meter)
    except HTTPException as exc:
        _record_meter(settings, session_id, _meter, "gk_auth")
        return False, f"GateKeeper auth failed: {exc.detail}"
    _record_meter(settings, session_id, _meter, "gk_auth")

    target_url = (
        f"{settings.gatekeeper_base_url}/api/proxy/"
        f"{settings.rs_service_name}{settings.rs_upload_path}{session_id}/"
    )
    try:
        status_code, body = _post_json(
            target_url,
            upload_payload,
            {"Authorization": f"Bearer {token}"},
            settings.gatekeeper_timeout_seconds,
        )
    except HTTPException as exc:
        return False, str(exc.detail)

    accounting.record_transfer(
        settings.storage_dir,
        session_id,
        scenario=settings.scenario,
        component="inf",
        endpoint=target_url,
        direction="up",
        kind="best_image_pair",
        byte_count=accounting.bytes_of(upload_payload),
        crosses_boundary=settings.scenario == "mixed",
        profile=settings.run_label,
    )
    accounting.record_transfer(
        settings.storage_dir,
        session_id,
        scenario=settings.scenario,
        component="inf",
        endpoint=target_url,
        direction="down",
        kind="best_image_pair",
        byte_count=accounting.bytes_of(body),
        crosses_boundary=settings.scenario == "mixed",
        profile=settings.run_label,
    )

    if status_code >= 400:
        return False, f"the Reporting service upload failed: HTTP {status_code}: {body}"
    if not _rs_confirms_pair_embedded(body):
        return False, (
            "the Reporting service stored the pair but its report does not carry both variants "
            f"(images_embedded is not true): {body}"
        )
    return True, body


def _rs_confirms_pair_embedded(body: str) -> bool:
    """The Reporting service answers ``stored`` as soon as
    the pair is on its disk; the session is delivered only when the Reporting service also says
    its report carries both variants (``images_embedded`` true). A body
    without the field (an older Reporting service) or one that is not a JSON object counts
    as not embedded, never as embedded.
    """
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("images_embedded") is True


def _continue_best_image_push_until_delivered(
    settings: Settings,
    session_id: str,
    push_attempt: Callable[[int], bool],
    attempts_so_far: int,
) -> None:
    """
    Post-exhaustion continuation: after the bounded sequence in ``_run_best_image_push_sequence``,
    the push keeps retrying in the background until it lands, at a fixed,
    low-frequency, operator-tunable interval
    (``best_image_push_continuation_interval_sec``) instead of continuing
    the exponential backoff, which would grow unboundedly. There is no
    attempt or deadline cap here (only the initial bounded sequence has
    one) -- this loop runs until the pair delivers or the session's
    metadata disappears (e.g. retention cleanup), whichever comes first.
    """
    attempt_number = attempts_so_far
    while True:
        time.sleep(settings.best_image_push_continuation_interval_sec)

        if not read_session_metadata(settings.storage_dir, session_id):
            logger.info(
                "Best-image push continuation: session %s no longer exists, stopping", session_id
            )
            return

        attempt_number += 1
        if push_attempt(attempt_number):
            return


def _run_best_image_push_sequence(settings: Settings, session_id: str) -> None:
    """
    Bounded-retry best-image-pair push: up to
    ``best_image_push_max_retries + 1`` attempts total, separated by
    exponential backoff starting at ``best_image_push_backoff_base_sec``,
    the whole sequence bounded by ``best_image_push_deadline_sec``. If that
    bounded sequence exhausts without delivering the pair, hands off to
    ``_continue_best_image_push_until_delivered`` so the session keeps
    self-healing in the background instead of remaining ``images_pending``
    forever.
    Marks the session ``images_pending`` for the duration and clears it on
    success. Runs entirely in the background (scheduled via
    BackgroundTasks) so push_session's own response is never held up by
    it -- report generation must never block on this.
    """
    try:
        payload = read_session_metadata(settings.storage_dir, session_id)
        if not payload:
            return
        best_image_id = payload.get("best_image_id", "")
        if not best_image_id:
            return

        clean_raw, annotated_raw = _resolve_pair_raw(settings, session_id, best_image_id, payload)
    except Exception as exc:  # noqa: BLE001 - unresolvable pair is not retryable
        logger.warning("Best-image push: cannot resolve pair for session %s: %s", session_id, exc)
        return

    clean_bytes, clean_ct, annotated_bytes, annotated_ct = compress_pair_to_cap(
        clean_raw,
        annotated_raw,
        settings.best_image_max_bytes_per_image,
        settings.best_image_max_bytes_combined,
    )
    artifact_hash = hashlib.sha256(clean_bytes + annotated_bytes).hexdigest()

    def push_attempt(attempt_number: int) -> bool:
        try:
            success, detail = _push_best_image_pair_once(
                settings,
                session_id,
                best_image_id,
                clean_bytes,
                clean_ct,
                annotated_bytes,
                annotated_ct,
                artifact_hash,
            )
        except Exception as exc:  # noqa: BLE001 - any failure just means "retry later"
            success, detail = False, str(exc)

        _update_session_metadata_fields(settings, session_id, best_image_push_attempts=attempt_number)

        if success:
            _update_session_metadata_fields(settings, session_id, images_pending=False)
            logger.info("Best-image pair pushed for session %s on attempt %d", session_id, attempt_number)
            return True

        logger.warning(
            "Best-image push attempt %d failed for session %s: %s",
            attempt_number, session_id, detail,
        )
        return False

    max_attempts = settings.best_image_push_max_retries + 1
    deadline = time.monotonic() + settings.best_image_push_deadline_sec
    backoff = settings.best_image_push_backoff_base_sec

    _update_session_metadata_fields(
        settings,
        session_id,
        images_pending=True,
        best_image_artifact_hash=artifact_hash,
        best_image_push_attempts=0,
    )

    for attempt_number in range(1, max_attempts + 1):
        if attempt_number > 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(backoff, remaining))
            backoff *= 2

        if push_attempt(attempt_number):
            return

    logger.warning(
        "Best-image push exhausted for session %s after %d attempt(s); "
        "continuing in the background every %ds until it lands",
        session_id, max_attempts, settings.best_image_push_continuation_interval_sec,
    )
    _continue_best_image_push_until_delivered(settings, session_id, push_attempt, max_attempts)


@app.post("/api/v1/sessions/{session_id}/push", response_model=PushResponse)
def push_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    override: Optional[SessionSubmission] = Body(default=None),
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> PushResponse:
    payload = read_session_metadata(settings.storage_dir, session_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Session metadata not found")

    submission = override or _build_submission_from_metadata(payload, session_id)

    if settings.best_image_push_enabled:
        background_tasks.add_task(_run_best_image_push_sequence, settings, session_id)

    if not settings.orch_push_enabled:
        return PushResponse(
            pushed=False,
            response_body="Orchestrator push disabled (INFERENCE_ORCH_PUSH_ENABLED=false).",
        )

    orch_service_name = os.getenv("INFERENCE_ORCH_SERVICE_NAME", "orchestrator")
    push_payload = submission.model_dump(mode="json", exclude_none=True)

    if settings.using_gatekeeper:
        target_url = (
            f"{settings.gatekeeper_base_url}/api/proxy/"
            f"{orch_service_name}{settings.orch_push_path}"
        )
        _meter: List[tuple[str, int]] = []
        token = _get_gatekeeper_token(settings, _meter=_meter)
        _record_meter(settings, session_id, _meter, "gk_auth")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        orch_base_url = os.getenv("INFERENCE_ORCH_BASE_URL")
        if not orch_base_url:
            raise HTTPException(
                status_code=500,
                detail="INFERENCE_ORCH_BASE_URL is required when not using GateKeeper",
            )
        target_url = f"{orch_base_url}{settings.orch_push_path}"
        if not settings.orch_edge_shared_secret:
            raise HTTPException(
                status_code=500,
                detail="INFERENCE_ORCH_EDGE_SHARED_SECRET is required when pushing to the "
                "orchestrator without GateKeeper",
            )
        headers = {"X-Edge-Shared-Secret": settings.orch_edge_shared_secret}

    status_code, body = _post_json(
        target_url,
        push_payload,
        headers,
        settings.gatekeeper_timeout_seconds,
    )

    accounting.record_transfer(
        settings.storage_dir,
        session_id,
        scenario=settings.scenario,
        component="inf",
        endpoint=target_url,
        direction="up",
        kind="session_submission",
        byte_count=accounting.bytes_of(push_payload),
        crosses_boundary=settings.scenario == "mixed",
        profile=settings.run_label,
    )
    accounting.record_transfer(
        settings.storage_dir,
        session_id,
        scenario=settings.scenario,
        component="inf",
        endpoint=target_url,
        direction="down",
        kind="session_submission",
        byte_count=accounting.bytes_of(body),
        crosses_boundary=settings.scenario == "mixed",
        profile=settings.run_label,
    )

    if status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Orchestrator push failed: {body}")

    return PushResponse(
        pushed=True,
        target_url=target_url,
        status_code=status_code,
        response_body=body,
    )


@app.get("/image/{image_id}/bytes")
def get_image_bytes(
    image_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    path = find_image_path(settings.storage_dir, image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/jpeg"
    suffix = path.suffix.lower()
    if suffix == ".png":
        media_type = "image/png"

    return FileResponse(path, media_type=media_type, filename=path.name)


def _resolve_annotated_path(image_id: str, prefix: str, settings: Settings) -> Path:
    """
    Given an image_id, resolve to the corresponding annotated or clean
    image file in the annotated_images directory.
    """
    orig_filename = _original_filename_for_image_id(settings.storage_dir, image_id)
    if not orig_filename:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot resolve original filename for image_id={image_id}",
        )

    stem = Path(orig_filename).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = _ANNOTATED_IMAGES_DIR / f"{prefix}_{stem}{ext}"
        if candidate.exists():
            return candidate

    raise HTTPException(
        status_code=404,
        detail=f"{prefix} image not found for {orig_filename}",
    )


_OVERLAY_COLOUR = (235, 104, 52)


def _validated_detection_box(index: int, d: Any) -> tuple[str, float, float, float, float, float]:
    """(label, score, x_min, y_min, x_max, y_max) of one persisted detection,
    the box normalised [0,1]. Refuses by field name: a coordinate outside
    [0,1] is never guessed to be pixels."""
    required = ("label", "score", "x_min", "y_min", "x_max", "y_max")
    numeric = required[1:]
    if (
        not isinstance(d, dict)
        or any(k not in d for k in required)
        or not all(isinstance(d[k], (int, float)) and not isinstance(d[k], bool) for k in numeric)
    ):
        raise HTTPException(
            status_code=422,
            detail=f"persisted best_image_detections entry {index} is malformed — refusing to render a partial annotation",
        )
    for k in required[2:]:
        if not 0.0 <= d[k] <= 1.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"persisted best_image_detections entry {index}: {k}={d[k]} is outside the "
                    "normalised range [0, 1] — refusing to render"
                ),
            )
    for lo, hi in (("x_min", "x_max"), ("y_min", "y_max")):
        if not d[lo] < d[hi]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"persisted best_image_detections entry {index}: {lo}={d[lo]} is not below "
                    f"{hi}={d[hi]} — refusing to render"
                ),
            )
    return (str(d["label"]), d["score"], d["x_min"], d["y_min"], d["x_max"], d["y_max"])


def _render_annotated_from_detections(original_path: Path, detections: list) -> bytes:
    """
    Real-session annotated render: the stored original with the session's
    persisted best-image detections drawn as boxes only, with no text. The
    persisted boxes are NORMALISED [0,1] xyxy (detectors/dfine_common.py,
    detectors/tiled_path.py); they are scaled by the image width/height,
    with the line width proportional to the longer side and one colour for
    every box. Pure PIL — no detector, no device runtime — so it is safe on
    any thread. An empty detections list renders honestly as the original
    with nothing drawn.
    """
    from PIL import Image, ImageDraw

    boxes = [_validated_detection_box(i, d) for i, d in enumerate(detections)]

    with Image.open(original_path) as im:
        img = im.convert("RGB")
    width, height = img.size
    longer = max(width, height)
    stroke = max(2, round(longer / 400))
    draw = ImageDraw.Draw(img)
    for _label, _score, x_min, y_min, x_max, y_max in boxes:
        left, top = round(x_min * width), round(y_min * height)
        right, bottom = min(round(x_max * width), width - 1), min(round(y_max * height), height - 1)
        draw.rectangle((left, top, right, bottom), outline=_OVERLAY_COLOUR, width=stroke)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class _ImageVariant(NamedTuple):
    data: bytes
    media_type: str
    filename: str


def _fixture_pair(image_id: str, settings: Settings) -> Optional[tuple[Path, Path]]:
    """(clean, annotated) fixture files for the image, only when BOTH exist."""
    try:
        return (
            _resolve_annotated_path(image_id, "clean", settings),
            _resolve_annotated_path(image_id, "annotated", settings),
        )
    except HTTPException:
        return None


def _resolve_image_variant(
    settings: Settings,
    image_id: str,
    variant: Literal["clean", "annotated"],
    *,
    session_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> _ImageVariant:
    """
    One variant of a best image — the single resolver behind
    ``GET /image/{id}/clean``, ``GET /image/{id}/annotated`` and the Mixed
    pair push. Precedence: the pre-generated fixture pair (byte-for-byte;
    the session's detections are not read) — else the REAL-SESSION variant:
    clean is the stored original itself, annotated is rendered from the
    session metadata's persisted best_image_detections. A real session
    without persisted detections refuses the annotated variant only (404
    naming the session); a session finalized before detections were
    persisted cannot fabricate one. The session is looked up from the
    image id unless the caller (the pair push) already holds it.
    """
    fixture = _fixture_pair(image_id, settings)
    if fixture is not None:
        path = fixture[0] if variant == "clean" else fixture[1]
        return _ImageVariant(path.read_bytes(), _content_type_for_suffix(path.suffix), path.name)

    if payload is None:
        found = _find_session_for_image_id(settings.storage_dir, image_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Cannot resolve a session for image_id={image_id}")
        session_id, payload = found
    original = find_image_path(settings.storage_dir, image_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"stored original not found for image_id={image_id}")
    if variant == "clean":
        return _ImageVariant(original.read_bytes(), _content_type_for_suffix(original.suffix), original.name)

    detections = payload.get("best_image_detections") if payload.get("best_image_id") == image_id else None
    if detections is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no fixture pair and no persisted best_image_detections for session {session_id} — "
                "cannot render a real-session annotated image"
            ),
        )
    return _ImageVariant(
        _render_annotated_from_detections(original, detections), "image/jpeg", f"annotated_{original.stem}.jpg"
    )


def _resolve_pair_raw(settings: Settings, session_id: str, best_image_id: str, payload: dict) -> tuple[bytes, bytes]:
    """
    (clean_raw, annotated_raw) for the best-image pair, both from
    ``_resolve_image_variant``. All-or-nothing: a pair without an
    annotated image is not pushed, so a missing annotated variant fails
    the whole pair (404) even though the clean route alone would serve.
    """
    clean = _resolve_image_variant(settings, best_image_id, "clean", session_id=session_id, payload=payload)
    annotated = _resolve_image_variant(settings, best_image_id, "annotated", session_id=session_id, payload=payload)
    return clean.data, annotated.data


def _variant_response(variant: _ImageVariant) -> Response:
    return Response(
        content=variant.data,
        media_type=variant.media_type,
        headers={"Content-Disposition": f'attachment; filename="{variant.filename}"'},
    )


@app.get("/image/{image_id}/annotated")
@app.get("/image/{image_id}/annotated/", include_in_schema=False)
def get_annotated_image(
    image_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Serve the annotated image (boxes drawn) for an image_id: the fixture pair, else a real session's rendered overlay."""
    return _variant_response(_resolve_image_variant(settings, image_id, "annotated"))


@app.get("/image/{image_id}/clean")
@app.get("/image/{image_id}/clean/", include_in_schema=False)
def get_clean_image(
    image_id: str,
    token: Optional[str] = Depends(gatekeeper_auth),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Serve the clean image (no boxes) for an image_id: the fixture pair, else a real session's stored original."""
    return _variant_response(_resolve_image_variant(settings, image_id, "clean"))
