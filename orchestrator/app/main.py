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
Session orchestrator service: intake, persistence and the state machine; the PDM and fuzzy fan-out with
fusion and failure isolation; report composition and triggering (the intake endpoint drives a session
all the way to REPORTED/FAILED); the query APIs (this module's GET endpoints); structured logging with
correlation-id propagation (`RequestIDMiddleware` below); the Gatekeeper self-registration at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from config import DeploymentProfile, get_settings
from connection_audit import install as _install_connection_audit
from contract import SessionSubmission
from db.session import SessionLocal, engine, get_db
from gk_registration import register_apis_to_gatekeeper
import housekeeping
from intake_auth import authenticate_intake
from logging_utils import configure_logging, request_id_var, session_id_var
from models import SessionState
from queries import (
    SessionNotFound,
    get_intake_state,
    get_paired_assessments,
    get_report_state,
    get_session_detail,
    list_sessions,
)
from schemas import IntakeResponse
from service import (
    EDGE_SYNC_ORIGIN,
    ContractVersionRejected,
    IdempotencyConflict,
    advance_session,
    intake_submission,
)

configure_logging(get_settings().log_level)
logger = logging.getLogger("orchestrator")
_install_connection_audit(get_settings().service_name)

housekeeping.resume_leftover_tombstones(get_settings())


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
    an admitted session-submission intake, so it never runs at all during
    a lull and misses a budget crossing caused by the session that just
    finished — this loop is what actually keeps the pass "active" through
    disk usage at or above EDGE_DISK_BUDGET_PCT, with zero session activity.
    The session-start kick stays as an additional fast-reaction trigger,
    not a substitute for this loop. Runs for the lifetime of the process;
    cancelled cleanly on shutdown (`lifespan`, below). A single failed
    pass is logged and never kills the loop -- unlike `_kick_housekeeping`
    (a one-shot BackgroundTasks callback where a bare `except Exception`
    would be unusual for this codebase's conventions), this loop has no
    other caller to observe or retry a failure, so it must swallow broad
    exceptions itself to keep running indefinitely."""
    settings = get_settings()
    while True:
        await asyncio.sleep(settings.edge_housekeeping_interval_sec)
        try:
            result = await _run_off_loop(_run_housekeeping_pass, get_settings())
            if result.budget_triggered:
                logger.info(
                    "Periodic housekeeping pass: purged=%d resumed_tombstones=%d errors=%d",
                    len(result.purged_sessions), len(result.resumed_tombstones), len(result.errors),
                )
        except Exception:  # noqa: BLE001 - the periodic loop must never die on one bad pass
            logger.exception("Periodic housekeeping pass failed")


def _run_housekeeping_pass(settings) -> housekeeping.HousekeepingResult:
    """Runs one housekeeping pass with its own DB session, created and
    closed within the calling thread (see `_periodic_housekeeping_loop`'s
    `_run_off_loop` call -- a SQLAlchemy Session must not cross
    threads)."""
    db = SessionLocal()
    try:
        return housekeeping.run_housekeeping(db, settings)
    finally:
        db.close()


_PERIODIC_TASK_SHUTDOWN_TIMEOUT_SECONDS = 2.0


@asynccontextmanager
async def lifespan(fa: FastAPI):
    settings = get_settings()
    if settings.deployment_profile is not DeploymentProfile.EDGE:
        register_apis_to_gatekeeper(settings, fa)
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


app = FastAPI(title="SessionOrchestrator", lifespan=lifespan)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Reads the inbound `X-Request-ID` header (when present)
    and makes it available to structured log lines and outbound calls for
    the duration of the request via a context-local var (`logging_utils`).
    Does not mint an id when the header is absent — mirrors
    OpenAgri-PestAndDiseaseManagement's
    convention verbatim: consumption of an id GateKeeper already generates
    upstream for any GK-proxied call, not end-to-end minting here.
    """

    async def dispatch(self, request: Request, call_next):
        incoming_request_id = request.headers.get("X-Request-ID")
        token = request_id_var.set(incoming_request_id or "-")
        try:
            response = await call_next(request)
            logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
        finally:
            request_id_var.reset(token)
        if incoming_request_id:
            response.headers["X-Request-ID"] = incoming_request_id
        return response


app.add_middleware(RequestIDMiddleware)


@app.get("/health")
def health() -> dict:
    """Liveness — no dependency on the database or any downstream service."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> dict:
    """Readiness — verifies the SQLite connection is usable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - readiness must report, not raise
        logger.warning("Readiness check failed: DB unreachable: %s", exc)
        response.status_code = 503
        return {"status": "degraded", "detail": "database unreachable"}
    return {"status": "ok"}


def _kick_housekeeping() -> None:
    """The automatic trigger for
    the housekeeping pass. Scheduled via FastAPI BackgroundTasks
    from every session-submission intake (`receive_session_submission`,
    below) so it fires without an operator calling
    `/internal/housekeeping/run`. A BackgroundTasks callback only runs
    after the response has already been built and is being sent, so this
    never adds latency to intake or affects its outcome: it is triggered at session start but
    runs outside the request/response path.
    Opens its own DB session (`SessionLocal`, not the request-scoped `db`
    dependency) because `get_db`'s `finally: db.close` has already run
    by the time a background task executes. Cheap to call
    unconditionally: `run_housekeeping` no-ops past its initial
    disk-usage check unless usage is already at or above
    EDGE_DISK_BUDGET_PCT; at a few dozen sessions a day with a single writer the extra
    per-submission DB session is negligible."""
    db = SessionLocal()
    try:
        result = housekeeping.run_housekeeping(db, get_settings())
        if result.budget_triggered:
            logger.info(
                "Automatic housekeeping kick (session start): purged=%d resumed_tombstones=%d errors=%d",
                len(result.purged_sessions), len(result.resumed_tombstones), len(result.errors),
            )
    finally:
        db.close()


@app.post("/internal/housekeeping/run")
def run_housekeeping_endpoint(db: Session = Depends(get_db)) -> dict:
    """Manual, on-demand trigger -- deliberately independent of session
    processing, invoked out of band (operator, cron, or a test) for an
    on-demand pass in addition to the automatic session-start kick
    (`_kick_housekeeping`, above) that fires on every intake without an
    operator calling this endpoint. Answers 404 outside the edge profile, whose
    network is internal-only."""
    if get_settings().deployment_profile is not DeploymentProfile.EDGE:
        raise HTTPException(status_code=404, detail="housekeeping runs on the edge profile only")
    result = housekeeping.run_housekeeping(db, get_settings())
    return {
        "resumed_tombstones": result.resumed_tombstones,
        "purged_sessions": result.purged_sessions,
        "errors": result.errors,
        "budget_triggered": result.budget_triggered,
    }


_INTAKE_ORIGINS = {None, EDGE_SYNC_ORIGIN}


@app.post("/api/v1/sessions/", response_model=IntakeResponse, dependencies=[Depends(authenticate_intake)])
def receive_session_submission(
    submission: SessionSubmission,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    origin: Optional[str] = Query(default=None),
    edge_report_id: Optional[str] = Query(default=None),
) -> IntakeResponse:
    if origin not in _INTAKE_ORIGINS:
        raise HTTPException(status_code=422, detail=f"unknown origin {origin!r}: only 'edge-sync' is accepted")
    if origin == EDGE_SYNC_ORIGIN and not (edge_report_id or "").strip():
        raise HTTPException(status_code=422, detail="origin 'edge-sync' requires a non-empty edge_report_id")

    background_tasks.add_task(_kick_housekeeping)

    session_id_token = session_id_var.set(submission.risk_input.session_id)
    try:
        return _receive_session_submission(submission, db, origin, edge_report_id)
    finally:
        session_id_var.reset(session_id_token)


def _report_id_if_reported(db: Session, session_id: str, state: str) -> Optional[str]:
    """`report_id` when `state` is REPORTED, `None` otherwise — shared by
    both the accepted and duplicate intake branches."""
    if state != SessionState.REPORTED.value:
        return None
    report = get_report_state(db, session_id)
    return report.report_id if report is not None else None


def _receive_session_submission(
    submission: SessionSubmission,
    db: Session,
    origin: Optional[str] = None,
    edge_report_id: Optional[str] = None,
) -> IntakeResponse:
    try:
        outcome = intake_submission(db, submission, origin=origin, edge_report_id=edge_report_id)
    except ContractVersionRejected as exc:
        logger.warning("intake rejected: contract_version issue: %s", exc.warning)
        raise HTTPException(status_code=400, detail=exc.warning) from exc
    except IdempotencyConflict as exc:
        logger.warning("intake rejected: idempotency conflict: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if outcome.duplicate:
        logger.info("intake: duplicate submission, state=%s", outcome.state)
        return IntakeResponse(
            status="duplicate",
            session_id=outcome.session_id,
            state=outcome.state,
            errors=outcome.errors,
            report_id=_report_id_if_reported(db, outcome.session_id, outcome.state),
        )
    logger.info("intake: accepted, state=%s", outcome.state)

    report_id = None
    try:
        session = advance_session(db, outcome.session_id)
        state = session.state
        report_id = _report_id_if_reported(db, outcome.session_id, state)
    except Exception as exc:  # noqa: BLE001 - never fail the ack on a processing error
        logger.exception("advance_session failed for session_id=%s", outcome.session_id)
        state = outcome.state

    return IntakeResponse(
        status="accepted", session_id=outcome.session_id, state=state, errors=outcome.errors, report_id=report_id
    )


@app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(authenticate_intake)])
def get_session(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_session_detail(db, session_id, settings=get_settings())
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"no session found for session_id={session_id!r}") from exc


@app.get("/api/v1/sessions/", dependencies=[Depends(authenticate_intake)])
def list_sessions_endpoint(
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    scenario: Optional[str] = None,
    include_pre_cutover: bool = False,
    db: Session = Depends(get_db),
):
    return list_sessions(db, start=start, end=end, scenario=scenario, exclude_pre_cutover=not include_pre_cutover)


@app.get("/api/v1/sessions/{session_id}/assessments", dependencies=[Depends(authenticate_intake)])
def get_session_assessments(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_paired_assessments(db, session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"no session found for session_id={session_id!r}") from exc


@app.get("/api/v1/sessions/{session_id}/report", dependencies=[Depends(authenticate_intake)])
def get_session_report(session_id: str, db: Session = Depends(get_db)):
    try:
        report = get_report_state(db, session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"no session found for session_id={session_id!r}") from exc
    if report is None:
        return {"state": "not_reported"}
    return report


@app.get("/api/v1/sessions/{session_id}/intake-state", dependencies=[Depends(authenticate_intake)])
def get_intake_state_endpoint(session_id: str, db: Session = Depends(get_db)):
    """The single answer to "is this
    session id clean?" -- read-only, across every intake-side table
    (idempotency key included), never just the session-detail record. Never
    404s on an absent session_id: a session id can be "held" purely by an
    idempotency_key row with no session_record at all."""
    return get_intake_state(db, session_id)
