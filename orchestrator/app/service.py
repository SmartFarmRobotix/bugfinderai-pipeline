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
Intake idempotency, the workflow state machine, and the cutover-boundary
query, the PDM fan-out step, the in-process
fuzzy fan-out, the shared fusion step applied to both paths, failure isolation between the two engines, and
report composition/triggering.

`advance_session` drives a session forward one bounded unit of work per
`process_session_step` call, committing once per call — the single commit
is what makes a restart mid-workflow safe: either a whole unit of work
(an assessment row, a state change, or both together) lands, or none of it
does. While `state == "assessing"`, each call independently concludes
whichever risk path (PDM, then fuzzy) has not yet concluded — a
"concluded" path is one that already has a `RiskAssessmentRecord` row
(status "present" or "failed"); the two paths conclude independently, so
one failing/timing out never blocks or unwinds the other. The session
reaches "assessed" once BOTH paths have concluded, whatever their
individual outcomes — report composition then runs even if one or both
paths failed (report_payload.py degrades gracefully when the authoritative
path is absent, never fabricating a band). For cloud/mixed, "assessed" is
a transient checkpoint, not a park: `process_session_step` immediately
advances it into "reporting", then attempts the report push (bounded
retries, mirroring the PDM path) until the session reaches the terminal
"reported" state or exhausts its retry budget into "failed".

The edge profile takes a
parallel but distinct path through the same state names: RECEIVED ->
ASSESSING runs the single score-band path (`_attempt_edge_score_band_path`,
no PDM/fuzzy/Gatekeeper call,) instead of the PDM+fuzzy pair; ASSESSED
is transient there too, but advances into edge-local report GENERATION
(`_attempt_edge_report`, rendering a PDF to local disk) rather than
cloud/mixed's REPORTING network push -- there is no edge REPORTING state,
since local rendering is not a bounded-retry network call to model as one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import DeploymentProfile, Settings, get_settings
from contract import ReportingEnvelope, RiskInput, SessionSubmission, check_contract_version
from edge_report import EdgeReportError, generate_edge_local_report
from forecast_builder import build_forecast_outlook
from forecast_outlook import ForecastResult
from fusion import FUSION_RULE_VERSION, UnknownBandError, nn_verdict as fusion_nn_verdict, select_fusion_rule
from fuzzy_client import FuzzyAssessmentResult, assess_via_fuzzy_engine
from gatekeeper import GatekeeperAuthError
from logging_utils import session_id_var
from models import (
    AttemptLog,
    CutoverBoundary,
    IdempotencyKey,
    ReportIndexRecord,
    RiskAssessmentRecord,
    SessionRecord,
    SessionState,
)
from pdm_client import PdmAssessmentResult, PdmCallError, assess_via_pdm
from report_payload import DESIGNATED_STATUS, compose_report_payload

logger = logging.getLogger("orchestrator.service")
from reporting_client import ReportingCallError, ReportPushResult, push_report
import sync_queue

PdmCallFn = Callable[[Settings, RiskInput], PdmAssessmentResult]
FuzzyCallFn = Callable[..., FuzzyAssessmentResult]
ReportCallFn = Callable[[Settings, dict], ReportPushResult]
ForecastCallFn = Callable[..., ForecastResult]

_PARKED_STATES = frozenset({SessionState.REPORTED.value, SessionState.FAILED.value})

_ASSESSED_PARK_REASON = (
    "awaiting-reporting (transient checkpoint — advance_session continues "
    "automatically into the reporting state on its next iteration; visible "
    "only to callers polling process_session_step one call at a time)"
)

_EDGE_ASSESSED_PARK_REASON = (
    "awaiting-edge-report-generation (transient checkpoint — "
    "advance_session continues automatically into local report generation "
    "on its next iteration; visible only to callers polling "
    "process_session_step one call at a time)"
)

_EDGE_SCORE_BAND_PATH = "edge-score-band"

EDGE_SYNC_ORIGIN = "edge-sync"


def _assessment_mode_for(session: SessionRecord) -> str:
    """`retroactive` for a synced edge session re-assessed in the cloud (its
    weather is the recorded weather of the session window, not a live
    reading), `live` for every other session."""
    return "retroactive" if session.origin == EDGE_SYNC_ORIGIN else "live"


def _reassessment_of(session: SessionRecord) -> Optional[str]:
    """The id of the on-device report a synced edge session re-assesses, else
    `None`."""
    return session.edge_report_id if session.origin == EDGE_SYNC_ORIGIN else None
_EDGE_SCORE_BAND_ENGINE_IDENTITY = "edge-score-band"
_EDGE_SCORE_BAND_ENGINE_VERSION = "v1"
_EDGE_SCORE_BAND_DETAILS = (
    "Score-only/degraded risk assessment (weather not applicable) - "
    "derived from the inference score alone; no weather-rule provenance."
)


def _parked_states(settings: Settings) -> frozenset[str]:
    return _PARKED_STATES


def content_hash_for(submission: SessionSubmission) -> str:
    """Deterministic hash of the submission's semantic content (not its
    wire bytes — key order-independent), used for intake idempotency."""
    canonical = json.dumps(submission.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_hash_for_payload(payload: dict) -> str:
    """Same construction as `content_hash_for`, over the composed report
    payload dict instead of a `SessionSubmission` — the report index's content hash."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_fusion_rule(settings: Settings):
    """`getattr`, not `settings.fusion_rule_version` -- several existing
    tests drive this module with a minimal fake Settings stand-in that only
    defines the fields their scenario needs; v1 (max-of-bands) is the default for
    any settings object that doesn't carry the field, matching `Settings`'
    own default. The single place the paths below resolve which fusion rule to call."""
    return select_fusion_rule(getattr(settings, "fusion_rule_version", FUSION_RULE_VERSION))


def compute_weather_window(risk_input: RiskInput) -> str:
    """The single shared weather-window reference — derived once from the risk input, identically for every risk
    path, independent of what any individual engine's response contains."""
    return f"{risk_input.time_start}  →  {risk_input.time_end}"


@dataclass
class IntakeOutcome:
    session_id: str
    state: str
    duplicate: bool
    errors: list[str]


class ContractVersionRejected(ValueError):
    def __init__(self, warning: str):
        super().__init__(warning)
        self.warning = warning


class IdempotencyConflict(ValueError):
    """Same session_id, different content — not the duplicate-retry case
    the intake contract covers."""


def intake_submission(
    db: Session,
    submission: SessionSubmission,
    origin: Optional[str] = None,
    edge_report_id: Optional[str] = None,
) -> IntakeOutcome:
    """Records a submission once. `origin`/`edge_report_id` mark a synced edge
    session submitted for re-assessment and are stored on the new record; a
    duplicate submission leaves the stored values as they are."""
    session_id_var.set(submission.risk_input.session_id)
    version_check = check_contract_version(submission.contract_version)
    if not version_check.accepted:
        raise ContractVersionRejected(version_check.warning or "unsupported contract_version")
    errors = [version_check.warning] if version_check.warning else []

    session_id = submission.risk_input.session_id
    new_hash = content_hash_for(submission)

    existing_key = db.scalar(select(IdempotencyKey).where(IdempotencyKey.session_id == session_id))
    if existing_key is not None:
        if existing_key.content_hash != new_hash:
            raise IdempotencyConflict(
                f"session_id {session_id!r} already submitted with different content"
            )
        existing_record = db.scalar(select(SessionRecord).where(SessionRecord.session_id == session_id))
        state = existing_record.state if existing_record else SessionState.RECEIVED.value
        logger.info("intake: duplicate submission acknowledged, existing state=%s", state)
        return IntakeOutcome(session_id=session_id, state=state, duplicate=True, errors=errors)

    db.add(IdempotencyKey(session_id=session_id, content_hash=new_hash))
    db.add(
        SessionRecord(
            session_id=session_id,
            contract_version=submission.contract_version,
            state=SessionState.RECEIVED.value,
            risk_input_json=submission.risk_input.model_dump_json(),
            reporting_envelope_json=submission.reporting_envelope.model_dump_json(),
            origin=origin,
            edge_report_id=edge_report_id,
        )
    )
    db.commit()

    logger.info("intake: new session recorded, state=%s", SessionState.RECEIVED.value)
    return IntakeOutcome(session_id=session_id, state=SessionState.RECEIVED.value, duplicate=False, errors=errors)


def _record_attempt(
    db: Session, session_id: str, call_type: str, attempt_number: int, status: str, error: Optional[str]
) -> None:
    if status == "succeeded":
        logger.info("outbound call %s attempt %d: succeeded", call_type, attempt_number)
    else:
        logger.warning("outbound call %s attempt %d: failed: %s", call_type, attempt_number, error)
    db.add(
        AttemptLog(
            session_id=session_id,
            call_type=call_type,
            attempt_number=attempt_number,
            status=status,
            error=error,
        )
    )


def _path_concluded(db: Session, session_id: str, path: str) -> bool:
    """A path has "concluded" once it has a `RiskAssessmentRecord` row
    (status "present" or "failed") — the marker `process_session_step` uses
    to decide whether a path still needs attempting (see module docstring)."""
    return (
        db.scalar(
            select(RiskAssessmentRecord).where(
                RiskAssessmentRecord.session_id == session_id,
                RiskAssessmentRecord.path == path,
            )
        )
        is not None
    )


def _attempt_pdm_path(
    db: Session,
    session: SessionRecord,
    settings: Settings,
    risk_input: RiskInput,
    pdm_call: PdmCallFn,
) -> SessionRecord:
    attempt_number = session.pdm_attempt_count + 1
    try:
        result = pdm_call(settings, risk_input)
        raw_level = result.raw_level
        weather_side = raw_level if raw_level is not None else fusion_nn_verdict(risk_input.infestation_score)
        fused = resolve_fusion_rule(settings)(risk_input.infestation_score, weather_side)
    except (PdmCallError, GatekeeperAuthError, UnknownBandError) as exc:
        session.pdm_attempt_count = attempt_number
        _record_attempt(db, session.session_id, "pdm_assess", attempt_number, "failed", str(exc))
        if attempt_number >= settings.max_outbound_retries:
            db.add(
                RiskAssessmentRecord(
                    session_id=session.session_id,
                    path="pdm",
                    status="failed",
                    failure_reason=str(exc),
                    assessment_mode=_assessment_mode_for(session),
                )
            )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    session.pdm_attempt_count = attempt_number
    _record_attempt(db, session.session_id, "pdm_assess", attempt_number, "succeeded", None)
    db.add(
        RiskAssessmentRecord(
            session_id=session.session_id,
            path="pdm",
            status="present",
            assessment_mode=_assessment_mode_for(session),
            level=raw_level,
            details=result.details,
            provenance_json=json.dumps(result.provenance),
            engine_identity="pdm-rule-engine",
            engine_version=result.provenance.get("pdm_app_version"),
            weather_window=compute_weather_window(risk_input),
            fused_verdict=fused.risk_assessment_level,
            fusion_rule_version=fused.fusion_rule_version,
        )
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def fuzzy_pest_kwargs(session: SessionRecord) -> dict[str, str]:
    """The session's pest (an EPPO code on its stored reporting envelope) as
    the fuzzy call's keyword argument; empty when the session carries none."""
    pest = ReportingEnvelope.model_validate_json(session.reporting_envelope_json).pest
    return {"pest": pest} if pest else {}


def _attempt_fuzzy_path(
    db: Session,
    session: SessionRecord,
    settings: Settings,
    risk_input: RiskInput,
    fuzzy_call: FuzzyCallFn,
) -> SessionRecord:
    """Single attempt, no retry budget: unlike PDM's network call, the
    fuzzy path is a pure in-process computation (plus one network hop to the Weather service through the Gatekeeper) — retrying identical inputs would just reproduce an identical
    failure. Always called only once PDM has already concluded (see
    `process_session_step`), and always parks the session in the same
    commit it concludes in, whatever its own outcome."""
    try:
        result = fuzzy_call(settings, risk_input, **fuzzy_pest_kwargs(session))
        fused = resolve_fusion_rule(settings)(risk_input.infestation_score, result.dominant_band)
    except Exception as exc:  # noqa: BLE001
        _record_attempt(db, session.session_id, "fuzzy_assess", 1, "failed", str(exc))
        db.add(
            RiskAssessmentRecord(
                session_id=session.session_id,
                path="fuzzy",
                status="failed",
                failure_reason=str(exc),
                assessment_mode=_assessment_mode_for(session),
                weather_window=compute_weather_window(risk_input),
            )
        )
    else:
        _record_attempt(db, session.session_id, "fuzzy_assess", 1, "succeeded", None)
        db.add(
            RiskAssessmentRecord(
                session_id=session.session_id,
                path="fuzzy",
                status="present",
                assessment_mode=_assessment_mode_for(session),
                level=result.dominant_band,
                details=(
                    f"Fuzzy Mamdani environmental pressure over "
                    f"{len(result.daily_scores)} day(s); dominant band "
                    f"{result.dominant_band}."
                ),
                provenance_json=json.dumps(
                    {
                        "daily_scores": result.daily_scores,
                        "threat_definition_version": result.threat_definition_version,
                        "dossier_version": result.dossier_version,
                        "vision_score_seen": False,
                        "weather_summary": {
                            "temperature_min_C": result.weather_summary.temperature_min_C,
                            "temperature_max_C": result.weather_summary.temperature_max_C,
                            "relative_humidity_avg_pct": result.weather_summary.relative_humidity_avg_pct,
                            "source": result.weather_summary.source,
                        },
                        "explanation": result.explanation,
                    }
                ),
                engine_identity="fuzzy-mamdani-risk",
                engine_version=result.engine_version,
                weather_window=compute_weather_window(risk_input),
                fused_verdict=fused.risk_assessment_level,
                fusion_rule_version=fused.fusion_rule_version,
            )
        )

    session.state = SessionState.ASSESSED.value
    session.park_reason = _ASSESSED_PARK_REASON
    logger.info("state transition: assessing -> assessed")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _attempt_edge_score_band_path(
    db: Session,
    session: SessionRecord,
    settings: Settings,
    risk_input: RiskInput,
) -> SessionRecord:
    """The edge profile's sole risk path, porting PDM's edge semantics. No PDM, no
    fuzzy, no Gatekeeper call of any kind: pure in-process computation
    from the session's own `infestation_score`, mirroring PDM's own
    weather-not-applicable degraded path but ported here since PDM itself is
    absent on the edge. Single attempt, no retry budget -- same rationale
    as `_attempt_fuzzy_path`: this is a pure computation, not a network
    call, so retrying identical inputs would just reproduce an identical
    result. Self-fused (mirrors `_attempt_pdm_path`'s own degraded-PDM
    precedent): with no weather-side signal to fuse against, the NN verdict
    is fused against itself, which reduces to the NN-only band without
    fabricating a weather reading."""
    weather_side = fusion_nn_verdict(risk_input.infestation_score)
    fused = resolve_fusion_rule(settings)(risk_input.infestation_score, weather_side)

    _record_attempt(db, session.session_id, "edge_score_band_assess", 1, "succeeded", None)
    db.add(
        RiskAssessmentRecord(
            session_id=session.session_id,
            path=_EDGE_SCORE_BAND_PATH,
            status="present",
            level=None,
            details=_EDGE_SCORE_BAND_DETAILS,
            provenance_json=json.dumps(
                {
                    "weather_rule_based": False,
                    "degraded_reason": "weather_not_applicable",
                    "weather_window": None,
                    "weather_rule_level_raw": None,
                    "edge_local": True,
                }
            ),
            engine_identity=_EDGE_SCORE_BAND_ENGINE_IDENTITY,
            engine_version=_EDGE_SCORE_BAND_ENGINE_VERSION,
            weather_window=None,
            fused_verdict=fused.risk_assessment_level,
            fusion_rule_version=fused.fusion_rule_version,
        )
    )
    session.state = SessionState.ASSESSED.value
    session.park_reason = _EDGE_ASSESSED_PARK_REASON
    logger.info("state transition: assessing -> assessed (edge score-band)")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _attempt_edge_report(
    db: Session,
    session: SessionRecord,
    settings: Settings,
) -> SessionRecord:
    """Edge-local report generation at
    session close -- the edge analogue of `_attempt_report`. Single attempt
    (local rendering to disk, not a bounded-retry network call): a failure
    here is a whole-session outcome (mirrors `_attempt_report`'s own "a
    report that never lands is a whole-session outcome" rule), moving the
    session straight to FAILED rather than retrying an I/O failure that
    retrying is unlikely to fix."""
    risk_input = RiskInput.model_validate_json(session.risk_input_json)
    envelope = ReportingEnvelope.model_validate_json(session.reporting_envelope_json)
    edge_assessment = db.scalar(
        select(RiskAssessmentRecord).where(
            RiskAssessmentRecord.session_id == session.session_id,
            RiskAssessmentRecord.path == _EDGE_SCORE_BAND_PATH,
        )
    )

    try:
        result = generate_edge_local_report(
            settings, session.session_id, risk_input, envelope, edge_assessment, session.contract_version
        )
    except EdgeReportError as exc:
        _record_attempt(db, session.session_id, "edge_report_generate", 1, "failed", str(exc))
        session.state = SessionState.FAILED.value
        session.failure_reason = str(exc)
        session.park_reason = None
        logger.warning("state transition: assessed -> failed: edge report generation failed: %s", exc)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    _record_attempt(db, session.session_id, "edge_report_generate", 1, "succeeded", None)
    db.add(
        ReportIndexRecord(
            session_id=session.session_id,
            report_id=None,
            content_hash=result.content_hash,
            origin="edge-local",
            state="generated",
        )
    )
    session.state = SessionState.REPORTED.value
    session.park_reason = None
    logger.info("state transition: assessed -> reported (edge-local): %s", result.pdf_path)
    db.add(session)
    db.commit()
    db.refresh(session)

    sync_queue.enqueue(
        settings.sync_queue_dir,
        session.session_id,
        result.sync_path,
        result.sync_sha256,
        result.pdf_path,
        result.content_hash,
        pair_path=result.pair_path,
    )

    return session


def _attempt_report(
    db: Session,
    session: SessionRecord,
    settings: Settings,
    report_call: ReportCallFn,
    forecast_call: ForecastCallFn = build_forecast_outlook,
) -> SessionRecord:
    """Single bounded-retry outbound call, mirroring
    `_attempt_pdm_path`'s pattern: compose the full payload fresh each
    attempt (cheap, pure, and always reflects the two RiskAssessmentRecord
    rows as currently stored), push it, and record the ReportIndexRecord on
    success. Exhaustion moves the SESSION itself to FAILED — a report that never reaches the Reporting service is
    a whole-session outcome, not one isolable component of it."""
    attempt_number = session.report_attempt_count + 1
    risk_input = RiskInput.model_validate_json(session.risk_input_json)
    envelope = ReportingEnvelope.model_validate_json(session.reporting_envelope_json)
    pdm_row = db.scalar(
        select(RiskAssessmentRecord).where(
            RiskAssessmentRecord.session_id == session.session_id,
            RiskAssessmentRecord.path == "pdm",
        )
    )
    fuzzy_row = db.scalar(
        select(RiskAssessmentRecord).where(
            RiskAssessmentRecord.session_id == session.session_id,
            RiskAssessmentRecord.path == "fuzzy",
        )
    )
    forecast: Optional[ForecastResult] = None
    if envelope.provenance.designation_status == DESIGNATED_STATUS:
        forecast = forecast_call(
            settings,
            risk_input.latitude,
            risk_input.longitude,
            risk_input.crop,
            envelope.pest,
            time_end=risk_input.time_end,
        )
    payload = compose_report_payload(
        risk_input=risk_input,
        envelope=envelope,
        pdm_assessment=pdm_row,
        fuzzy_assessment=fuzzy_row,
        contract_version=session.contract_version,
        site_label=getattr(settings, "report_site_label", None),
        forecast=forecast,
        reassessment_of=_reassessment_of(session),
    )
    content_hash = content_hash_for_payload(payload)

    try:
        result = report_call(settings, payload)
    except ReportingCallError as exc:
        session.report_attempt_count = attempt_number
        _record_attempt(db, session.session_id, "report_push", attempt_number, "failed", str(exc))
        if attempt_number >= settings.max_outbound_retries:
            session.state = SessionState.FAILED.value
            session.failure_reason = str(exc)
            logger.warning("state transition: reporting -> failed: retry budget exhausted: %s", exc)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    session.report_attempt_count = attempt_number
    _record_attempt(db, session.session_id, "report_push", attempt_number, "succeeded", None)
    db.add(
        ReportIndexRecord(
            session_id=session.session_id,
            report_id=result.report_id,
            content_hash=content_hash,
            origin="edge-reassessed" if session.origin == EDGE_SYNC_ORIGIN else "cloud-generated",
            state="reported",
        )
    )
    session.state = SessionState.REPORTED.value
    session.park_reason = None
    logger.info("state transition: reporting -> reported, report_id=%s", result.report_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def process_session_step(
    db: Session,
    session: SessionRecord,
    settings: Optional[Settings] = None,
    pdm_call: PdmCallFn = assess_via_pdm,
    fuzzy_call: FuzzyCallFn = assess_via_fuzzy_engine,
    report_call: ReportCallFn = push_report,
    forecast_call: ForecastCallFn = build_forecast_outlook,
) -> SessionRecord:
    """One bounded unit of work. Safe to call repeatedly (including after a
    crash/restart) on the same session — see module docstring."""
    settings = settings or get_settings()
    session_id_var.set(session.session_id)

    if session.state == SessionState.RECEIVED.value:
        session.state = SessionState.ASSESSING.value
        logger.info("state transition: received -> assessing")
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    if session.state == SessionState.ASSESSING.value:
        risk_input = RiskInput.model_validate_json(session.risk_input_json)

        if getattr(settings, "deployment_profile", DeploymentProfile.CLOUD) is DeploymentProfile.EDGE:
            if not _path_concluded(db, session.session_id, _EDGE_SCORE_BAND_PATH):
                return _attempt_edge_score_band_path(db, session, settings, risk_input)
            session.state = SessionState.ASSESSED.value
            session.park_reason = _EDGE_ASSESSED_PARK_REASON
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

        if not _path_concluded(db, session.session_id, "pdm"):
            return _attempt_pdm_path(db, session, settings, risk_input, pdm_call)

        if not _path_concluded(db, session.session_id, "fuzzy"):
            return _attempt_fuzzy_path(db, session, settings, risk_input, fuzzy_call)

        session.state = SessionState.ASSESSED.value
        session.park_reason = _ASSESSED_PARK_REASON
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    if session.state == SessionState.ASSESSED.value:
        if getattr(settings, "deployment_profile", DeploymentProfile.CLOUD) is DeploymentProfile.EDGE:
            return _attempt_edge_report(db, session, settings)
        session.state = SessionState.REPORTING.value
        logger.info("state transition: assessed -> reporting")
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    if session.state == SessionState.REPORTING.value:
        return _attempt_report(db, session, settings, report_call, forecast_call)

    return session


def _once(call: ForecastCallFn) -> ForecastCallFn:
    """`call` answering every later call with its first result."""
    answered: list = []

    def wrapper(*args, **kwargs):
        if not answered:
            answered.append(call(*args, **kwargs))
        return answered[0]

    return wrapper


def advance_session(
    db: Session,
    session_id: str,
    settings: Optional[Settings] = None,
    pdm_call: PdmCallFn = assess_via_pdm,
    fuzzy_call: FuzzyCallFn = assess_via_fuzzy_engine,
    report_call: ReportCallFn = push_report,
    forecast_call: ForecastCallFn = build_forecast_outlook,
) -> SessionRecord:
    """Drive a session forward until it reaches a parked/terminal state. A
    session can stay in ASSESSING across several calls: up to
    `max_outbound_retries` for the PDM path (process_session_step records
    its exhaustion on the PDM RiskAssessmentRecord, not the session — see
    module docstring), then exactly one more for the fuzzy path's single
    attempt, one more to advance ASSESSED -> REPORTING, then up to
    `max_outbound_retries` more for the report push — so the loop cannot use
    "state changed" as its halting condition, only "state is parked yet".
    The iteration cap is a termination safety net, not the driver:
    process_session_step's own per-step bounds are what guarantee the
    session concludes within a bounded number of calls."""
    session = db.scalar(select(SessionRecord).where(SessionRecord.session_id == session_id))
    if session is None:
        raise LookupError(f"no session record for session_id={session_id!r}")

    settings = settings or get_settings()
    forecast_call = _once(forecast_call)
    max_iterations = 2 * settings.max_outbound_retries + 6
    parked_states = _parked_states(settings)
    for _ in range(max_iterations):
        session = process_session_step(
            db, session, settings=settings, pdm_call=pdm_call, fuzzy_call=fuzzy_call,
            report_call=report_call, forecast_call=forecast_call,
        )
        if session.state in parked_states:
            break
    return session


def get_cutover_boundary(db: Session) -> Optional[CutoverBoundary]:
    return db.scalar(select(CutoverBoundary).order_by(CutoverBoundary.id))


def list_evaluable_sessions(db: Session) -> list[SessionRecord]:
    """An evaluation-shaped query that excludes
    pre-cutover-boundary sessions."""
    boundary = get_cutover_boundary(db)
    stmt = select(SessionRecord).order_by(SessionRecord.created_at)
    if boundary is not None:
        stmt = stmt.where(SessionRecord.created_at >= boundary.boundary_at)
    return list(db.scalars(stmt))
