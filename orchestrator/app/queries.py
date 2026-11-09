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
Query APIs for downstream consumers: session by id (with assessments and
report state), sessions by time range/scenario, and paired-assessment
retrieval — evaluation-shaped, so the evaluation tooling, the report path,
and operators never need direct database access.

`scenario` (cloud vs. mixed vs. edge) is not a column the store writes at
intake — no submitted field says which scenario produced a session. For
an edge session it is derived from deployment profile, not ingestion
mode: the presence of an `edge-score-band`
RiskAssessmentRecord (`service.py`'s `_attempt_edge_score_band_path`,
which only ever runs under `DeploymentProfile.EDGE`) is this session's
own record of having been processed on the edge profile -- authoritative,
not re-derived from execution_target, and checked BEFORE the
cloud/mixed conversion-chain heuristic below (mixed also executes on the
same Jetson hardware, so `execution_target == "jetson"` alone cannot
distinguish edge from mixed).

For cloud/mixed it is derived, best-effort, from the reporting envelope's
detector provenance (`DetectorProvenance.conversion_chain.
execution_target`, "cloud"/"jetson"): "cloud" execution -> the cloud scenario,
"jetson" execution -> the mixed scenario. A session whose envelope carries no
conversion chain (the stub/lookup detectors) reports "unknown" rather than guessing.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings
from models import AttemptLog, IdempotencyKey, ReportIndexRecord, RiskAssessmentRecord, SessionRecord
from reporting_client import fetch_report_status
from service import _EDGE_SCORE_BAND_PATH, get_cutover_boundary

logger = logging.getLogger("orchestrator.queries")


class SessionNotFound(LookupError):
    """No session record exists for the requested session_id."""


def _scenario_for(db: Session, session: SessionRecord) -> str:
    has_edge_score_band = db.scalar(
        select(RiskAssessmentRecord.id).where(
            RiskAssessmentRecord.session_id == session.session_id,
            RiskAssessmentRecord.path == _EDGE_SCORE_BAND_PATH,
        )
    )
    if has_edge_score_band is not None:
        return "edge"

    try:
        envelope = json.loads(session.reporting_envelope_json)
    except (TypeError, ValueError):
        return "unknown"
    conversion_chain = ((envelope.get("provenance") or {}).get("conversion_chain")) or {}
    execution_target = conversion_chain.get("execution_target")
    if execution_target == "cloud":
        return "cloud"
    if execution_target == "jetson":
        return "mixed"
    return "unknown"


@dataclass
class AssessmentView:
    path: str
    status: str
    level: Optional[str]
    fused_verdict: Optional[str]
    fusion_rule_version: Optional[str]
    engine_identity: Optional[str]
    engine_version: Optional[str]
    weather_window: Optional[str]
    failure_reason: Optional[str]
    assessment_mode: str

    @classmethod
    def from_record(cls, record: RiskAssessmentRecord) -> "AssessmentView":
        return cls(
            path=record.path,
            status=record.status,
            level=record.level,
            fused_verdict=record.fused_verdict,
            fusion_rule_version=record.fusion_rule_version,
            engine_identity=record.engine_identity,
            engine_version=record.engine_version,
            weather_window=record.weather_window,
            failure_reason=record.failure_reason,
            assessment_mode=record.assessment_mode,
        )


@dataclass
class ReportView:
    report_id: Optional[str]
    content_hash: Optional[str]
    origin: Optional[str]
    state: str
    images_pending: Optional[bool] = None

    @classmethod
    def from_record(cls, record: ReportIndexRecord) -> "ReportView":
        return cls(
            report_id=record.report_id,
            content_hash=record.content_hash,
            origin=record.origin,
            state=record.state,
        )


@dataclass
class SessionDetail:
    session_id: str
    state: str
    scenario: str
    contract_version: Optional[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    failure_reason: Optional[str]
    park_reason: Optional[str]
    assessments: Dict[str, AssessmentView] = field(default_factory=dict)
    report: Optional[ReportView] = None


def _assessments_for(db: Session, session_id: str) -> Dict[str, AssessmentView]:
    rows = db.scalars(select(RiskAssessmentRecord).where(RiskAssessmentRecord.session_id == session_id))
    return {row.path: AssessmentView.from_record(row) for row in rows}


def _report_for(db: Session, session_id: str) -> Optional[ReportView]:
    row = db.scalar(
        select(ReportIndexRecord)
        .where(ReportIndexRecord.session_id == session_id)
        .order_by(ReportIndexRecord.id.desc())
    )
    return ReportView.from_record(row) if row is not None else None


def _detail_from_record(db: Session, session: SessionRecord) -> SessionDetail:
    return SessionDetail(
        session_id=session.session_id,
        state=session.state,
        scenario=_scenario_for(db, session),
        contract_version=session.contract_version,
        created_at=session.created_at,
        updated_at=session.updated_at,
        failure_reason=session.failure_reason,
        park_reason=session.park_reason,
        assessments=_assessments_for(db, session.session_id),
        report=_report_for(db, session.session_id),
    )


def _get_session_record(db: Session, session_id: str) -> SessionRecord:
    session = db.scalar(select(SessionRecord).where(SessionRecord.session_id == session_id))
    if session is None:
        raise SessionNotFound(session_id)
    return session


def get_session_detail(
    db: Session,
    session_id: str,
    settings: Optional[Settings] = None,
    status_call: Callable[[Optional[Settings], str], Optional[bool]] = fetch_report_status,
) -> SessionDetail:
    """Session by id: state, provenance-derived scenario, both risk paths'
    assessments, and report state, in one call. For a
    cloud-generated report with an id, also queries the Reporting
    service's by-session status (`status_call`, stubbable by callers) for
    `report.images_pending` -- one GET per call, only when the record
    could actually still be pending. An edge-local/edge-synced report or
    no report at all leaves `images_pending` None without calling out.
    `status_call` is wrapped defensively: the production callable
    (`fetch_report_status`) never raises, but a status probe failure must
    never break the session detail read regardless."""
    detail = _detail_from_record(db, _get_session_record(db, session_id))
    if detail.report is not None and detail.report.report_id and detail.report.origin == "cloud-generated":
        try:
            detail.report.images_pending = status_call(settings, session_id)
        except Exception:  # noqa: BLE001 - a status probe failure must never break the session detail read
            logger.info("report status probe failed for session_id=%s", session_id, exc_info=True)
            detail.report.images_pending = None
    return detail


def list_sessions(
    db: Session,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    scenario: Optional[str] = None,
    exclude_pre_cutover: bool = True,
) -> List[SessionDetail]:
    """Sessions by time range/scenario. Excludes sessions created before the cutover boundary by default (the same
    filter as `service.list_evaluable_sessions`); pass
    `exclude_pre_cutover=False` for an operator's raw, unfiltered view."""
    stmt = select(SessionRecord).order_by(SessionRecord.created_at)
    if exclude_pre_cutover:
        boundary = get_cutover_boundary(db)
        if boundary is not None:
            stmt = stmt.where(SessionRecord.created_at >= boundary.boundary_at)
    if start is not None:
        stmt = stmt.where(SessionRecord.created_at >= start)
    if end is not None:
        stmt = stmt.where(SessionRecord.created_at <= end)

    details = [_detail_from_record(db, session) for session in db.scalars(stmt)]
    if scenario is not None:
        details = [d for d in details if d.scenario == scenario]
    return details


def get_paired_assessments(db: Session, session_id: str) -> Dict[str, AssessmentView]:
    """Paired-assessment retrieval — both paths' rows for one
    session, keyed by path; a path that never concluded is simply absent
    from the dict, not a fabricated placeholder."""
    _get_session_record(db, session_id)
    return _assessments_for(db, session_id)


def get_intake_state(db: Session, session_id: str) -> Dict[str, object]:
    """Whether ANY intake-side state
    exists for a session id, across every table intake and its downstream
    writers touch -- idempotency_key included. Deliberately independent of
    `get_session_detail`/`SessionNotFound`: an `idempotency_key` row can survive a
    partial state clear after `session_record` is removed -- a session-detail 404 must never be read as "clean" when
    a survivor sits in any of these tables. `held` is the OR of the five
    per-table booleans; a future new table joins this dict in the same
    change that adds the table."""
    state = {
        "idempotency_key": db.scalar(
            select(IdempotencyKey.id).where(IdempotencyKey.session_id == session_id)
        )
        is not None,
        "session_record": db.scalar(select(SessionRecord.id).where(SessionRecord.session_id == session_id))
        is not None,
        "risk_assessment_record": db.scalar(
            select(RiskAssessmentRecord.id).where(RiskAssessmentRecord.session_id == session_id)
        )
        is not None,
        "report_index_record": db.scalar(
            select(ReportIndexRecord.id).where(ReportIndexRecord.session_id == session_id)
        )
        is not None,
        "attempt_log": db.scalar(select(AttemptLog.id).where(AttemptLog.session_id == session_id)) is not None,
    }
    return {"session_id": session_id, "held": any(state.values()), "state": state}


def get_report_state(db: Session, session_id: str) -> Optional[ReportView]:
    """Report state for one session — `None` when no report has been
    generated (or successfully indexed) yet, never an error: "not reported
    yet" is a normal, queryable state, not a failure."""
    _get_session_record(db, session_id)
    return _report_for(db, session_id)
