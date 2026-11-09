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
Persistence schema: session records,
dual-assessment table, report index, idempotency keys, and the legacy
cutover boundary.

Both the `pdm` and `fuzzy` risk paths are populated, each
with its fused verdict and fusion-rule version.
"""

from __future__ import annotations

import datetime
import enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class SessionState(str, enum.Enum):
    RECEIVED = "received"
    ASSESSING = "assessing"
    ASSESSED = "assessed"
    REPORTING = "reporting"
    REPORTED = "reported"
    FAILED = "failed"


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class IdempotencyKey(Base):
    """
    Persisted before any processing begins — the record that
    makes a re-delivered submission (same session_id + content hash)
    detectable as a duplicate rather than reprocessed.
    """

    __tablename__ = "idempotency_key"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class SessionRecord(Base):
    __tablename__ = "session_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    contract_version: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False, default=SessionState.RECEIVED.value)

    risk_input_json: Mapped[str] = mapped_column(Text, nullable=False)
    reporting_envelope_json: Mapped[str] = mapped_column(Text, nullable=False)

    pdm_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    report_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    park_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    origin: Mapped[str | None] = mapped_column(String, nullable=True)
    edge_report_id: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class RiskAssessmentRecord(Base):
    """
    One row per (session, path). `path` is "pdm" or "fuzzy". `level` carries the RAW, pre-score weather-side component
    for both paths: PDM's
    `provenance["weather_rule_level_raw"]` (null when PDM ran degraded — no
    weather-rule engine, nothing raw to store), or the fuzzy path's dominant
    Mamdani band. Never PDM's own already-score-combined `level` field —
    storing that here would count the vision score twice. `fused_verdict`/`fusion_rule_version` carry the shared fusion
    step's output, null only when `status` is "failed" (no band
    to fuse). `engine_version` is the assessing engine/app's own version
    (PDM's app version / `FuzzyMamdaniRisk`'s version) — other version
    metadata (PDM's `ruleset_version`, the fuzzy path's
    `threat_definition_version`/`dossier_version`) rides inside
    `provenance_json` instead of a dedicated column, mirroring how the
    fuzzy path already stores its own extra version fields there.
    """

    __tablename__ = "risk_assessment_record"
    __table_args__ = (UniqueConstraint("session_id", "path", name="uq_risk_assessment_session_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("session_record.session_id"), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    level: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    engine_identity: Mapped[str | None] = mapped_column(String, nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String, nullable=True)
    weather_window: Mapped[str | None] = mapped_column(String, nullable=True)

    fused_verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    fusion_rule_version: Mapped[str | None] = mapped_column(String, nullable=True)

    assessment_mode: Mapped[str] = mapped_column(String, nullable=False, default="live")

    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ReportIndexRecord(Base):
    """
    Report index. Populated by `service.py` on a successful report: `origin` is
    "cloud-generated", "edge-reassessed" or "edge-local".
    """

    __tablename__ = "report_index_record"
    __table_args__ = (UniqueConstraint("session_id", "content_hash", name="uq_report_index_session_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("session_record.session_id"), nullable=False, index=True)
    report_id: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    origin: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class AttemptLog(Base):
    """Recorded attempt history for outbound calls."""

    __tablename__ = "attempt_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("session_record.session_id"), nullable=False, index=True)
    call_type: Mapped[str] = mapped_column(String, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class CutoverBoundary(Base):
    """The store starts empty; this single row records when
    (boundary_at) and why (decision_ref) — seeded by the initial Alembic
    migration at apply time, never by application code, so the boundary
    reflects when the schema first became live, not a request timestamp.
    """

    __tablename__ = "cutover_boundary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    boundary_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision_ref: Mapped[str] = mapped_column(String, nullable=False)
