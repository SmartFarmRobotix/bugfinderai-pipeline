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

"""Initial schema: session records, dual-assessment table, report index,
idempotency keys, cutover boundary

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-08-11 00:00:00

"""
import datetime
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "idempotency_key",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False, unique=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_idempotency_key_session_id", "idempotency_key", ["session_id"])

    op.create_table(
        "session_record",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False, unique=True),
        sa.Column("contract_version", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("risk_input_json", sa.Text(), nullable=False),
        sa.Column("reporting_envelope_json", sa.Text(), nullable=False),
        sa.Column("pdm_attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_session_record_session_id", "session_record", ["session_id"])

    op.create_table(
        "risk_assessment_record",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("session_record.session_id"), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("level", sa.String(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("provenance_json", sa.Text(), nullable=True),
        sa.Column("engine_identity", sa.String(), nullable=True),
        sa.Column("engine_version", sa.String(), nullable=True),
        sa.Column("weather_window", sa.String(), nullable=True),
        sa.Column("fused_verdict", sa.String(), nullable=True),
        sa.Column("fusion_rule_version", sa.String(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "path", name="uq_risk_assessment_session_path"),
    )
    op.create_index("ix_risk_assessment_record_session_id", "risk_assessment_record", ["session_id"])

    op.create_table(
        "report_index_record",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("session_record.session_id"), nullable=False),
        sa.Column("report_id", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("origin", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "content_hash", name="uq_report_index_session_hash"),
    )
    op.create_index("ix_report_index_record_session_id", "report_index_record", ["session_id"])

    op.create_table(
        "attempt_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("session_record.session_id"), nullable=False),
        sa.Column("call_type", sa.String(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_attempt_log_session_id", "attempt_log", ["session_id"])

    op.create_table(
        "cutover_boundary",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("boundary_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_ref", sa.String(), nullable=False),
    )

    cutover_boundary = sa.table(
        "cutover_boundary",
        sa.column("id", sa.Integer()),
        sa.column("boundary_at", sa.DateTime(timezone=True)),
        sa.column("decision_ref", sa.String()),
    )
    op.bulk_insert(
        cutover_boundary,
        [
            {
                "id": 1,
                "boundary_at": datetime.datetime.now(datetime.timezone.utc),
                "decision_ref": "initial-schema",
            }
        ],
    )


def downgrade() -> None:
    op.drop_table("cutover_boundary")
    op.drop_index("ix_attempt_log_session_id", table_name="attempt_log")
    op.drop_table("attempt_log")
    op.drop_index("ix_report_index_record_session_id", table_name="report_index_record")
    op.drop_table("report_index_record")
    op.drop_index("ix_risk_assessment_record_session_id", table_name="risk_assessment_record")
    op.drop_table("risk_assessment_record")
    op.drop_index("ix_session_record_session_id", table_name="session_record")
    op.drop_table("session_record")
    op.drop_index("ix_idempotency_key_session_id", table_name="idempotency_key")
    op.drop_table("idempotency_key")
