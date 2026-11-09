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

"""The migration applies from empty; schema round-trip."""
from __future__ import annotations

import datetime
import os

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, text

from models import CutoverBoundary, IdempotencyKey, SessionRecord, SessionState


class TestMigrationAppliesFromEmpty:
    def test_all_tables_created(self, db):
        inspector = inspect(db.get_bind())
        table_names = set(inspector.get_table_names())
        assert table_names >= {
            "idempotency_key",
            "session_record",
            "risk_assessment_record",
            "report_index_record",
            "attempt_log",
            "cutover_boundary",
            "alembic_version",
        }

    def test_store_starts_empty(self, db):
        assert db.query(SessionRecord).count() == 0
        assert db.query(IdempotencyKey).count() == 0

    def test_cutover_boundary_seeded_by_migration(self, db):
        boundary = db.query(CutoverBoundary).one()
        assert boundary.decision_ref == "initial-schema"
        assert boundary.boundary_at is not None


class TestSchemaRoundTrip:
    def test_session_record_round_trip(self, db):
        record = SessionRecord(
            session_id="sess-roundtrip",
            contract_version="1.2.0",
            state=SessionState.RECEIVED.value,
            risk_input_json="{}",
            reporting_envelope_json="{}",
        )
        db.add(record)
        db.commit()

        reloaded = db.query(SessionRecord).filter_by(session_id="sess-roundtrip").one()
        assert reloaded.state == SessionState.RECEIVED.value
        assert isinstance(reloaded.created_at, datetime.datetime)


class TestSessionRecordOriginMigration:
    """Upgrading from the
    previous head adds the two nullable columns; rows already there read
    null in both; the downgrade drops them."""

    _PREVIOUS_HEAD = "d4e5f6a7b8c9"

    @staticmethod
    def _config(db_path) -> AlembicConfig:
        service_root = os.path.join(os.path.dirname(__file__), "..", "..")
        cfg = AlembicConfig(os.path.join(service_root, "alembic.ini"))
        cfg.set_main_option("script_location", os.path.join(service_root, "alembic"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        return cfg

    @staticmethod
    def _columns(db_path) -> dict:
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            return {c["name"]: c for c in inspect(engine).get_columns("session_record")}
        finally:
            engine.dispose()

    def test_upgrade_from_the_previous_head_adds_two_nullable_columns(self, tmp_path):
        db_path = tmp_path / "migrate.db"
        cfg = self._config(db_path)
        command.upgrade(cfg, self._PREVIOUS_HEAD)
        before = self._columns(db_path)
        assert "origin" not in before and "edge_report_id" not in before

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO session_record (session_id, contract_version, state, risk_input_json, "
                    "reporting_envelope_json, pdm_attempt_count, report_attempt_count, created_at, updated_at) "
                    "VALUES ('old', '1.5.0', 'received', '{}', '{}', 0, 0, '2026-01-01', '2026-01-01')"
                )
            )
        command.upgrade(cfg, "head")

        after = self._columns(db_path)
        assert after["origin"]["nullable"] and after["edge_report_id"]["nullable"]
        with engine.connect() as conn:
            row = conn.execute(text("SELECT origin, edge_report_id FROM session_record WHERE session_id='old'")).one()
        engine.dispose()
        assert tuple(row) == (None, None)

    def test_downgrade_drops_the_two_columns(self, tmp_path):
        db_path = tmp_path / "migrate-down.db"
        cfg = self._config(db_path)
        command.upgrade(cfg, "head")
        assert "origin" in self._columns(db_path)

        command.downgrade(cfg, self._PREVIOUS_HEAD)

        columns = self._columns(db_path)
        assert "origin" not in columns and "edge_report_id" not in columns
