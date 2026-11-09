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

"""The evaluation query excludes pre-cutover sessions and reports the boundary."""
from __future__ import annotations

import datetime

from models import SessionRecord, SessionState
from service import get_cutover_boundary, list_evaluable_sessions


def _insert_session(db, session_id: str, created_at: datetime.datetime) -> None:
    record = SessionRecord(
        session_id=session_id,
        state=SessionState.RECEIVED.value,
        risk_input_json="{}",
        reporting_envelope_json="{}",
        created_at=created_at,
        updated_at=created_at,
    )
    db.add(record)
    db.commit()


class TestCutoverBoundary:
    def test_boundary_is_recorded_and_reportable(self, db):
        boundary = get_cutover_boundary(db)
        assert boundary is not None
        assert boundary.decision_ref == "initial-schema"

    def test_legacy_dated_session_excluded_from_evaluation(self, db):
        boundary = get_cutover_boundary(db)
        pre_cutover = boundary.boundary_at - datetime.timedelta(days=365)
        post_cutover = boundary.boundary_at + datetime.timedelta(seconds=1)

        _insert_session(db, "legacy-sess", pre_cutover)
        _insert_session(db, "current-sess", post_cutover)

        evaluable = list_evaluable_sessions(db)
        evaluable_ids = {s.session_id for s in evaluable}

        assert "legacy-sess" not in evaluable_ids
        assert "current-sess" in evaluable_ids

    def test_store_with_only_current_sessions_returns_everything(self, db):
        boundary = get_cutover_boundary(db)
        post_cutover = boundary.boundary_at + datetime.timedelta(seconds=1)
        _insert_session(db, "sess-a", post_cutover)

        evaluable = list_evaluable_sessions(db)
        assert {s.session_id for s in evaluable} == {"sess-a"}
