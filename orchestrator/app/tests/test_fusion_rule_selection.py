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
Both rules produce recorded verdicts with rule versions on seeded sessions, and the active rule
is visible in provenance -- exercised through `service.py`'s real wiring
(`resolve_fusion_rule`), not `fusion.py` in isolation (see test_fusion.py
for the rule functions' own behavior).
"""
from __future__ import annotations

from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fusion import FUSION_RULE_PRINCIPLED, FUSION_RULE_VERSION
from fuzzy_client import FuzzyAssessmentResult, WeatherWindowSummary
from models import RiskAssessmentRecord
from pdm_client import PdmAssessmentResult
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


class _FakeSettings:
    max_outbound_retries = 3
    fusion_rule_version = FUSION_RULE_VERSION


class _FakeSettingsPrincipled(_FakeSettings):
    fusion_rule_version = FUSION_RULE_PRINCIPLED


def _submission(session_id: str) -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=0.95,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    envelope = ReportingEnvelope(
        session_id=session_id,
        best_image_id="img-1",
        image_count=1,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status="stub",
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
        ),
    )
    return SessionSubmission(contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope)


def _pdm_succeeds(settings, risk_input):
    return PdmAssessmentResult(
        level="ignored-by-orchestrator-fusion",
        details="ok",
        provenance={
            "weather_rule_level_raw": "low",
            "ruleset_version": "1.0.0",
            "pdm_app_version": "0.2.0",
        },
    )


def _fuzzy_succeeds(settings, risk_input):
    return FuzzyAssessmentResult(
        dominant_band="low",
        daily_scores=[{"date": "2026-01-01", "value": 5.0, "risk_level": "low"}],
        engine_version="2.0.0",
        threat_definition_version="1.0.0",
        dossier_version="v2",
        weather_summary=WeatherWindowSummary(
            temperature_min_C=10.0,
            temperature_max_C=20.0,
            relative_humidity_avg_pct=50.0,
            source="openmeteo",
        ),
    )


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-fusion-select")


def _run(db, session_id: str, settings) -> None:
    intake_submission(db, _submission(session_id))
    advance_session(
        db,
        session_id,
        settings=settings,
        pdm_call=_pdm_succeeds,
        fuzzy_call=_fuzzy_succeeds,
        report_call=_report_succeeds,
    )


class TestBothRulesProduceRecordedVerdictsOnSeededSessions:
    def test_v1_max_of_bands_selected_by_default(self, db) -> None:
        _run(db, "sess-fuse-v1", _FakeSettings())

        for path in ("pdm", "fuzzy"):
            row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-fuse-v1", path=path).one()
            assert row.fusion_rule_version == FUSION_RULE_VERSION
            assert row.fused_verdict == "High"

    def test_principled_rule_selected_via_settings_produces_a_different_recorded_verdict(self, db) -> None:
        _run(db, "sess-fuse-v2", _FakeSettingsPrincipled())

        for path in ("pdm", "fuzzy"):
            row = db.query(RiskAssessmentRecord).filter_by(session_id="sess-fuse-v2", path=path).one()
            assert row.fusion_rule_version == FUSION_RULE_PRINCIPLED
            assert row.fused_verdict == "Low"
