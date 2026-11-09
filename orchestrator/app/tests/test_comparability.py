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

"""An evaluation-shaped comparability
check over the orchestrator's own query API — paired FUSED verdicts with raw components retrievable
for seeded sessions, and per-session agreement/divergence between the two
risk paths' fused verdicts computable directly from what the API returns.
Every assertion below goes through `queries.py`'s functions only, never a
raw `db.query(...)` of its own (matching test_queries.py's convention).

The comparison isolates the weather-risk producer only when the NN score
band doesn't dominate both paths on its own (fusion.py's module docstring):
every seeded session here uses a low `infestation_score` so the raw
weather-side band is what actually decides each path's fused verdict, and
any agreement/divergence observed is attributable to the PDM-vs-fuzzy
weather-risk producer, not to the score swamping both.

Storage and report presentation of the comparison are out of scope here: this test proves the
data needed for the comparison is retrievable and well-formed, not that a comparison report gets
rendered anywhere.
"""
from __future__ import annotations

from typing import Optional

from contract import (
    CONTRACT_VERSION,
    DetectorProvenance,
    GrowthStage,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
)
from fuzzy_client import FuzzyAssessmentError, FuzzyAssessmentResult, WeatherWindowSummary
from pdm_client import PdmAssessmentResult
from queries import AssessmentView, get_paired_assessments, list_sessions
from reporting_client import ReportPushResult
from service import advance_session, intake_submission


class _FakeSettings:
    max_outbound_retries = 3


def _submission(session_id: str) -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=0.05,
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


def _pdm(raw_level: str):
    def _call(settings, risk_input):
        return PdmAssessmentResult(
            level="ignored-by-orchestrator-fusion",
            details="ok",
            provenance={
                "weather_rule_level_raw": raw_level,
                "ruleset_version": "1.0.0",
                "pdm_app_version": "0.2.0",
            },
        )

    return _call


def _fuzzy(dominant_band: str):
    def _call(settings, risk_input):
        return FuzzyAssessmentResult(
            dominant_band=dominant_band,
            daily_scores=[{"date": "2026-01-01", "value": 10.0, "risk_level": dominant_band}],
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

    return _call


def _fuzzy_fails(settings, risk_input):
    raise FuzzyAssessmentError("Weather service unreachable (comparability test)")


def _report_succeeds(settings, payload):
    return ReportPushResult(report_id="report-cmp")


def _seed(db, session_id: str, pdm_call, fuzzy_call) -> None:
    intake_submission(db, _submission(session_id))
    advance_session(
        db,
        session_id,
        settings=_FakeSettings(),
        pdm_call=pdm_call,
        fuzzy_call=fuzzy_call,
        report_call=_report_succeeds,
    )


def _agreement(paired: dict[str, AssessmentView]) -> Optional[bool]:
    """The comparison itself: `None` when either path's fused verdict
    is absent (failed/never concluded) — divergence needs both sides
    present to actually compare, never a fabricated agreement/divergence
    against a missing one."""
    pdm = paired.get("pdm")
    fuzzy = paired.get("fuzzy")
    if pdm is None or fuzzy is None or pdm.fused_verdict is None or fuzzy.fused_verdict is None:
        return None
    return pdm.fused_verdict == fuzzy.fused_verdict


class TestPairedFusedVerdictsWithRawComponents:
    def test_agreeing_session_exposes_raw_components_and_agreement(self, db) -> None:
        _seed(db, "sess-cmp-agree", _pdm("moderate"), _fuzzy("moderate"))

        paired = get_paired_assessments(db, "sess-cmp-agree")
        assert paired["pdm"].level == "moderate"
        assert paired["fuzzy"].level == "moderate"
        assert paired["pdm"].fused_verdict == paired["fuzzy"].fused_verdict == "Medium"
        assert _agreement(paired) is True

    def test_diverging_session_exposes_raw_components_and_divergence(self, db) -> None:
        _seed(db, "sess-cmp-diverge", _pdm("low"), _fuzzy("high"))

        paired = get_paired_assessments(db, "sess-cmp-diverge")
        assert paired["pdm"].level == "low"
        assert paired["fuzzy"].level == "high"
        assert paired["pdm"].fused_verdict == "Low"
        assert paired["fuzzy"].fused_verdict == "High"
        assert _agreement(paired) is False

    def test_one_path_failed_is_not_a_fabricated_agreement(self, db) -> None:
        _seed(db, "sess-cmp-partial", _pdm("moderate"), _fuzzy_fails)

        paired = get_paired_assessments(db, "sess-cmp-partial")
        assert paired["pdm"].fused_verdict is not None
        assert paired["fuzzy"].status == "failed"
        assert paired["fuzzy"].fused_verdict is None
        assert _agreement(paired) is None


class TestAgreementRateComputableAcrossSeededSessions:
    def test_agreement_rate_over_a_seeded_population(self, db) -> None:
        _seed(db, "sess-cmp-pop-agree-1", _pdm("moderate"), _fuzzy("moderate"))
        _seed(db, "sess-cmp-pop-agree-2", _pdm("low"), _fuzzy("low"))
        _seed(db, "sess-cmp-pop-diverge-1", _pdm("low"), _fuzzy("high"))

        sessions = list_sessions(db)
        pop = [d for d in sessions if d.session_id.startswith("sess-cmp-pop-")]
        assert len(pop) == 3

        outcomes = [_agreement(d.assessments) for d in pop]
        assert outcomes.count(True) == 2
        assert outcomes.count(False) == 1
        assert None not in outcomes
