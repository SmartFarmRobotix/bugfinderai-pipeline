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

"""Unit tests for the versioned session contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.contract import (
    CONTRACT_VERSION,
    ConversionChainProvenance,
    DetectorProvenance,
    GrowthStage,
    PerImageResult,
    ReportingEnvelope,
    RiskInput,
    SessionSubmission,
    check_contract_version,
)


def _risk_input(**overrides):
    fields = dict(
        session_id="sess-1",
        infestation_score=0.42,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    fields.update(overrides)
    return RiskInput(**fields)


def _provenance(**overrides) -> DetectorProvenance:
    fields = dict(
        detector_impl_id="lookup-detector",
        detector_version="1.0.0",
        designation_status="stub",
        aggregation_rule_version="topfrac-v1",
        target_class_set=["brassica_seedling"],
        confidence_floor=0.3,
    )
    fields.update(overrides)
    return DetectorProvenance(**fields)


def _envelope(**overrides) -> ReportingEnvelope:
    fields = dict(
        session_id="sess-1",
        best_image_id="img-1",
        image_count=3,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=_provenance(),
    )
    fields.update(overrides)
    return ReportingEnvelope(**fields)


class TestGrowthStage:
    def test_bbch_range_valid(self):
        gs = GrowthStage(representation="bbch_range", bbch_min=10, bbch_max=19)
        assert gs.bbch_min == 10 and gs.bbch_max == 19

    def test_bbch_range_min_greater_than_max_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="bbch_range", bbch_min=20, bbch_max=10)

    def test_bbch_range_missing_bounds_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="bbch_range")

    def test_bbch_range_with_stage_id_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="bbch_range", bbch_min=1, bbch_max=2, stage_id="BBCH19")

    def test_enum_valid(self):
        gs = GrowthStage(representation="enum", stage_id="flowering")
        assert gs.stage_id == "flowering"

    def test_enum_missing_stage_id_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="enum")

    def test_enum_with_bbch_fields_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="enum", stage_id="x", bbch_min=1)

    def test_unknown_bbch_range_has_no_values(self):
        gs = GrowthStage.unknown_for("bbch_range")
        assert gs.unknown is True
        assert gs.bbch_min is None and gs.bbch_max is None

    def test_unknown_enum_has_no_stage_id(self):
        gs = GrowthStage.unknown_for("enum")
        assert gs.unknown is True
        assert gs.stage_id is None

    def test_unknown_with_bbch_values_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="bbch_range", unknown=True, bbch_min=1, bbch_max=2)

    def test_unknown_with_stage_id_rejected(self):
        with pytest.raises(ValidationError):
            GrowthStage(representation="enum", unknown=True, stage_id="x")


class TestRiskInput:
    def test_valid_risk_input(self):
        ri = _risk_input()
        assert ri.session_id == "sess-1"

    def test_score_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            _risk_input(infestation_score=1.5)
        with pytest.raises(ValidationError):
            _risk_input(infestation_score=-0.1)

    def test_score_bounds_inclusive(self):
        assert _risk_input(infestation_score=0.0).infestation_score == 0.0
        assert _risk_input(infestation_score=1.0).infestation_score == 1.0

    def test_coordinates_may_be_absent(self):
        risk_input = _risk_input(latitude=None, longitude=None)
        assert risk_input.latitude is None and risk_input.longitude is None

    def test_empty_session_id_rejected(self):
        with pytest.raises(ValidationError):
            _risk_input(session_id="")

    def test_rejects_best_image_id_field(self):
        """Risk input SHALL NOT contain best_image_id."""
        with pytest.raises(ValidationError):
            RiskInput(
                session_id="sess-1",
                infestation_score=0.1,
                crop="brassica-oleracea",
                growth_stage=GrowthStage.unknown_for("enum"),
                time_start="t0",
                time_end="t1",
                latitude=1.0,
                longitude=2.0,
                best_image_id="img-1",
            )

    def test_rejects_bounding_boxes_field(self):
        """Risk input SHALL NOT contain detection-internal fields."""
        with pytest.raises(ValidationError):
            RiskInput(
                session_id="sess-1",
                infestation_score=0.1,
                crop="brassica-oleracea",
                growth_stage=GrowthStage.unknown_for("enum"),
                time_start="t0",
                time_end="t1",
                latitude=1.0,
                longitude=2.0,
                bounding_boxes=[],
            )

    def test_growth_stage_required(self):
        with pytest.raises(ValidationError):
            RiskInput(
                session_id="sess-1",
                infestation_score=0.1,
                crop="brassica-oleracea",
                time_start="t0",
                time_end="t1",
                latitude=1.0,
                longitude=2.0,
            )


class TestPerImageResult:
    def test_position_and_provenance_are_optional_but_paired(self):
        positioned = PerImageResult(
            image_id="img-1",
            score=0.5,
            position={"latitude": 10.5, "longitude": 20.5},
            position_provenance="replay",
        )
        assert positioned.position_provenance == "replay"
        assert PerImageResult(image_id="img-2", score=0.0).position is None

    def test_position_without_provenance_is_rejected(self):
        with pytest.raises(ValidationError):
            PerImageResult(
                image_id="img-1",
                score=0.5,
                position={"latitude": 10.5, "longitude": 20.5},
            )


class TestReportingEnvelope:
    def test_valid_envelope(self):
        env = _envelope()
        assert env.best_image_id == "img-1"

    def test_coordinates_may_be_absent(self):
        envelope = _envelope(latitude=None, longitude=None)
        assert envelope.latitude is None and envelope.longitude is None

    def test_provenance_fields_present(self):
        env = _envelope()
        assert env.provenance.detector_impl_id == "lookup-detector"
        assert env.provenance.designation_status == "stub"

    def test_provenance_required(self):
        with pytest.raises(ValidationError):
            ReportingEnvelope(
                session_id="sess-1",
                best_image_id="img-1",
                image_count=1,
                time_start="t0",
                time_end="t1",
                latitude=1.0,
                longitude=2.0,
            )

    def test_bad_designation_status_rejected(self):
        with pytest.raises(ValidationError):
            _provenance(designation_status="not-a-real-status")

    def test_contract_is_at_1_5_0(self):
        assert CONTRACT_VERSION == "1.5.0"

    def test_pest_is_optional_and_absent_by_default(self):
        env = _envelope()
        assert env.pest is None
        assert "pest" not in env.model_dump(exclude_none=True)

    def test_an_eppo_code_pest_validates(self):
        assert _envelope(pest="BRVCBR").pest == "BRVCBR"
        assert _envelope(pest="A1").pest == "A1"

    @pytest.mark.parametrize(
        "bad", ["brevicoryne brassicae", "brvcbr", "", "TOOLONGCODE", "BRE-BRA", "BRE BRA"]
    )
    def test_a_malformed_pest_is_rejected(self, bad):
        with pytest.raises(ValidationError) as excinfo:
            _envelope(pest=bad)
        assert "pest" in str(excinfo.value)

    def test_risk_input_carries_no_pest(self):
        assert "pest" not in RiskInput.model_fields
        with pytest.raises(ValidationError):
            _risk_input(pest="BRVCBR")

    def test_conversion_chain_fields_default_absent(self):
        """Stub/lookup/null provenance omits the new fields; stays distinguishable."""
        prov = _provenance()
        assert prov.model_version is None
        assert prov.model_source is None
        assert prov.model_license is None
        assert prov.conversion_chain is None

    def test_execution_provider_populated_for_cpu_served_session(self):
        """a session served by a
        CPU-configured ORT detector stores execution_provider:
        'CPUExecutionProvider'."""
        prov = _provenance(
            detector_impl_id="dfine-ort-detector",
            designation_status="integration-only",
            execution_provider="CPUExecutionProvider",
        )
        assert prov.execution_provider == "CPUExecutionProvider"

    def test_execution_provider_defaults_absent_for_detector_without_provider_concept(self):
        """a detector with no
        provider concept (stub/lookup/null, and the Jetson TensorRT
        executor) omits execution_provider and still validates."""
        prov = _provenance()
        assert prov.execution_provider is None

    def test_unknown_field_still_rejected_extra_forbid_unchanged(self):
        """Adding execution_provider
        does not loosen DetectorProvenance's extra='forbid' — a genuinely
        unknown field is still rejected."""
        with pytest.raises(ValidationError):
            _provenance(not_a_real_field="CUDAExecutionProvider")

    def test_conversion_chain_provenance_populated_for_real_detector(self):
        chain = ConversionChainProvenance(
            execution_target="cloud",
            pt_version="dfine-l-stock-sha256:abc123",
            pt_source="dfine-stock-pretrained",
            pt_license="Apache-2.0",
            onnx_artifact="dfine-l-cloud.onnx",
            onnx_sha256="def456",
            engine_artifact="dfine-l-cloud-fp16.engine",
            engine_sha256="ghi789",
        )
        prov = _provenance(
            detector_impl_id="tensorrt-detector",
            designation_status="integration-only",
            model_version="dfine-l-stock",
            model_source="dfine-stock-pretrained",
            model_license="Apache-2.0",
            conversion_chain=chain,
        )
        assert prov.conversion_chain.execution_target == "cloud"
        assert prov.conversion_chain.engine_sha256 == "ghi789"

    def test_conversion_chain_bad_execution_target_rejected(self):
        with pytest.raises(ValidationError):
            ConversionChainProvenance(
                execution_target="not-a-target",
                pt_version="v1",
                pt_source="s",
                pt_license="Apache-2.0",
                onnx_artifact="a.onnx",
                onnx_sha256="x",
                engine_artifact="a.engine",
                engine_sha256="y",
            )

    def test_conversion_chain_onnx_terminal_record_validates_without_engine_fields(self):
        """Engine sub-fields are OPTIONAL — an ONNX Runtime
        dev/cloud session's conversion chain
        has no engine to record, and must still validate."""
        chain = ConversionChainProvenance(
            execution_target="cloud",
            pt_version="dfine-l-stock-sha256:abc123",
            pt_source="dfine-coco-stock",
            pt_license="Apache-2.0",
            onnx_artifact="dfine-l-cloud.onnx",
            onnx_sha256="def456",
        )
        assert chain.engine_artifact is None
        assert chain.engine_sha256 is None

        prov = _provenance(
            detector_impl_id="dfine-ort-detector",
            designation_status="integration-only",
            model_version="dfine-l-stock",
            model_source="dfine-coco-stock",
            model_license="Apache-2.0",
            conversion_chain=chain,
        )
        assert prov.conversion_chain.engine_artifact is None

    def test_conversion_chain_engine_terminal_record_still_validates(self):
        """The Jetson TensorRT session's engine-terminal chain (existing
        behavior) is unaffected by making the fields optional."""
        chain = ConversionChainProvenance(
            execution_target="jetson",
            pt_version="dfine-s-stock-sha256:abc123",
            pt_source="dfine-coco-stock",
            pt_license="Apache-2.0",
            onnx_artifact="dfine-s-jetson.onnx",
            onnx_sha256="def456",
            engine_artifact="dfine-s-jetson-fp16.engine",
            engine_sha256="ghi789",
        )
        assert chain.engine_artifact == "dfine-s-jetson-fp16.engine"
        assert chain.engine_sha256 == "ghi789"

    def test_conversion_chain_unpaired_engine_fields_rejected(self):
        """The engine stage is whole or absent: an artifact path without
        its verified hash (or vice versa) is an unauditable half-record —
        'optional' never means independently omittable."""
        base = dict(
            execution_target="jetson",
            pt_version="dfine-s-stock-sha256:abc123",
            pt_source="dfine-coco-stock",
            pt_license="Apache-2.0",
            onnx_artifact="dfine-s-jetson.onnx",
            onnx_sha256="def456",
        )
        with pytest.raises(ValidationError, match="together"):
            ConversionChainProvenance(**base, engine_artifact="orphan.engine")
        with pytest.raises(ValidationError, match="together"):
            ConversionChainProvenance(**base, engine_sha256="orphan-hash")


class TestSessionSubmission:
    def test_explicit_contract_version_round_trips(self):
        sub = SessionSubmission(
            contract_version=CONTRACT_VERSION, risk_input=_risk_input(), reporting_envelope=_envelope()
        )
        assert sub.contract_version == CONTRACT_VERSION

    def test_omitted_contract_version_is_none_not_defaulted(self):
        """No pydantic default: a consumer must be able to observe 'missing'."""
        sub = SessionSubmission(risk_input=_risk_input(), reporting_envelope=_envelope())
        assert sub.contract_version is None

    def test_risk_input_and_envelope_are_distinct_structures(self):
        sub = SessionSubmission(risk_input=_risk_input(), reporting_envelope=_envelope())
        risk_dict = sub.risk_input.model_dump()
        assert "best_image_id" not in risk_dict
        assert "provenance" not in risk_dict


class TestVersionCheck:
    def test_missing_version_accepted_with_warning(self):
        result = check_contract_version(None)
        assert result.accepted is True
        assert result.warning is not None

    def test_current_version_accepted_no_warning(self):
        result = check_contract_version(CONTRACT_VERSION)
        assert result.accepted is True
        assert result.warning is None

    def test_unknown_major_rejected(self):
        major = int(CONTRACT_VERSION.split(".", 1)[0])
        result = check_contract_version(f"{major + 1}.0.0")
        assert result.accepted is False

    def test_unparseable_version_rejected(self):
        result = check_contract_version("not-a-version")
        assert result.accepted is False
