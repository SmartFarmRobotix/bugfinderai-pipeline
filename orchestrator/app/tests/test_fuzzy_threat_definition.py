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

"""BRVCBR ThreatDefinition v1.

The bundled agstack-pnd catalog has no Brassica entry.
This tests that the orchestrator's own definition, authored from published
BRVCBR agronomic sources, loads via a `get_threat`-equivalent path,
carries a version id, and cites its source version.
"""
from __future__ import annotations

import json

import pytest

from agstack_pnd.foundation.types import RiskLevel, ThreatType
from fuzzy_threats import FuzzyThreatNotFound, get_threat, get_threat_record, get_threat_record_by_eppo, get_threat_record_for_crop
from fuzzy_threats import loader


def test_get_threat_returns_engine_ready_definition() -> None:
    definition = get_threat("brassica-oleracea", "Brevicoryne brassicae")

    assert definition.scientific_name == "Brevicoryne brassicae"
    assert definition.common_name == "Cabbage aphid"
    assert definition.threat_type == ThreatType.INSECT
    assert definition.crop == "brassica-oleracea"
    assert len(definition.fuzzy_rules) >= 1


def test_get_threat_is_case_insensitive_like_upstream_get_threat() -> None:
    definition = get_threat("BRASSICA-OLERACEA", "brevicoryne brassicae")
    assert definition.scientific_name == "Brevicoryne brassicae"


def test_unknown_threat_raises() -> None:
    with pytest.raises(FuzzyThreatNotFound):
        get_threat("apple", "Nonexistent species")


def test_record_carries_version_and_dossier_citation() -> None:
    record = get_threat_record("brassica-oleracea", "Brevicoryne brassicae")

    assert record.version == "1.0.0"
    assert record.dossier == "literature (see _authoring_notes)"
    assert record.dossier_version == "v2"
    assert record.review_status == "awaiting-agronomic-review"


def test_bio_params_match_the_adopted_literature_values() -> None:
    definition = get_threat("brassica-oleracea", "Brevicoryne brassicae")
    bp = definition.bio_params

    assert bp.t_base == 4.0
    assert bp.t_optimal_min == 20.0
    assert bp.t_optimal_max == 25.0


def test_fuzzy_rules_never_treat_humidity_or_rain_as_risk_raising() -> None:
    """The sign of the humidity effect is contested in the literature;
    precipitation dampens rather than raises risk, which this engine's
    rain_min mechanism cannot express directly -- rain_min=0.0 everywhere
    is the honest non-risk-raising encoding (see the JSON's authoring notes
    for the documented limitation)."""
    definition = get_threat("brassica-oleracea", "Brevicoryne brassicae")

    for rule in definition.fuzzy_rules:
        assert rule.hum_lo == 0
        assert rule.hum_hi == 100
        assert rule.rain_min == 0.0


def test_no_fabricated_critical_tier() -> None:
    """The cited evidence supports low/moderate/high only -- no source
    distinguishes a more-severe-than-high band, and the PDM weather-rule
    ruleset independently reaches the same three tiers."""
    definition = get_threat("brassica-oleracea", "Brevicoryne brassicae")
    levels = {rule.risk_level for rule in definition.fuzzy_rules}
    assert RiskLevel.CRITICAL not in levels


KALE = "brassica-oleracea-sabellica"


def test_kale_crop_resolves_to_the_brevicoryne_brassicae_record() -> None:
    record = get_threat_record_for_crop(KALE)

    assert record.eppo_code == "BRVCBR"
    assert record.definition.scientific_name == "Brevicoryne brassicae"
    assert record.definition.crop == KALE


def test_kale_record_differs_from_the_cabbage_record_only_in_the_crop() -> None:
    kale = get_threat_record_for_crop(KALE)
    cabbage = get_threat_record("brassica-oleracea", "Brevicoryne brassicae")

    assert (kale.version, kale.dossier, kale.dossier_version, kale.review_status, kale.eppo_code) == (
        cabbage.version, cabbage.dossier, cabbage.dossier_version, cabbage.review_status, cabbage.eppo_code,
    )
    kale_def, cabbage_def = kale.definition, cabbage.definition
    assert kale_def.crop != cabbage_def.crop
    assert kale_def.model_dump(exclude={"crop"}) == cabbage_def.model_dump(exclude={"crop"})


def test_kale_pair_lookup_and_case_insensitivity() -> None:
    assert get_threat(KALE, "Brevicoryne brassicae").crop == KALE
    assert get_threat_record("BRASSICA-OLERACEA-SABELLICA", "brevicoryne brassicae").eppo_code == "BRVCBR"


def test_crop_only_lookup_is_exact_so_cabbage_never_resolves_to_kale() -> None:
    assert get_threat_record_for_crop("brassica-oleracea").definition.crop == "brassica-oleracea"
    with pytest.raises(FuzzyThreatNotFound):
        get_threat_record_for_crop("brassica-oleracea-italica")


def test_legacy_eppo_code_resolves_to_the_brvcbr_record() -> None:
    """BREBRA was never an EPPO code; a
    lookup by the legacy code resolves the same record as BRVCBR."""
    legacy = get_threat_record_by_eppo(KALE, "BREBRA")
    current = get_threat_record_by_eppo(KALE, "BRVCBR")

    assert legacy.eppo_code == "BRVCBR"
    assert legacy == current


def test_no_data_file_carries_the_legacy_code_outside_its_correction_note() -> None:
    for path in sorted(loader._DATA_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        notes = raw.get("_authoring_notes", [])
        remaining_notes = [n for n in notes if "corrected from BREBRA" not in n]
        text_without_correction_notes = json.dumps({**raw, "_authoring_notes": remaining_notes})
        assert "BREBRA" not in text_without_correction_notes, path


def test_cucumber_records_are_pair_resolvable_and_myzupe_is_the_only_default() -> None:
    myzupe = get_threat_record_by_eppo("cucumis-sativus", "MYZUPE")
    aphigo = get_threat_record_by_eppo("cucumis-sativus", "APHIGO")
    assert myzupe.default_for_crop is True
    assert aphigo.default_for_crop is False
    assert myzupe.definition.model_dump(exclude={"scientific_name", "common_name"}) == aphigo.definition.model_dump(exclude={"scientific_name", "common_name"})
    assert get_threat_record_for_crop("cucumis-sativus").eppo_code == "MYZUPE"


def test_crop_only_lookup_refuses_an_ambiguous_catalogue_by_crop_name(tmp_path, monkeypatch) -> None:
    entry = {
        "scientific_name": "One", "common_name": "One", "eppo_code": "ONE", "version": "1", "dossier": "d", "dossier_version": "1", "review_status": "test",
        "threat_definition": {"scientific_name": "One", "common_name": "One", "threat_type": "insect", "crop": "ambiguous-crop", "bio_params": {}, "fuzzy_rules": [], "special_rules": {}},
    }
    for name in ("one.json", "two.json"):
        (tmp_path / name).write_text(json.dumps(entry), encoding="utf-8")
    monkeypatch.setattr(loader, "_DATA_DIR", tmp_path)
    loader._records.clear()
    with pytest.raises(FuzzyThreatNotFound, match="ambiguous-crop.*ambiguous"):
        loader.get_threat_record_for_crop("ambiguous-crop")
