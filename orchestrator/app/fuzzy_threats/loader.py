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

"""Orchestrator-owned ThreatDefinition catalog.

The pinned `agstack-pnd` bundled catalog (8 threats / 5 crops) has no Brassica
or cucumber aphid entries. This module loads the orchestrator's own
literature-sourced definitions, templated off the upstream insect entries -- the same way
`agstack_pnd.catalog.loader._load_catalog` builds `ThreatDefinition` objects
from JSON, but keeps it in a JSON file this repo owns rather than the pinned
dependency's bundled data.

`ThreatDefinition` itself carries no version/citation fields, so each JSON
file wraps the engine-shaped `threat_definition` object with sibling
provenance metadata (`version`, `dossier`, `dossier_version`,
`review_status`) recovered via `get_threat_record`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agstack_pnd.foundation.types import (
    BioParams,
    FuzzyRule,
    RiskLevel,
    ThreatDefinition,
    ThreatType,
)

_DATA_DIR = Path(__file__).parent / "data"

LEGACY_EPPO_CODES = {"BREBRA": "BRVCBR"}


class FuzzyThreatNotFound(LookupError):
    def __init__(self, crop: str, scientific_name: str, eppo_code: str | None = None) -> None:
        pest = f"eppo_code={eppo_code!r}" if eppo_code is not None else f"scientific_name={scientific_name!r}"
        super().__init__(f"No orchestrator-owned ThreatDefinition for crop={crop!r}, {pest}")


@dataclass(frozen=True)
class ThreatDefinitionRecord:
    """A `ThreatDefinition` plus the provenance the pydantic model can't carry."""

    definition: ThreatDefinition
    version: str
    dossier: str
    dossier_version: str
    review_status: str
    eppo_code: str
    default_for_crop: bool = False


def _threat_definition_from_entry(entry: dict[str, object]) -> ThreatDefinition:
    bio = entry.get("bio_params", {})
    rules_raw = entry.get("fuzzy_rules", [])
    fuzzy_rules = [
        FuzzyRule(
            hum_lo=r["hum_lo"],
            hum_hi=r["hum_hi"],
            temp_lo=r["temp_lo"],
            temp_hi=r["temp_hi"],
            rain_min=r.get("rain_min", 0.0),
            risk_level=RiskLevel(r["risk_level"]),
        )
        for r in rules_raw
    ]
    return ThreatDefinition(
        scientific_name=entry["scientific_name"],
        common_name=entry["common_name"],
        threat_type=ThreatType(entry.get("threat_type", "insect")),
        crop=entry["crop"],
        bio_params=BioParams(**bio),
        fuzzy_rules=fuzzy_rules,
        special_rules=entry.get("special_rules", {}),
    )


def _load_record(path: Path) -> ThreatDefinitionRecord:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return ThreatDefinitionRecord(
        definition=_threat_definition_from_entry(raw["threat_definition"]),
        version=raw["version"],
        dossier=raw["dossier"],
        dossier_version=raw["dossier_version"],
        review_status=raw["review_status"],
        eppo_code=raw["eppo_code"],
        default_for_crop=bool(raw.get("default_for_crop", False)),
    )


_records: dict[Path, ThreatDefinitionRecord] = {}


def _load(path: Path) -> ThreatDefinitionRecord:
    if path not in _records:
        _records[path] = _load_record(path)
    return _records[path]


def get_threat_record(crop: str, scientific_name: str) -> ThreatDefinitionRecord:
    """Look up a specific crop-threat pairing with its version and source metadata."""
    for path in sorted(_DATA_DIR.glob("*.json")):
        record = _load(path)
        d = record.definition
        if d.crop.lower() == crop.lower() and d.scientific_name.lower() == scientific_name.lower():
            return record
    raise FuzzyThreatNotFound(crop, scientific_name)


def get_threat(crop: str, scientific_name: str) -> ThreatDefinition:
    """`agstack_pnd.catalog.get_threat`-equivalent: the bare `ThreatDefinition`."""
    return get_threat_record(crop, scientific_name).definition


def get_threat_record_by_eppo(crop: str, eppo_code: str) -> ThreatDefinitionRecord:
    """Look up a crop-threat pairing by the threat's EPPO code. Raises
    `FuzzyThreatNotFound` when the catalog has no record for that exact pair
    never another pest's record for the crop."""
    resolved_eppo_code = LEGACY_EPPO_CODES.get(eppo_code.upper(), eppo_code)
    for path in sorted(_DATA_DIR.glob("*.json")):
        record = _load(path)
        if record.definition.crop.lower() == crop.lower() and record.eppo_code.lower() == resolved_eppo_code.lower():
            return record
    raise FuzzyThreatNotFound(crop, "", eppo_code=eppo_code)


def get_threat_record_for_crop(crop: str) -> ThreatDefinitionRecord:
    """Look up the catalog entry by crop alone. Raises `FuzzyThreatNotFound` (scientific_name left blank in the
    message -- crop-only lookup) when no entry matches, exactly like
    `get_threat_record`; callers must treat that as "no catalog identity
    available for this session", never fabricate one."""
    records = [
        _load(path)
        for path in sorted(_DATA_DIR.glob("*.json"))
        if _load(path).definition.crop.lower() == crop.lower()
    ]
    if len(records) == 1:
        return records[0]
    if len(records) > 1:
        defaults = [record for record in records if record.default_for_crop]
        if len(defaults) == 1:
            return defaults[0]
        raise FuzzyThreatNotFound(crop, "ambiguous")
    raise FuzzyThreatNotFound(crop, "")
