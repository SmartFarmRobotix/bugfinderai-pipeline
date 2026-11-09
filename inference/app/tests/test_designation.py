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
Unit tests for per-execution-target designation-status resolution, keyed
by verified sha256.
"""
from __future__ import annotations

import pytest

from app.detectors.designation import (
    DesignationManifestError,
    assert_per_target_designation_uniqueness,
    resolve_designation_status,
)

_INTEGRATION_SHA = "a" * 64
_DESIGNATED_SHA = "b" * 64
_ABSENT_SHA = "c" * 64


def _manifest() -> dict:
    return {
        "schema_version": "1.0.0",
        "targets": {
            "cloud": {
                "artifacts": {
                    _INTEGRATION_SHA: {"designation_status": "integration-only"},
                }
            },
            "jetson": {
                "artifacts": {
                    _DESIGNATED_SHA: {"designation_status": "designated"},
                }
            },
        },
    }


class TestResolveDesignationStatus:
    def test_case_a_manifest_lists_artifact_as_integration_only(self):
        status = resolve_designation_status(_INTEGRATION_SHA, "cloud", _manifest())
        assert status == "integration-only"

    def test_case_b_manifest_lists_artifact_as_designated(self):
        status = resolve_designation_status(_DESIGNATED_SHA, "jetson", _manifest())
        assert status == "designated"

    def test_case_c_artifact_absent_from_manifest_fails_loudly(self):
        """An artifact not in the manifest is never silently integration-only."""
        with pytest.raises(DesignationManifestError, match=_ABSENT_SHA):
            resolve_designation_status(_ABSENT_SHA, "cloud", _manifest())

    def test_unknown_execution_target_fails_loudly(self):
        with pytest.raises(DesignationManifestError, match="mixed"):
            resolve_designation_status(_INTEGRATION_SHA, "mixed", _manifest())

    def test_hash_present_under_wrong_target_is_treated_as_absent(self):
        """The designated hash lives under 'jetson' — looking it up under
        'cloud' must fail loudly, not silently cross targets."""
        with pytest.raises(DesignationManifestError):
            resolve_designation_status(_DESIGNATED_SHA, "cloud", _manifest())

    def test_invalid_status_value_rejected(self):
        manifest = {
            "targets": {
                "cloud": {"artifacts": {_INTEGRATION_SHA: {"designation_status": "field-eligible"}}}
            }
        }
        with pytest.raises(DesignationManifestError):
            resolve_designation_status(_INTEGRATION_SHA, "cloud", manifest)

    def test_resolution_enforces_per_target_uniqueness_of_whole_manifest(self):
        """A manifest designating TWO artifacts for one target is rejected
        at resolution time — even when the artifact being resolved has a
        perfectly valid entry of its own under a DIFFERENT target. The
        uniqueness invariant is enforced on every executor load, not only
        when the standalone helper happens to be called."""
        manifest = _manifest()
        manifest["targets"]["cloud"]["artifacts"][_ABSENT_SHA] = {
            "designation_status": "designated"
        }
        manifest["targets"]["cloud"]["artifacts"][_INTEGRATION_SHA] = {
            "designation_status": "designated"
        }
        with pytest.raises(DesignationManifestError, match="cloud"):
            resolve_designation_status(_DESIGNATED_SHA, "jetson", manifest)


class TestPerTargetDesignationUniqueness:
    def test_single_designated_per_target_passes(self):
        assert_per_target_designation_uniqueness(_manifest())

    def test_two_designated_entries_for_same_target_rejected(self):
        manifest = {
            "targets": {
                "cloud": {
                    "artifacts": {
                        _INTEGRATION_SHA: {"designation_status": "designated"},
                        _DESIGNATED_SHA: {"designation_status": "designated"},
                    }
                }
            }
        }
        with pytest.raises(DesignationManifestError, match="cloud"):
            assert_per_target_designation_uniqueness(manifest)

    def test_no_designated_entries_is_allowed(self):
        """Not yet having designated a fine-tune for a target is legitimate — uniqueness
        only forbids MORE than one, not zero."""
        manifest = {"targets": {"cloud": {"artifacts": {_INTEGRATION_SHA: {"designation_status": "integration-only"}}}}}
        assert_per_target_designation_uniqueness(manifest)
