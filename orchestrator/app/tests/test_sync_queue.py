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
A power-cycle leaves the queue intact and pending. Since sync_queue.py holds no in-process state
(module docstring), "power-cycle" is simulated the same way
test_state_machine.py's TestRestartMidWorkflowResumes simulates a process
restart for the SQLite-backed state machine: never reuse a Python object
across the "before" and "after" halves of a test, only the on-disk
directory path -- proving the durable state really lives on disk, not in
any cached handle. `enqueue` now references the
sync copy (`pdf_path`/`content_hash`) plus the local report's own path and
hash (`local_pdf_path`/`local_content_hash`), and refuses (writing no
manifest) a sync copy over `SYNC_COPY_MAX_BYTES` -- these cells write real
files under `tmp_path` so `enqueue`'s own `os.path.getsize` check has
something real to measure, mirroring how a real sync copy is always an
on-disk PDF by the time it reaches the queue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sync_queue


def _write_pdf(path, size_bytes: int = 32) -> str:
    """A real file at `path`, `size_bytes` long -- `enqueue` measures the
    sync copy's actual size on disk, so every cell below needs one."""
    path.write_bytes(b"%PDF-1.4 " + b"\0" * max(size_bytes - 9, 0))
    return str(path)


class TestEnqueueAndReadBack:
    def test_enqueue_creates_a_pending_manifest(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf", size_bytes=64)

        entry = sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "sync-hash-1", local_pdf, "local-hash-1")

        assert entry.state == sync_queue.PENDING
        assert entry.session_id == "sess-1"
        assert entry.content_hash == "sync-hash-1"
        assert entry.pdf_path == sync_pdf
        assert entry.local_pdf_path == local_pdf
        assert entry.local_content_hash == "local-hash-1"

        manifest_path = tmp_path / "sess-1.manifest.json"
        assert manifest_path.is_file()
        on_disk = json.loads(manifest_path.read_text())
        assert on_disk["state"] == "pending"
        assert on_disk["cloud_report_id"] is None
        assert on_disk["pdf_path"] == sync_pdf
        assert on_disk["content_hash"] == "sync-hash-1"
        assert on_disk["local_pdf_path"] == local_pdf
        assert on_disk["local_content_hash"] == "local-hash-1"

    def test_enqueue_is_idempotent_and_does_not_reset_state(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_synced(tmp_path, "sess-1", "cloud-uuid-1")

        entry = sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "cloud-uuid-1"

    def test_read_entry_missing_raises(self, tmp_path):
        with pytest.raises(sync_queue.QueueEntryNotFound):
            sync_queue.read_entry(tmp_path, "no-such-session")


class TestEnqueueRefusesOversizedSyncCopy:
    def test_a_file_over_the_bound_is_refused_and_writes_no_manifest(self, tmp_path):
        oversized = _write_pdf(tmp_path / "sess-big.sync.pdf", size_bytes=2_500_000)
        local_pdf = _write_pdf(tmp_path / "sess-big.pdf")

        with pytest.raises(sync_queue.SyncQueueRefused) as excinfo:
            sync_queue.enqueue(tmp_path, "sess-big", oversized, "hash-big", local_pdf, "local-hash-big")

        message = str(excinfo.value)
        assert "2500000" in message
        assert str(sync_queue.SYNC_COPY_MAX_BYTES) in message
        assert excinfo.value.size == 2_500_000

        assert not (tmp_path / "sess-big.manifest.json").exists()

    def test_a_file_at_or_under_the_bound_is_accepted(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-ok.sync.pdf", size_bytes=sync_queue.SYNC_COPY_MAX_BYTES)
        local_pdf = _write_pdf(tmp_path / "sess-ok.pdf")

        entry = sync_queue.enqueue(tmp_path, "sess-ok", sync_pdf, "hash-ok", local_pdf, "local-hash-ok")
        assert entry.state == sync_queue.PENDING


class TestLegacyFourArgumentEnqueue:
    """`local_pdf_path`/`local_hash` are optional, so a four-argument call (a fake PDF with no
    local counterpart) keeps working; the service's own call site always passes all six."""

    def test_four_argument_call_writes_a_manifest_with_null_local_fields(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-legacy.sync.pdf")

        entry = sync_queue.enqueue(tmp_path, "sess-legacy", sync_pdf, "sync-hash-legacy")

        assert entry.pdf_path == sync_pdf
        assert entry.content_hash == "sync-hash-legacy"
        assert entry.local_pdf_path is None
        assert entry.local_content_hash is None

        on_disk = json.loads((tmp_path / "sess-legacy.manifest.json").read_text())
        assert on_disk["pdf_path"] == sync_pdf
        assert on_disk["content_hash"] == "sync-hash-legacy"
        assert on_disk["local_pdf_path"] is None
        assert on_disk["local_content_hash"] is None

    def test_six_argument_call_writes_both_the_sync_and_local_pairs(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-full.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-full.pdf", size_bytes=64)

        entry = sync_queue.enqueue(tmp_path, "sess-full", sync_pdf, "sync-hash-full", local_pdf, "local-hash-full")

        assert entry.pdf_path == sync_pdf
        assert entry.content_hash == "sync-hash-full"
        assert entry.local_pdf_path == local_pdf
        assert entry.local_content_hash == "local-hash-full"

    def test_size_guard_applies_to_the_four_argument_form_too(self, tmp_path):
        oversized = _write_pdf(tmp_path / "sess-legacy-big.sync.pdf", size_bytes=2_500_000)

        with pytest.raises(sync_queue.SyncQueueRefused) as excinfo:
            sync_queue.enqueue(tmp_path, "sess-legacy-big", oversized, "hash-big")

        assert str(sync_queue.SYNC_COPY_MAX_BYTES) in str(excinfo.value)
        assert not (tmp_path / "sess-legacy-big.manifest.json").exists()


class TestPairPath:
    """`pair_path`
    round-trips through the manifest and reads as None on a manifest written
    before the field existed."""

    def test_pair_path_round_trips_through_the_manifest(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-pair.sync.pdf")
        pair = str(tmp_path / "sess-pair.sync-pair.json")

        entry = sync_queue.enqueue(tmp_path, "sess-pair", sync_pdf, "h", pair_path=pair)

        assert entry.pair_path == pair
        assert json.loads((tmp_path / "sess-pair.manifest.json").read_text())["pair_path"] == pair
        assert sync_queue.read_entry(tmp_path, "sess-pair").pair_path == pair

    def test_omitted_pair_path_is_null(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-nopair.sync.pdf")

        entry = sync_queue.enqueue(tmp_path, "sess-nopair", sync_pdf, "h")

        assert entry.pair_path is None

    def test_legacy_manifest_without_the_field_reads_as_none(self, tmp_path):
        manifest = {
            "session_id": "sess-old",
            "content_hash": "h",
            "pdf_path": "/x.pdf",
            "state": "pending",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        (tmp_path / "sess-old.manifest.json").write_text(json.dumps(manifest))

        assert sync_queue.read_entry(tmp_path, "sess-old").pair_path is None

    def test_state_transitions_keep_the_pair_path(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-keep.sync.pdf")
        sync_queue.enqueue(tmp_path, "sess-keep", sync_pdf, "h", pair_path="/p.json")

        sync_queue.mark_uploading(tmp_path, "sess-keep")

        assert sync_queue.read_entry(tmp_path, "sess-keep").pair_path == "/p.json"


class TestReassessFields:
    """The re-assessment
    fields round-trip through the manifest, read with defaults on a legacy
    one, and `list_open_reassessments` names the synced entries still open."""

    def _synced(self, tmp_path, session_id: str):
        sync_queue.enqueue(tmp_path, session_id, _write_pdf(tmp_path / f"{session_id}.sync.pdf"), f"h-{session_id}")
        sync_queue.mark_synced(tmp_path, session_id, f"edge-{session_id}")

    def test_a_new_entry_has_the_defaults(self, tmp_path):
        entry = sync_queue.enqueue(tmp_path, "s", _write_pdf(tmp_path / "s.sync.pdf"), "h")

        assert (entry.reassess_state, entry.reassess_attempts) == (None, 0)
        assert (entry.reassess_reason, entry.cloud_reassessed_report_id) == (None, None)

    def test_legacy_manifest_reads_with_the_defaults(self, tmp_path):
        manifest = {
            "session_id": "sess-old",
            "content_hash": "h",
            "pdf_path": "/x.pdf",
            "state": "synced",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        (tmp_path / "sess-old.manifest.json").write_text(json.dumps(manifest))

        entry = sync_queue.read_entry(tmp_path, "sess-old")

        assert (entry.reassess_state, entry.reassess_attempts) == (None, 0)
        assert (entry.reassess_reason, entry.cloud_reassessed_report_id) == (None, None)

    def test_mark_reassess_round_trips_and_counts_attempts(self, tmp_path):
        self._synced(tmp_path, "s")

        sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_PENDING, reason="unreachable", attempt=True)
        entry = sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_PENDING, reason="unreachable", attempt=True)
        assert (entry.reassess_state, entry.reassess_attempts, entry.reassess_reason) == ("pending", 2, "unreachable")

        on_disk = sync_queue.read_entry(tmp_path, "s")
        assert (on_disk.reassess_state, on_disk.reassess_attempts) == ("pending", 2)

    def test_the_report_id_is_kept_and_progress_resets_the_attempts(self, tmp_path):
        self._synced(tmp_path, "s")
        sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_PENDING, reason="x", attempt=True)

        submitted = sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_SUBMITTED, report_id="cloud-1")
        assert (submitted.cloud_reassessed_report_id, submitted.reassess_attempts) == ("cloud-1", 0)

        pending_pair = sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_SUBMITTED, reason="not embedded")
        assert pending_pair.cloud_reassessed_report_id == "cloud-1"

    def test_mark_reassess_leaves_the_upload_fields_alone(self, tmp_path):
        self._synced(tmp_path, "s")

        entry = sync_queue.mark_reassess(tmp_path, "s", sync_queue.REASSESS_FAILED, reason="refused")

        assert (entry.state, entry.cloud_report_id) == (sync_queue.SYNCED, "edge-s")

    def test_list_open_reassessments(self, tmp_path):
        for name in ("legacy", "pending", "submitted", "done", "failed"):
            self._synced(tmp_path, name)
        sync_queue.mark_reassess(tmp_path, "pending", sync_queue.REASSESS_PENDING, attempt=True)
        sync_queue.mark_reassess(tmp_path, "submitted", sync_queue.REASSESS_SUBMITTED, report_id="c")
        sync_queue.mark_reassess(tmp_path, "done", sync_queue.REASSESS_DONE)
        sync_queue.mark_reassess(tmp_path, "failed", sync_queue.REASSESS_FAILED, reason="r")
        sync_queue.enqueue(tmp_path, "unsynced", _write_pdf(tmp_path / "unsynced.sync.pdf"), "h")

        open_ids = {entry.session_id for entry in sync_queue.list_open_reassessments(tmp_path)}

        assert open_ids == {"legacy", "pending", "submitted"}


class TestPowerCycleSurvival:
    def test_queue_survives_simulated_power_cycle_still_pending(self, tmp_path):
        queue_dir = tmp_path / "queue"
        sync_pdf = _write_pdf(tmp_path / "report.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "report.pdf")
        sync_queue.enqueue(queue_dir, "sess-power-cycle", sync_pdf, "hash-abc", local_pdf, "local-hash-abc")

        entry = sync_queue.read_entry(queue_dir, "sess-power-cycle")
        assert entry.state == sync_queue.PENDING
        assert entry.content_hash == "hash-abc"

        pending = sync_queue.list_syncable(queue_dir)
        assert [e.session_id for e in pending] == ["sess-power-cycle"]

    def test_power_cycle_mid_upload_leaves_uploading_state_intact_and_syncable(self, tmp_path):
        queue_dir = tmp_path / "queue"
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(queue_dir, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_uploading(queue_dir, "sess-1")

        entry = sync_queue.read_entry(queue_dir, "sess-1")
        assert entry.state == sync_queue.UPLOADING

        syncable = sync_queue.list_syncable(queue_dir)
        assert [e.session_id for e in syncable] == ["sess-1"]


class TestStateTransitionsAreAtomic:
    def test_mark_synced_no_longer_syncable(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_synced(tmp_path, "sess-1", "cloud-uuid")

        assert sync_queue.list_syncable(tmp_path) == []
        entry = sync_queue.read_entry(tmp_path, "sess-1")
        assert entry.state == sync_queue.SYNCED
        assert entry.cloud_report_id == "cloud-uuid"

    def test_mark_failed_records_reason_and_stays_syncable(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_failed(tmp_path, "sess-1", "connection refused")

        entry = sync_queue.read_entry(tmp_path, "sess-1")
        assert entry.state == sync_queue.FAILED
        assert entry.failure_reason == "connection refused"
        assert [e.session_id for e in sync_queue.list_syncable(tmp_path)] == ["sess-1"]

    def test_mark_failed_records_the_failed_hash(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_failed(tmp_path, "sess-1", "refused by the Gatekeeper's body limit (413)", failed_hash="hash-1")

        entry = sync_queue.read_entry(tmp_path, "sess-1")
        assert entry.failed_hash == "hash-1"

    def test_no_leftover_temp_files_after_transitions(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_uploading(tmp_path, "sess-1")
        sync_queue.mark_synced(tmp_path, "sess-1", "cloud-uuid")

        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["sess-1.manifest.json", "sess-1.pdf", "sess-1.sync.pdf"]


class TestSyncConfirmedMarker:
    """The durable
    marker `mark_sync_confirmed` writes must survive deletion of the
    transient manifest -- it is a SEPARATE file specifically so that
    housekeeping.py purging `<session_id>.manifest.json` can never take
    the inference service's own purge-eligibility signal down with it."""

    def test_marker_written_at_a_distinct_path_from_the_manifest(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_sync_confirmed(tmp_path, "sess-1")

        marker_path = tmp_path / ".synced" / "sess-1.json"
        assert marker_path.is_file()
        on_disk = json.loads(marker_path.read_text())
        assert on_disk["session_id"] == "sess-1"

    def test_marker_survives_manifest_deletion(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_sync_confirmed(tmp_path, "sess-1")
        sync_queue.mark_synced(tmp_path, "sess-1", "cloud-uuid")

        (tmp_path / "sess-1.manifest.json").unlink()

        assert (tmp_path / ".synced" / "sess-1.json").is_file()

    def test_marker_write_is_idempotent(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "sess-1.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1", local_pdf, "local-hash-1")
        sync_queue.mark_sync_confirmed(tmp_path, "sess-1")
        sync_queue.mark_sync_confirmed(tmp_path, "sess-1")

        assert (tmp_path / ".synced" / "sess-1.json").is_file()


class TestListSyncableOrdering:
    def test_oldest_queued_first(self, tmp_path, monkeypatch):
        import datetime

        times = iter(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:05:00+00:00",
                "2026-01-01T00:10:00+00:00",
            ]
        )

        def _fake_now_iso():
            return next(times)

        monkeypatch.setattr(sync_queue, "_utcnow_iso", _fake_now_iso)

        sync_c = _write_pdf(tmp_path / "sess-c.sync.pdf")
        sync_a = _write_pdf(tmp_path / "sess-a.sync.pdf")
        sync_b = _write_pdf(tmp_path / "sess-b.sync.pdf")
        local_pdf = _write_pdf(tmp_path / "local.pdf")

        sync_queue.enqueue(tmp_path, "sess-c", sync_c, "hash-c", local_pdf, "local-hash-c")
        sync_queue.enqueue(tmp_path, "sess-a", sync_a, "hash-a", local_pdf, "local-hash-a")
        sync_queue.enqueue(tmp_path, "sess-b", sync_b, "hash-b", local_pdf, "local-hash-b")

        ordered = [e.session_id for e in sync_queue.list_syncable(tmp_path)]
        assert ordered == ["sess-c", "sess-a", "sess-b"]


class TestFailureCountAndKind:

    def _queued(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1")
        return sync_pdf

    def test_a_manifest_from_before_the_fields_loads_with_defaults(self, tmp_path):
        self._queued(tmp_path)
        manifest = tmp_path / "sess-1.manifest.json"
        data = json.loads(manifest.read_text())
        del data["failure_count"], data["failure_kind"]
        manifest.write_text(json.dumps(data))

        entry = sync_queue.read_entry(tmp_path, "sess-1")

        assert entry.failure_count == 0
        assert entry.failure_kind is None
        assert [e.session_id for e in sync_queue.list_entries(tmp_path)] == ["sess-1"]

    def test_mark_failed_counts_each_failure_and_records_the_kind(self, tmp_path):
        self._queued(tmp_path)

        first = sync_queue.mark_failed(tmp_path, "sess-1", "500", kind=sync_queue.FAILURE_TRANSIENT)
        second = sync_queue.mark_failed(tmp_path, "sess-1", "500 again", kind=sync_queue.FAILURE_TRANSIENT)

        assert (first.failure_count, second.failure_count) == (1, 2)
        entry = sync_queue.read_entry(tmp_path, "sess-1")
        assert entry.failure_count == 2
        assert entry.failure_kind == "transient"

    def test_mark_uploading_keeps_the_count(self, tmp_path):
        self._queued(tmp_path)
        sync_queue.mark_failed(tmp_path, "sess-1", "500", kind=sync_queue.FAILURE_TRANSIENT)
        sync_queue.mark_uploading(tmp_path, "sess-1")

        assert sync_queue.read_entry(tmp_path, "sess-1").failure_count == 1

    def test_mark_synced_resets_the_count_and_the_kind(self, tmp_path):
        self._queued(tmp_path)
        sync_queue.mark_failed(tmp_path, "sess-1", "500", kind=sync_queue.FAILURE_TRANSIENT)

        sync_queue.mark_synced(tmp_path, "sess-1", "cloud-uuid")

        entry = sync_queue.read_entry(tmp_path, "sess-1")
        assert entry.failure_count == 0
        assert entry.failure_kind is None


class TestRetire:
    def test_retire_moves_the_manifest_with_state_and_reason(self, tmp_path):
        sync_pdf = _write_pdf(tmp_path / "sess-1.sync.pdf")
        sync_queue.enqueue(tmp_path, "sess-1", sync_pdf, "hash-1")
        sync_queue.enqueue(tmp_path, "sess-2", _write_pdf(tmp_path / "sess-2.sync.pdf"), "hash-2")

        sync_queue.retire(tmp_path, "sess-1", "the queued file is too large")

        assert not (tmp_path / "sess-1.manifest.json").exists()
        moved = json.loads((tmp_path / ".retired" / "sess-1.manifest.json").read_text())
        assert moved["state"] == sync_queue.RETIRED
        assert moved["failure_reason"] == "the queued file is too large"
        assert [e.session_id for e in sync_queue.list_entries(tmp_path)] == ["sess-2"]
        assert [e.session_id for e in sync_queue.list_syncable(tmp_path)] == ["sess-2"]
        retired = sync_queue.list_retired(tmp_path)
        assert [e.session_id for e in retired] == ["sess-1"]
        assert retired[0].state == sync_queue.RETIRED
        assert Path(sync_pdf).exists()

    def test_list_retired_is_empty_without_the_directory(self, tmp_path):
        assert sync_queue.list_retired(tmp_path) == []

    def test_retire_of_an_unknown_session_raises(self, tmp_path):
        with pytest.raises(sync_queue.QueueEntryNotFound):
            sync_queue.retire(tmp_path, "nope", "reason")
