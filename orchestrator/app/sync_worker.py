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
Store-and-forward sync upload worker.
Manual-trigger baseline -- run this module whenever the edge
node has been physically brought to connectivity:

    python sync_worker.py

(or `docker exec <orchestrator-container> python sync_worker.py`, run from /code/app; the sync service of deploy/edge/compose.yml sets that working directory). Reads the filesystem queue (sync_queue.py),
refuses to run on a placeholder/unset sync credential, and
uploads each pending/uploading/failed entry to the cloud Reporting
Service's additive edge-ingest endpoint via Gatekeeper,
idempotently -- a report already accepted by the Reporting service (same session_id +
content_hash) is marked synced locally without creating a duplicate.

After an entry's upload the worker
also submits the session to the cloud orchestrator's intake (marked as an
edge re-assessment) and then pushes the capped image pair to the Reporting
service (`_reassess`). The outcome is recorded on the manifest; it never
changes the entry's synced state, the sync marker or the uploaded copy, and
an open re-assessment is retried at the start of every later run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

import sync_queue
from config import PLACEHOLDER_SECRET_VALUES, Settings, get_settings
from db.session import SessionLocal
from cloud_intake_client import (
    PairPushError,
    PairPushHTTPError,
    ReassessCallError,
    ReassessHTTPError,
    push_image_pair,
    submit_for_reassessment,
)
from models import ReportIndexRecord, SessionRecord
from reporting_sync_client import ReportSyncCallError, ReportSyncHTTPError, push_synced_report

_BODY_LIMIT_FAILURE_REASON = "refused by the Gatekeeper's body limit (413)"

logger = logging.getLogger("orchestrator.sync_worker")

_REJECTION_KINDS = {sync_queue.FAILURE_BODY_LIMIT, sync_queue.FAILURE_REJECTED}


def _failure_kind(exc: ReportSyncCallError) -> str:
    """413 is the Gatekeeper's body limit; any other 4xx is a rejection of
    the request; everything else -- 5xx, an unreachable service, and the
    client's own malformed-answer errors (a non-JSON or uuid-less reply
    carries the 2xx status it arrived with) -- may pass on a later run."""
    if isinstance(exc, ReportSyncHTTPError):
        if exc.status_code == 413:
            return sync_queue.FAILURE_BODY_LIMIT
        if 400 <= exc.status_code < 500:
            return sync_queue.FAILURE_REJECTED
    return sync_queue.FAILURE_TRANSIENT


def _file_sha256(path: str) -> str:
    """The skip-a-repeat-failure check
    hashes the queued FILE's current bytes, not the manifest's own
    `content_hash` field -- the manifest can go stale (e.g. a legacy
    four-argument `enqueue` call, or a manifest field edited independently
    of the file it points at), and a comparison against stale metadata
    could skip a file that has actually changed since the failure it
    matches."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SyncCredentialError(RuntimeError):
    """The sync-scoped Gatekeeper credential is unset or a known placeholder value."""


@dataclass
class SyncRunResult:
    uploaded: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    retired: List[Tuple[str, str]] = field(default_factory=list)
    reassessed: List[str] = field(default_factory=list)
    reassess_failed: List[Tuple[str, str]] = field(default_factory=list)


def _http_error_detail(body: str) -> str:
    """The `detail` of a JSON error body, else the body as it came."""
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return body
    if isinstance(parsed, dict) and parsed.get("detail") is not None:
        return str(parsed["detail"])
    return body


def _is_final_refusal(status: int) -> bool:
    """A client-side refusal the same request cannot outgrow: a 4xx other
    than the auth pair. Everything else (401/403, 5xx, a malformed 2xx) may
    pass on a later run."""
    return 400 <= status < 500 and status not in (401, 403)


def _reassess(
    db: Session,
    settings: Settings,
    entry: sync_queue.QueueEntry,
    edge_report_id: str,
    result: "SyncRunResult",
) -> None:
    """One pass of the cloud re-assessment of an entry: the intake first
    (unless a report id is already recorded), then the image pair, in that
    order -- the intake creates the Reporting index row the pair route
    re-renders against. Records its outcome on the manifest and in `result`;
    raises nothing the caller must handle (the PDF sync is never undone by
    this step)."""
    if entry.reassess_state in (sync_queue.REASSESS_DONE, sync_queue.REASSESS_FAILED):
        return
    queue_dir = settings.sync_queue_dir
    session_id = entry.session_id

    def fail(reason: str) -> None:
        sync_queue.mark_reassess(queue_dir, session_id, sync_queue.REASSESS_FAILED, reason=reason)
        logger.warning("sync: session_id=%s re-assessment failed: %s", session_id, reason)
        result.reassess_failed.append((session_id, reason))

    def attempt_failed(reason: str, state: str) -> None:
        """A failure that may pass on a later run: one attempt, `failed`
        once the transient budget is spent."""
        updated = sync_queue.mark_reassess(queue_dir, session_id, state, reason=reason, attempt=True)
        logger.warning(
            "sync: session_id=%s re-assessment attempt %d failed: %s", session_id, updated.reassess_attempts, reason
        )
        if updated.reassess_attempts >= settings.sync_max_transient_failures:
            fail(reason)

    report_id = entry.cloud_reassessed_report_id
    if report_id is None:
        if not edge_report_id:
            fail("no synced report id is recorded for this entry")
            return
        record = db.scalar(select(SessionRecord).where(SessionRecord.session_id == session_id))
        if record is None:
            fail("no session record on the device to submit")
            return
        submission = {
            "contract_version": record.contract_version,
            "risk_input": json.loads(record.risk_input_json),
            "reporting_envelope": json.loads(record.reporting_envelope_json),
        }
        try:
            answer = submit_for_reassessment(settings, session_id, submission, edge_report_id)
        except ReassessHTTPError as exc:
            if _is_final_refusal(exc.status):
                fail(f"the cloud intake refused the session ({exc.status}): {_http_error_detail(exc.body)}")
            else:
                attempt_failed(str(exc), sync_queue.REASSESS_PENDING)
            return
        except ReassessCallError as exc:
            attempt_failed(str(exc), sync_queue.REASSESS_PENDING)
            return

        if answer.status not in ("accepted", "duplicate") or answer.state != "reported" or not answer.report_id:
            attempt_failed(
                f"the cloud intake answered {answer.status} with state {answer.state!r} and "
                f"report id {answer.report_id!r}: no cloud report yet",
                sync_queue.REASSESS_PENDING,
            )
            return
        report_id = answer.report_id
        sync_queue.mark_reassess(
            queue_dir, session_id, sync_queue.REASSESS_SUBMITTED, reason=None, report_id=report_id
        )
        logger.info("sync: session_id=%s submitted for cloud re-assessment, report_id=%s", session_id, report_id)

    if not entry.pair_path:
        reason = "no image pair was saved with this report (queued without one), so none was pushed"
        sync_queue.mark_reassess(queue_dir, session_id, sync_queue.REASSESS_DONE, reason=reason)
        logger.info("sync: session_id=%s re-assessed without an image pair: %s", session_id, reason)
        result.reassessed.append(session_id)
        return
    try:
        pair = json.loads(Path(entry.pair_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"the image pair file {entry.pair_path} cannot be read: {exc}")
        return

    try:
        pushed = push_image_pair(settings, session_id, pair)
    except PairPushHTTPError as exc:
        if _is_final_refusal(exc.status):
            fail(f"the Reporting service refused the image pair ({exc.status}): {_http_error_detail(exc.body)}")
        else:
            attempt_failed(str(exc), sync_queue.REASSESS_SUBMITTED)
        return
    except PairPushError as exc:
        attempt_failed(str(exc), sync_queue.REASSESS_SUBMITTED)
        return

    if pushed.images_embedded is True:
        sync_queue.mark_reassess(queue_dir, session_id, sync_queue.REASSESS_DONE, reason=None)
        logger.info("sync: session_id=%s re-assessed, image pair embedded", session_id)
        result.reassessed.append(session_id)
        return
    reason = "the Reporting service has not confirmed the image pair as embedded in its report"
    sync_queue.mark_reassess(queue_dir, session_id, sync_queue.REASSESS_SUBMITTED, reason=reason)
    logger.info("sync: session_id=%s image pair pending: %s", session_id, reason)


def _reassess_isolated(
    db: Session,
    settings: Settings,
    entry: sync_queue.QueueEntry,
    edge_report_id: str,
    result: "SyncRunResult",
) -> None:
    """`_reassess` behind a net: whatever it raises, the entry's upload
    stays as it is and the failure counts as one attempt."""
    try:
        _reassess(db, settings, entry, edge_report_id, result)
    except Exception as exc:  # noqa: BLE001 -- the PDF sync is never undone by this step
        logger.exception("sync: session_id=%s re-assessment step raised", entry.session_id)
        try:
            current = sync_queue.read_entry(settings.sync_queue_dir, entry.session_id)
            updated = sync_queue.mark_reassess(
                settings.sync_queue_dir,
                entry.session_id,
                current.reassess_state or sync_queue.REASSESS_PENDING,
                reason=f"unexpected error: {exc}",
                report_id=current.cloud_reassessed_report_id,
                attempt=True,
            )
            if updated.reassess_attempts >= settings.sync_max_transient_failures:
                sync_queue.mark_reassess(
                    settings.sync_queue_dir,
                    entry.session_id,
                    sync_queue.REASSESS_FAILED,
                    reason=f"unexpected error: {exc}",
                )
                result.reassess_failed.append((entry.session_id, f"unexpected error: {exc}"))
        except Exception:  # noqa: BLE001
            logger.exception("sync: session_id=%s re-assessment outcome could not be recorded", entry.session_id)


def validate_sync_credentials(settings: Settings) -> None:
    """Refuse to sync on a missing or
    placeholder/default credential --
    checked here, at sync-invocation time, not folded into config.py's
    startup validator (sync credentials are only needed when this worker runs, later than edge
    operation)."""
    username = settings.sync_gatekeeper_username
    password = settings.sync_gatekeeper_password
    if not username or not password:
        raise SyncCredentialError(
            "ORCH_SYNC_GATEKEEPER_USERNAME/ORCH_SYNC_GATEKEEPER_PASSWORD must both be "
            "set to run sync"
        )
    if username.strip().lower() in PLACEHOLDER_SECRET_VALUES or password.strip().lower() in PLACEHOLDER_SECRET_VALUES:
        raise SyncCredentialError(
            "the sync Gatekeeper credential is set to a known placeholder value; "
            "refusing to sync. Set ORCH_SYNC_GATEKEEPER_USERNAME/"
            "ORCH_SYNC_GATEKEEPER_PASSWORD to real, non-placeholder values."
        )


def run_sync_once(db: Session, settings: Settings) -> SyncRunResult:
    """One pass over the queue: (re)attempt every syncable entry exactly
    once. Safe to call repeatedly -- including immediately after an
    interrupted prior run."""
    validate_sync_credentials(settings)

    result = SyncRunResult()

    def retire(session_id: str, reason: str) -> None:
        """Move the entry out of the queue and mark the session's local
        index row; the report file on the device is never touched."""
        sync_queue.retire(settings.sync_queue_dir, session_id, reason)
        report_index = db.scalar(
            select(ReportIndexRecord).where(
                ReportIndexRecord.session_id == session_id,
                ReportIndexRecord.origin == "edge-local",
            )
        )
        if report_index is not None:
            report_index.state = "retired"
            db.add(report_index)
            db.commit()
        logger.warning("sync: session_id=%s retired: %s", session_id, reason)
        result.retired.append((session_id, reason))

    if settings.sync_reassess_enabled:
        for open_entry in sync_queue.list_open_reassessments(settings.sync_queue_dir):
            _reassess_isolated(db, settings, open_entry, open_entry.cloud_report_id or "", result)

    for entry in sync_queue.list_syncable(settings.sync_queue_dir):
        queued_file = Path(entry.pdf_path)
        if not queued_file.is_file():
            retire(entry.session_id, f"the queued file no longer exists: {entry.pdf_path}")
            continue
        size = queued_file.stat().st_size
        if size > sync_queue.SYNC_COPY_MAX_BYTES:
            retire(
                entry.session_id,
                f"the queued file {entry.pdf_path} is {size} bytes, over the "
                f"{sync_queue.SYNC_COPY_MAX_BYTES}-byte bound: it can never pass the Gatekeeper's body limit",
            )
            continue

        if entry.state == sync_queue.FAILED and entry.failed_hash is not None:
            current_hash = _file_sha256(entry.pdf_path)
            if current_hash == entry.failed_hash and entry.failure_kind in _REJECTION_KINDS:
                logger.info(
                    "sync: session_id=%s unchanged since the last failure, skipped", entry.session_id
                )
                continue
            if current_hash != entry.content_hash:
                sync_queue._transition(settings.sync_queue_dir, entry.session_id, content_hash=current_hash)
                entry.content_hash = current_hash

        sync_queue.mark_uploading(settings.sync_queue_dir, entry.session_id)
        try:
            push_result = push_synced_report(
                settings, entry.session_id, entry.content_hash, entry.pdf_path
            )
        except ReportSyncCallError as exc:
            kind = _failure_kind(exc)
            reason = _BODY_LIMIT_FAILURE_REASON if kind == sync_queue.FAILURE_BODY_LIMIT else str(exc)
            logger.warning("sync upload failed for session_id=%s (%s): %s", entry.session_id, kind, reason)
            failed_entry = sync_queue.mark_failed(
                settings.sync_queue_dir,
                entry.session_id,
                reason,
                failed_hash=_file_sha256(entry.pdf_path),
                kind=kind,
            )
            result.failed.append((entry.session_id, reason))
            if (
                kind == sync_queue.FAILURE_TRANSIENT
                and failed_entry.failure_count >= settings.sync_max_transient_failures
            ):
                retire(entry.session_id, reason)
            continue

        if settings.sync_reassess_enabled:
            _reassess_isolated(db, settings, entry, push_result.report_id, result)

        sync_queue.mark_sync_confirmed(settings.sync_queue_dir, entry.session_id)
        sync_queue.mark_synced(settings.sync_queue_dir, entry.session_id, push_result.report_id)
        report_index = db.scalar(
            select(ReportIndexRecord).where(
                ReportIndexRecord.session_id == entry.session_id,
                ReportIndexRecord.origin == "edge-local",
            )
        )
        if report_index is not None:
            report_index.report_id = push_result.report_id
            report_index.state = "synced"
            db.add(report_index)
            db.commit()
        logger.info(
            "sync: session_id=%s uploaded, report_id=%s, deduplicated=%s",
            entry.session_id,
            push_result.report_id,
            push_result.deduplicated,
        )
        result.uploaded.append(entry.session_id)

    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    try:
        validate_sync_credentials(settings)
    except SyncCredentialError as exc:
        logger.error("sync refused: %s", exc)
        return 1

    db = SessionLocal()
    try:
        result = run_sync_once(db, settings)
    finally:
        db.close()

    if settings.sync_reassess_enabled:
        logger.info(
            "sync run complete: uploaded=%d failed=%d retired=%d reassessed=%d reassess_failed=%d",
            len(result.uploaded),
            len(result.failed),
            len(result.retired),
            len(result.reassessed),
            len(result.reassess_failed),
        )
    else:
        logger.info(
            "sync run complete: uploaded=%s failed=%s retired=%s",
            len(result.uploaded),
            len(result.failed),
            len(result.retired),
        )
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
