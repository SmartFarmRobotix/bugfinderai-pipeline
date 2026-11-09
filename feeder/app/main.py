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
Session feeder: command-line entry point.

Selects a stored prior-session ("replay session") from the configured local
image store and republishes it as a new inference-session submission. Runs identically on x86 (CI/dev, against the bundled fixture
data root) and the Jetson (real prior-session library).

Usage::

    python -m app.main
    python -m app.main --session-id sess-001
    python -m app.main --no-push --dry-run

Exit codes:
    0 — session replayed (and pushed, if auto_push) successfully.
    1 — fatal configuration/connectivity/store error.
    2 — session submission or push failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Callable, Optional

import httpx

from app import accounting, link_probe, selector
from app.client import SessionFeederClient
from app.config import DeploymentProfile, Settings, get_settings
from app.connection_audit import install as _install_connection_audit
from app.manifest import ManifestError
from app.selection import NoSessionsAvailableError, SessionNotFoundError, record_replayed, select_session_id
from app.store import load_replay_session

_EFFECTIVE_PROFILE_BY_SCENARIO = {
    "cloud": DeploymentProfile.CLOUD,
    "mixed": DeploymentProfile.MIXED,
    "edge": DeploymentProfile.EDGE,
}


def _setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def _write_result_file(
    settings: Settings, session_id: str, push: Optional[Any] = None, decision_block: Optional[dict] = None
) -> None:
    """
    Structured API-response artifact ({session_id, pdm_session_id,
    report_id, pushed}), plus ``link_probe_decision`` for auto-mode sessions.
    ``push`` is None whenever an auto-mode
    session reaches here with no push attempted or a failed push: the decision block must still reach the file whenever
    one is configured, so the push-derived fields degrade to their
    not-pushed defaults rather than the call being skipped.
    """
    if not settings.result_file:
        return
    result: dict[str, Any] = {
        "session_id": session_id,
        "pdm_session_id": push.pdm_session_id if push is not None else None,
        "report_id": push.report_id if push is not None else None,
        "pushed": push.pushed if push is not None else False,
    }
    if decision_block is not None:
        result["link_probe_decision"] = decision_block
    try:
        settings.result_file.write_text(json.dumps(result))
    except OSError as exc:
        logging.getLogger("feeder.main").warning("Could not write result file %s: %s", settings.result_file, exc)


def _decision_block(outcome: link_probe.ProbeOutcome) -> dict[str, Any]:
    """The {samples_kbps, median_kbps, scenario, failure_reason, payload_bytes, rtt_ms, ramp_samples_kbps} shape."""
    scenario_name = selector.select_scenario(median_kbps=outcome.median_kbps, failure_reason=outcome.failure_reason)
    return {
        "samples_kbps": outcome.samples_kbps,
        "median_kbps": outcome.median_kbps,
        "scenario": scenario_name,
        "failure_reason": outcome.failure_reason,
        "payload_bytes": outcome.payload_bytes,
        "rtt_ms": outcome.rtt_ms,
        "ramp_samples_kbps": outcome.ramp_samples_kbps,
    }


def _resolve_effective_profile(
    settings: Settings,
    transport: link_probe.ProbeTransport,
    clock: Callable[[], float],
) -> tuple[Settings, dict[str, Any], int]:
    """Resolve
    REPLAY_DEPLOYMENT_PROFILE=auto to an effective profile in
    {cloud, mixed, edge} -- called exactly once per session, by `run`,
    before any submission machinery. Everything downstream consumes only
    the returned effective settings, never the original `auto` value.
    An auto-resolved CLOUD session gets the cloud target's base URL/key; mixed/edge keep the existing local target
    untouched.
    """
    outcome = link_probe.run_probe(settings, transport, clock)
    decision = _decision_block(outcome)
    effective_profile = _EFFECTIVE_PROFILE_BY_SCENARIO[decision["scenario"]]

    updates: dict[str, Any] = {"deployment_profile": effective_profile}
    if effective_profile is DeploymentProfile.CLOUD:
        updates["inference_base_url"] = settings.cloud_inference_base_url
        updates["inference_api_key"] = settings.effective_cloud_inference_api_key

    effective_settings = settings.model_copy(update=updates)
    return effective_settings, decision, outcome.bytes_written


def _log_accounting_summary(settings: Settings, session_id: str, logger: logging.Logger) -> None:
    """This line is the whole point of the
    accounting record existing -- it must be the *only* source for these
    numbers. If the record does not have a value, this component does not claim it.
    """
    record = accounting.load_record(settings, session_id)
    if not record:
        logger.info("Network accounting: no record produced for session %s", session_id)
        return
    logger.info(
        "Network accounting: session=%s scenario=%s bytes_up=%d bytes_down=%d "
        "mb_total=%.6f (boundary-crossing only; local edge/replay traffic excluded)",
        record["session_id"],
        record["scenario"],
        record["bytes_up"],
        record["bytes_down"],
        record["mb_total"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="session-feeder",
        description="Replay a locally-stored prior-session image batch as a new inference session.",
    )
    parser.add_argument("--session-id", default=None, help="Explicit stored session id to replay (default: deterministic selection).")
    parser.add_argument("--no-push", action="store_true", default=False, help="Skip the push step (submit only).")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Select and validate a session but do not submit it.")
    return parser.parse_args()


def run(
    settings: Settings,
    args: argparse.Namespace,
    *,
    probe_transport: Optional[link_probe.ProbeTransport] = None,
    probe_clock: Optional[Callable[[], float]] = None,
    client_transport: Optional[httpx.BaseTransport] = None,
) -> int:
    logger = logging.getLogger("feeder.main")
    logger.info("session feeder starting")
    logger.info("%s", settings.describe())

    session_id_override = args.session_id or settings.session_id_override
    try:
        session_id = select_session_id(settings.data_root, settings.marker_filename, session_id_override)
    except (NoSessionsAvailableError, SessionNotFoundError) as exc:
        logger.error("Could not select a session to replay: %s", exc)
        return 1

    decision_block: Optional[dict] = None
    if settings.deployment_profile is DeploymentProfile.AUTO:
        transport = probe_transport or link_probe.HttpxProbeTransport(verify=settings.tls_verify)
        clock = probe_clock or time.monotonic
        settings, decision_block, probe_bytes_written = _resolve_effective_profile(settings, transport, clock)
        if decision_block["failure_reason"] is None:
            logger.info(
                "auto profile resolved -> %s (median=%.3f kbps, payload=%d bytes, rtt=%.0f ms)",
                settings.deployment_profile.value,
                decision_block["median_kbps"],
                decision_block["payload_bytes"],
                decision_block["rtt_ms"] or 0,
            )
        else:
            logger.warning(
                "auto profile resolved -> %s (probe failure: %s)",
                settings.deployment_profile.value,
                decision_block["failure_reason"],
            )
        probe_accounting_settings = settings.model_copy(update={"deployment_profile": DeploymentProfile.CLOUD})
        probe_accountant = accounting.ByteAccountant(probe_accounting_settings, session_id)
        probe_accountant.record(
            endpoint=f"{probe_accounting_settings.cloud_inference_base_url}{link_probe.PROBE_SINK_PATH}",
            direction="up",
            kind="probe",
            byte_count=probe_bytes_written,
        )
        probe_accountant.record_decision(decision_block)
        _write_result_file(settings, session_id, None, decision_block)

    try:
        session = load_replay_session(settings.data_root, session_id, settings.manifest_filename)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        logger.error("Rejected session folder %s: %s", session_id, exc)
        return 1

    logger.info(
        "Selected session %s: %d image(s) accepted, %d skipped (unsupported format), %d skipped (unreferenced)",
        session.session_id,
        len(session.images),
        len(session.skipped_unsupported),
        len(session.skipped_extra),
    )

    if not session.images:
        logger.error("Session %s has zero submittable images after format enforcement — nothing to replay", session_id)
        return 1

    if args.dry_run:
        logger.info("DRY RUN — would submit %d image(s) via %s mode; no request sent", len(session.images), settings.ingestion_mode.value)
        for img in session.images:
            logger.info("  - %s%s", img.filename, f" [{img.position.latitude}, {img.position.longitude}]" if img.position else "")
        return 0

    with SessionFeederClient(settings, transport=client_transport) as client:
        logger.info("Connecting to InferenceService at %s ...", settings.inference_base_url)
        if not client.wait_for_inference_service(max_retries=30, retry_delay=2.0):
            logger.error("InferenceService unreachable. Aborting.")
            return 1
        logger.info("InferenceService is healthy.")

        submission = client.submit_session(session)
        if not submission.success:
            logger.error("Session submission failed: %s", submission.error)
            return 2
        logger.info(
            "Session created: id=%s, best_image=%s, score=%.3f (mode=%s)",
            submission.session_id,
            submission.best_image_id,
            submission.infestation_score,
            submission.ingestion_mode,
        )

        record_replayed(settings.data_root, settings.marker_filename, session.session_id)

        push_failed = False
        if settings.auto_push and not args.no_push:
            push = client.push_session(submission.session_id)
            if push.success and push.pushed:
                logger.info("Push succeeded -> %s", push.target_url)
                _write_result_file(settings, submission.session_id, push, decision_block)
            elif push.success and not push.pushed:
                logger.warning("Push completed but PDM push is disabled on the inference service side")
                _write_result_file(settings, submission.session_id, push, decision_block)
            else:
                logger.error("Push failed: %s", push.error)
                push_failed = True
                if decision_block is not None:
                    _write_result_file(settings, submission.session_id, push, decision_block)
        elif decision_block is not None:
            _write_result_file(settings, submission.session_id, None, decision_block)

        if push_failed:
            return 2

    _log_accounting_summary(settings, submission.session_id, logger)

    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()
    _setup_logging(settings.log_level)
    _install_connection_audit("feeder")
    return run(settings, args)


if __name__ == "__main__":
    sys.exit(main())
