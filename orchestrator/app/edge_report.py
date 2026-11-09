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
Edge-local report generation at session close. Reuses `report_payload.compose_report_payload` UNCHANGED
passing the edge score-band `RiskAssessmentRecord` into the
`pdm_assessment` slot and `None` for `fuzzy_assessment` -- so the
designation-branch content (conclusions/recommendations for fine-tuned
sessions, the integration-only notice for stock sessions,
`report_content.py`) is produced by the exact same code cloud/mixed
sessions go through, not a second edge-specific copy.

Renders through the mirrored `report_pdf.pest_session_pdf_core`. The image source is a direct HTTP call to the
edge-local inference service, on the internal network only (never the Gatekeeper):
the edge stack's own network topology is what makes this reachable while blocking all egress, and the inference service's
`gatekeeper_auth` dependency still requires the pre-shared `X-API-Key` even
with Gatekeeper disabled. It also renders a second PDF,
`{session_id}.sync.pdf`, from the same payload with the best-image pair
capped through `image_cap.cap_image_bytes` -- the copy the sync queue
references and uploads, kept under the Gatekeeper's body limit by
construction. The local PDF above keeps its full-resolution variants for
the farmer on the device; rendering the sync copy never touches it. The same capped pair is
also written as `{session_id}.sync-pair.json` (the body of the Reporting
service's best-image-pair upload), so the sync worker can push the pair to
the cloud after the PDF without asking the inference service again -- the
purge may already have removed the images by then. The sync render and the
pair file read each variant once, through one memoising source, so the
bytes in the file are the bytes embedded in the sync copy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

from config import Settings
from contract import ReportingEnvelope, RiskInput
from image_cap import cap_image_bytes
from models import RiskAssessmentRecord
from report_payload import compose_report_payload
from report_pdf.pest_session_pdf_core import PdfRenderError, render_pest_session_pdf

logger = logging.getLogger("orchestrator.edge_report")

_SYNC_IMAGE_MAX_BYTES = 100_000


class EdgeReportError(RuntimeError):
    """Edge-local report generation failed."""


@dataclass
class EdgeReportResult:
    pdf_path: str
    content_hash: str
    images_embedded: bool
    sync_path: str
    sync_sha256: str
    pair_path: Optional[str] = None


def _edge_image_source(settings: Settings, best_image_id: str):
    def _fetch(variant: str) -> Optional[Tuple[bytes, str]]:
        if not best_image_id:
            return None
        base = settings.edge_inference_base_url.rstrip("/")
        url = f"{base}/image/{best_image_id}/{variant}"
        headers = {}
        if settings.edge_inference_api_key:
            headers["X-API-Key"] = settings.edge_inference_api_key
        try:
            response = requests.get(url, headers=headers, timeout=settings.edge_inference_call_timeout_seconds)
        except requests.RequestException as exc:
            logger.warning("edge image fetch (%s) failed: %s", variant, exc)
            return None
        if response.status_code >= 400:
            logger.warning("edge image fetch (%s) failed: status %s", variant, response.status_code)
            return None
        if not response.content:
            return None
        content_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        return response.content, content_type

    return _fetch


def _capped_image_source(source, max_bytes: int = _SYNC_IMAGE_MAX_BYTES):
    """Wraps an `ImageSource` so every variant it returns is capped through
    `cap_image_bytes` -- the content type is only retagged to `image/jpeg`
    when capping actually re-encoded the bytes (an already-small variant
    passes through with its original content type, per `cap_image_bytes`'s
    own unchanged-if-small-enough contract)."""

    def _fetch(variant: str):
        result = source(variant)
        if result is None:
            return None
        image_bytes, content_type = result
        capped = cap_image_bytes(image_bytes, max_bytes)
        if capped is image_bytes:
            return image_bytes, content_type
        return capped, "image/jpeg"

    return _fetch


def _memoised_image_source(source):
    """Wraps an `ImageSource` so each variant is fetched (and capped) at most
    once, a miss included: the sync render and the pair file share the bytes."""
    cache: dict = {}

    def _fetch(variant: str):
        if variant not in cache:
            cache[variant] = source(variant)
        return cache[variant]

    return _fetch


def _write_pair_file(
    settings: Settings,
    session_id: str,
    envelope: ReportingEnvelope,
    contract_version: Optional[str],
    image_source,
) -> Optional[str]:
    """Writes the capped pair beside the sync copy in the body shape of the
    Reporting service's best-image-pair upload; `None` (no file, one warning)
    when either variant is missing."""
    clean = image_source("clean")
    annotated = image_source("annotated")
    if clean is None or annotated is None:
        missing = [name for name, variant in (("clean", clean), ("annotated", annotated)) if variant is None]
        logger.warning(
            "sync pair not written for session_id=%s: image variant unavailable: %s",
            session_id,
            ", ".join(missing),
        )
        return None
    clean_bytes, clean_type = clean
    annotated_bytes, annotated_type = annotated
    body = {
        "contract_version": contract_version,
        "image_id": envelope.best_image_id,
        "artifact_hash": hashlib.sha256(clean_bytes + annotated_bytes).hexdigest(),
        "clean_b64": base64.b64encode(clean_bytes).decode("ascii"),
        "clean_content_type": clean_type,
        "annotated_b64": base64.b64encode(annotated_bytes).decode("ascii"),
        "annotated_content_type": annotated_type,
    }
    pair_path = os.path.join(settings.edge_report_dir, f"{session_id}.sync-pair.json")
    tmp_path = f"{pair_path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    os.replace(tmp_path, pair_path)
    return pair_path


def _content_hash_for_payload(payload: dict) -> str:
    """Identical construction to `service.content_hash_for_payload` --
    duplicated (not imported) to avoid a circular import: `service.py`
    calls into this module, so this module cannot import back from
    `service.py` at module scope."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_edge_local_report(
    settings: Settings,
    session_id: str,
    risk_input: RiskInput,
    envelope: ReportingEnvelope,
    edge_assessment: RiskAssessmentRecord,
    contract_version: Optional[str],
) -> EdgeReportResult:
    """Compose the report payload (unchanged from cloud/mixed) and render it to a local PDF at
    `{settings.edge_report_dir}/{session_id}.pdf`. Raises `EdgeReportError`
    on failure -- never leaves a partial file at the final path (the
    underlying `render_pest_session_pdf` only calls `pdf.output` once,
    which fpdf2 writes atomically enough here; the housekeeping pass (`housekeeping.py`) owns
    atomic visibility of the artifact set as a whole)."""
    payload = compose_report_payload(
        risk_input=risk_input,
        envelope=envelope,
        pdm_assessment=edge_assessment,
        fuzzy_assessment=None,
        contract_version=contract_version,
        site_label=getattr(settings, "report_site_label", None),
        is_edge=True,
    )
    image_source = _edge_image_source(settings, envelope.best_image_id)
    output_path = os.path.join(settings.edge_report_dir, f"{session_id}.pdf")

    try:
        result = render_pest_session_pdf(payload, image_source, output_path)
    except PdfRenderError as exc:
        raise EdgeReportError(str(exc)) from exc

    sync_image_source = _memoised_image_source(
        _capped_image_source(_edge_image_source(settings, envelope.best_image_id))
    )
    sync_output_path = os.path.join(settings.edge_report_dir, f"{session_id}.sync.pdf")

    try:
        render_pest_session_pdf(payload, sync_image_source, sync_output_path)
    except PdfRenderError as exc:
        raise EdgeReportError(str(exc)) from exc

    with open(sync_output_path, "rb") as sync_file:
        sync_sha256 = hashlib.sha256(sync_file.read()).hexdigest()

    pair_path = _write_pair_file(settings, session_id, envelope, contract_version, sync_image_source)

    return EdgeReportResult(
        pdf_path=output_path,
        content_hash=_content_hash_for_payload(payload),
        images_embedded=result.images_embedded,
        sync_path=sync_output_path,
        sync_sha256=sync_sha256,
        pair_path=pair_path,
    )
