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
An offline run under the fine-tuned designation yields a PDF containing report_content.py's
conclusion/recommendation markers; under the stock designation a PDF with the integration-only
notice and no agricultural conclusion.

Every test here runs with `ORCH_EDGE_INFERENCE_BASE_URL` pointed at a
refused loopback port (nothing listens on it) -- no inference service is
reachable in this test process, so these are genuinely OFFLINE runs: the image
fetch fails closed (returns None per variant) and the report still
completes, exactly like the real edge stack when the inference service is briefly
unavailable. Verifies the actual rendered PDF text (via pypdf), not just
the composed payload dict -- the report_content.py markers must survive
all the way through `pest_session_pdf_core`'s FPDF rendering.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pypdf
from PIL import Image

import edge_report
import sync_queue
from config import DeploymentProfile
from contract import CONTRACT_VERSION, DetectorProvenance, GrowthStage, ReportingEnvelope, RiskInput, SessionSubmission
from models import ReportIndexRecord, RiskAssessmentRecord, SessionState
from report_content import (
    CONCLUSION_MARKER,
    FUSED_VERDICT_AUTHORITATIVE_MARKER,
    FUSED_VERDICT_COMPARATIVE_MARKER,
    FUSED_VERDICT_EDGE_MARKER,
    RECOMMENDATION_MARKER,
    STOCK_MODEL_NOTICE_MARKER,
)
from report_payload import compose_report_payload
from report_pdf.pest_session_pdf_core import _ASSETS_DIR, render_pest_session_pdf
from service import advance_session, intake_submission


class _EdgeSettings:
    deployment_profile = DeploymentProfile.EDGE
    max_outbound_retries = 3
    fusion_rule_version = "v1-max-of-bands"
    edge_inference_call_timeout_seconds = 5

    def __init__(self, edge_report_dir: str):
        self.edge_report_dir = edge_report_dir
        self.edge_inference_base_url = "http://127.0.0.1:1"
        self.edge_inference_api_key = "test-edge-inference-key"
        self.sync_queue_dir = edge_report_dir + "-sync-queue"


def _submission(
    session_id: str,
    designation_status: str,
    infestation_score: float = 0.85,
) -> SessionSubmission:
    risk_input = RiskInput(
        session_id=session_id,
        infestation_score=infestation_score,
        crop="brassica-oleracea",
        growth_stage=GrowthStage.unknown_for("enum"),
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
    )
    envelope = ReportingEnvelope(
        session_id=session_id,
        best_image_id="img-1",
        image_count=2,
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-01T01:00:00Z",
        latitude=45.0,
        longitude=10.0,
        provenance=DetectorProvenance(
            detector_impl_id="lookup-detector",
            detector_version="1.0.0",
            designation_status=designation_status,
            aggregation_rule_version="topfrac-v1",
            target_class_set=["brassica_seedling"],
            confidence_floor=0.3,
        ),
    )
    return SessionSubmission(contract_version=CONTRACT_VERSION, risk_input=risk_input, reporting_envelope=envelope)


def _pdf_text(pdf_path) -> str:
    """Matching against the pinned multi-word sentence literals must be WHITESPACE-
    NORMALIZED -- pypdf's `extract_text`, like `pdftotext`, pads fpdf2's
    multi_cell prose with extra interior spaces (observed live: "Note:
    detections in this report..."), which a raw substring match would
    fail against a correctly rendered report."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_path.read_bytes()))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return " ".join(raw.split())


class TestEdgeReportGeneration:
    def test_full_edge_session_reaches_reported_with_local_pdf(self, db, tmp_path):
        session_id = "edge-report-basic"
        intake_submission(db, _submission(session_id, designation_status="designated"))

        result = advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        assert result.state == SessionState.REPORTED.value
        assert result.park_reason is None
        assert result.failure_reason is None

        pdf_path = tmp_path / f"{session_id}.pdf"
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")

        report_index = db.query(ReportIndexRecord).filter_by(session_id=session_id).one()
        assert report_index.origin == "edge-local"
        assert report_index.state == "generated"
        assert report_index.content_hash

    def test_fine_tuned_designation_yields_conclusion_and_recommendation_markers(self, db, tmp_path):
        session_id = "edge-report-designated"
        intake_submission(db, _submission(session_id, designation_status="designated", infestation_score=0.85))

        advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        text = _pdf_text(tmp_path / f"{session_id}.pdf")
        assert CONCLUSION_MARKER in text
        assert RECOMMENDATION_MARKER in text
        assert STOCK_MODEL_NOTICE_MARKER not in text

    def test_stock_designation_yields_integration_only_notice_and_no_agricultural_content(self, db, tmp_path):
        session_id = "edge-report-stock"
        intake_submission(db, _submission(session_id, designation_status="stub", infestation_score=0.85))

        advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        text = _pdf_text(tmp_path / f"{session_id}.pdf")
        assert STOCK_MODEL_NOTICE_MARKER in text
        assert CONCLUSION_MARKER not in text
        assert RECOMMENDATION_MARKER not in text
        assert "version 1.0.0" in text
        assert "v1.0.0" not in text

    def test_integration_only_status_also_yields_stock_notice(self, db, tmp_path):
        """Designation boundary: only exactly "designated" counts as
        validated for agricultural claims (report_payload.py's
        DESIGNATED_STATUS) -- "integration-only" (a real detector, just not
        yet field-validated) suppresses agricultural content exactly like
        "stub" does."""
        session_id = "edge-report-integration-only"
        intake_submission(db, _submission(session_id, designation_status="integration-only", infestation_score=0.85))

        advance_session(db, session_id, settings=_EdgeSettings(str(tmp_path)))

        text = _pdf_text(tmp_path / f"{session_id}.pdf")
        assert STOCK_MODEL_NOTICE_MARKER in text
        assert CONCLUSION_MARKER not in text

    def test_report_generation_failure_marks_session_failed(self, db, tmp_path, monkeypatch):
        """An unwritable report directory is a whole-session failure, not a silent no-op."""
        session_id = "edge-report-failure"
        intake_submission(db, _submission(session_id, designation_status="designated"))

        unwritable = tmp_path / "nested" / "reports"
        (tmp_path / "nested").write_text("not a directory")

        result = advance_session(db, session_id, settings=_EdgeSettings(str(unwritable)))

        assert result.state == SessionState.FAILED.value
        assert result.failure_reason is not None


class TestSyncCopyWithinBodyLimit:
    """The edge orchestrator
    renders a full-resolution local PDF (unchanged) and a second, capped
    sync copy from the same payload -- the copy the sync queue will
    reference and the Gatekeeper's body limit actually bounds. No fixture
    bytes in git -- the cells generate their own JPEGs with Pillow."""

    @staticmethod
    def _large_noisy_jpeg(seed: int, width: int = 4056, height: int = 3040, quality: int = 90) -> bytes:
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels, mode="RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def _embedded_jpeg_sizes(pdf_path: Path) -> list:
        """Exact byte sizes of every /DCTDecode (JPEG) stream embedded in
        `pdf_path`, via poppler's `pdfimages -j` -- a verbatim extraction
        (proven byte-identical to the source JPEG, not a re-encode).
        `-j` only writes a `.jpg` for actual JPEG streams; the report's
        other embedded assets (the bundled PNG logos) come out as `.ppm`
        (decompressed) and are not the pair under test."""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "img")
            subprocess.run(["pdfimages", "-j", str(pdf_path), prefix], check=True, capture_output=True)
            return [
                os.path.getsize(os.path.join(tmp, name))
                for name in sorted(os.listdir(tmp))
                if name.startswith("img") and name.endswith(".jpg")
            ]

    def test_local_full_weight_sync_capped_local_unchanged_by_sync_render(self, tmp_path, monkeypatch):
        session_id = "edge-report-sync-cap"
        submission = _submission(session_id, designation_status="designated")
        settings = _EdgeSettings(str(tmp_path))

        clean_bytes = self._large_noisy_jpeg(seed=1)
        annotated_bytes = self._large_noisy_jpeg(seed=2)

        def _fake_edge_image_source(_settings, _best_image_id):
            def _fetch(variant):
                return (clean_bytes, "image/jpeg") if variant == "clean" else (annotated_bytes, "image/jpeg")

            return _fetch

        monkeypatch.setattr(edge_report, "_edge_image_source", _fake_edge_image_source)

        local_path = tmp_path / f"{session_id}.pdf"
        local_snapshots = []
        original_render = edge_report.render_pest_session_pdf

        def _tracking_render(payload, image_source, output_path):
            render_result = original_render(payload, image_source, output_path)
            if local_path.exists():
                local_snapshots.append(local_path.read_bytes())
            return render_result

        monkeypatch.setattr(edge_report, "render_pest_session_pdf", _tracking_render)

        edge_row = RiskAssessmentRecord(
            session_id=session_id,
            path="edge-score-band",
            status="present",
            level=None,
            fused_verdict="High",
            fusion_rule_version="v1-max-of-bands",
            engine_identity="edge-score-band",
            engine_version="v1",
            weather_window=None,
        )

        result = edge_report.generate_edge_local_report(
            settings,
            session_id,
            submission.risk_input,
            submission.reporting_envelope,
            edge_row,
            submission.contract_version,
        )

        sync_path = Path(result.sync_path)
        assert local_path.is_file()
        assert sync_path.is_file()
        assert sync_path.name == f"{session_id}.sync.pdf"

        assert local_path.stat().st_size > (len(clean_bytes) + len(annotated_bytes)) - 5_000

        assert sync_path.stat().st_size < 2_000_000
        embedded_sizes = self._embedded_jpeg_sizes(sync_path)
        assert len(embedded_sizes) == 2
        for size in embedded_sizes:
            assert size <= 100_000

        assert len(local_snapshots) == 2
        assert local_snapshots[0] == local_snapshots[1] == local_path.read_bytes()


class TestSyncPairFile:
    """The capped pair is
    saved as `{session_id}.sync-pair.json` beside the sync copy, in the body
    shape of the Reporting service's best-image-pair upload, from the same
    bytes the sync copy embeds."""

    _EDGE_ROW = dict(
        path="edge-score-band",
        status="present",
        level=None,
        fused_verdict="High",
        fusion_rule_version="v1-max-of-bands",
        engine_identity="edge-score-band",
        engine_version="v1",
        weather_window=None,
    )

    @staticmethod
    def _solid_jpeg(shade: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (shade % 256, (shade * 7) % 256, (shade * 13) % 256)).save(buf, format="JPEG")
        return buf.getvalue()

    def _generate(self, tmp_path, monkeypatch, fetch_for):
        session_id = "edge-report-pair"
        submission = _submission(session_id, designation_status="designated")

        def _fake_edge_image_source(_settings, _best_image_id):
            return fetch_for

        monkeypatch.setattr(edge_report, "_edge_image_source", _fake_edge_image_source)
        edge_row = RiskAssessmentRecord(session_id=session_id, **self._EDGE_ROW)
        return edge_report.generate_edge_local_report(
            _EdgeSettings(str(tmp_path)),
            session_id,
            submission.risk_input,
            submission.reporting_envelope,
            edge_row,
            submission.contract_version,
        )

    def test_pair_file_has_the_upload_body_shape_and_the_capped_bytes(self, tmp_path, monkeypatch):
        clean_bytes = TestSyncCopyWithinBodyLimit._large_noisy_jpeg(seed=1)
        annotated_bytes = TestSyncCopyWithinBodyLimit._large_noisy_jpeg(seed=2)

        def _fetch(variant):
            return (clean_bytes, "image/jpeg") if variant == "clean" else (annotated_bytes, "image/jpeg")

        result = self._generate(tmp_path, monkeypatch, _fetch)

        pair_path = tmp_path / "edge-report-pair.sync-pair.json"
        assert result.pair_path == str(pair_path)
        body = json.loads(pair_path.read_text())
        assert set(body) == {
            "contract_version",
            "image_id",
            "artifact_hash",
            "clean_b64",
            "clean_content_type",
            "annotated_b64",
            "annotated_content_type",
        }
        assert body["contract_version"] == CONTRACT_VERSION
        assert body["image_id"] == "img-1"
        clean = base64.b64decode(body["clean_b64"])
        annotated = base64.b64decode(body["annotated_b64"])
        assert 0 < len(clean) <= 100_000
        assert 0 < len(annotated) <= 100_000
        assert body["clean_content_type"] == "image/jpeg"
        assert body["annotated_content_type"] == "image/jpeg"
        assert body["artifact_hash"] == hashlib.sha256(clean + annotated).hexdigest()

    def test_pair_bytes_are_the_bytes_the_sync_copy_embeds(self, tmp_path, monkeypatch):
        fetches = []

        def _fetch(variant):
            fetches.append(variant)
            return self._solid_jpeg(len(fetches) * 17), "image/jpeg"

        result = self._generate(tmp_path, monkeypatch, _fetch)

        body = json.loads(Path(result.pair_path).read_text())
        pair_bytes = {base64.b64decode(body["clean_b64"]), base64.b64decode(body["annotated_b64"])}
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "img")
            subprocess.run(["pdfimages", "-j", result.sync_path, prefix], check=True, capture_output=True)
            embedded = {
                Path(tmp, name).read_bytes() for name in os.listdir(tmp) if name.endswith(".jpg")
            }
        assert embedded == pair_bytes
        assert sorted(fetches).count("clean") == 2
        assert sorted(fetches).count("annotated") == 2

    def test_missing_variant_writes_no_file_and_warns(self, tmp_path, monkeypatch, caplog):
        def _fetch(variant):
            return (self._solid_jpeg(5), "image/jpeg") if variant == "clean" else None

        logging.getLogger("orchestrator.edge_report").disabled = False
        with caplog.at_level("WARNING", logger="orchestrator.edge_report"):
            result = self._generate(tmp_path, monkeypatch, _fetch)

        assert result.pair_path is None
        assert not (tmp_path / "edge-report-pair.sync-pair.json").exists()
        warnings = [r.getMessage() for r in caplog.records if "sync pair not written" in r.getMessage()]
        assert len(warnings) == 1
        assert "annotated" in warnings[0]

    def test_service_queues_the_pair_path_on_the_entry(self, db, tmp_path, monkeypatch):
        session_id = "edge-report-pair-queued"
        intake_submission(db, _submission(session_id, designation_status="designated"))

        def _fake_edge_image_source(_settings, _best_image_id):
            return lambda variant: (self._solid_jpeg(3), "image/jpeg")

        monkeypatch.setattr(edge_report, "_edge_image_source", _fake_edge_image_source)
        settings = _EdgeSettings(str(tmp_path))

        advance_session(db, session_id, settings=settings)

        entry = sync_queue.read_entry(Path(settings.sync_queue_dir), session_id)
        assert entry.pair_path == str(tmp_path / f"{session_id}.sync-pair.json")
        assert Path(entry.pair_path).is_file()

    def test_service_queues_no_pair_path_when_the_images_are_unreachable(self, db, tmp_path):
        session_id = "edge-report-pair-none"
        intake_submission(db, _submission(session_id, designation_status="designated"))
        settings = _EdgeSettings(str(tmp_path))

        advance_session(db, session_id, settings=settings)

        entry = sync_queue.read_entry(Path(settings.sync_queue_dir), session_id)
        assert entry.pair_path is None


_OFFICIAL_LOGO_SHA256 = "ecf04b07fa49822333e4bb68310d0b5d1f36fdea66e338e703c474fd1137fe45"

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_SAMPLE_MIRROR_PAYLOAD = {
    "envelope": {
        "session_id": "mirror-pdf-core-test-session",
        "best_image_id": "img-1",
        "time_start": "2026-01-13T07:23:26.820000+00:00",
        "time_end": "2026-01-13T07:57:49.040000+00:00",
        "latitude": 45.0,
        "longitude": 10.0,
        "provenance": {
            "detector_impl_id": "lookup-detector",
            "detector_version": "1.0.0",
            "designation_status": "stub",
            "target_class_set": ["brassica_seedling"],
            "confidence_floor": 0.3,
        },
    },
    "infestation_score": 0.55,
    "risk_assessment": {"level": "Medium", "details": "test"},
    "pest_crop_context": None,
    "risk_reasoning": {"nn_score": 0.55, "nn_verdict": "Moderate", "nn_note": "test"},
    "weather_summary": None,
    "forecast_outlook": None,
}


def _render(payload, with_image=False):
    def image_source(variant: str):
        if with_image and variant == "clean":
            return (_TINY_PNG, "image/png")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        output_path = str(Path(tmp) / "mirror_test.pdf")
        render_pest_session_pdf(payload, image_source, output_path)
        reader = pypdf.PdfReader(output_path)
        raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw, "".join(raw.split())


def _payload(**overrides):
    payload = copy.deepcopy(_SAMPLE_MIRROR_PAYLOAD)
    payload.update(overrides)
    return payload


class TestMirrorRendererParityWithRS:
    """The same POSITIVE extracted-text properties
    the Reporting service suite asserts, run through the mirror's own render path."""

    def test_window_renders_human_readable_no_iso_microseconds(self):
        raw, squeezed = _render(_payload())
        assert "13Jan2026,07:23-07:57UTC" in squeezed
        assert "2026-01-13T07:23:26.820000" not in raw
        assert "2026-01-13T07:57:49.040000" not in raw

    def test_location_with_site_label(self):
        raw, squeezed = _render(_payload(site_label="Field site"))
        assert "Fieldsite" in squeezed
        assert "45.0" in raw
        assert "10.0" in raw

    def test_location_position_unavailable(self):
        payload = _payload()
        payload["envelope"] = dict(payload["envelope"])
        payload["envelope"].pop("latitude")
        payload["envelope"].pop("longitude")
        raw, _ = _render(payload)
        assert "Position unavailable for this session" in raw

    @staticmethod
    def _render_variants(*, clean=True, annotated=True):
        def image_source(variant: str):
            available = clean if variant == "clean" else annotated
            return (_TINY_PNG, "image/png") if available else None

        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "variants_test.pdf")
            result = render_pest_session_pdf(_payload(), image_source, output_path)
            reader = pypdf.PdfReader(output_path)
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
        return result, " ".join(raw.split())

    def test_both_variants_render_with_their_captions_once_each_in_order(self):
        result, text = self._render_variants()
        assert text.count("Session image") == 1
        assert text.count("Best image with the detected regions") == 1
        assert len(_CLEAN_CAPTION.findall(text)) == 1
        positions = [
            text.index("Session image"),
            _CLEAN_CAPTION.search(text).start(),
            text.index("Best image with the detected regions"),
        ]
        assert positions == sorted(positions)
        for old in (
            "Session Images",
            "RGB capture",
            "Detection Overlay (bounding boxes)",
            "Capture",
            "Session images unavailable",
        ):
            assert old not in text, old
        assert result.images_embedded is True

    def test_missing_annotated_variant_renders_only_the_first_caption(self):
        result, text = self._render_variants(annotated=False)
        assert len(_CLEAN_CAPTION.findall(text)) == 1
        assert "Best image with the detected regions" not in text
        assert result.images_embedded is False

    def test_missing_clean_variant_renders_only_the_annotated_caption(self):
        result, text = self._render_variants(clean=False)
        assert text.count("Best image with the detected regions") == 1
        assert not _CLEAN_CAPTION.search(text)
        assert result.images_embedded is False

    def test_no_variant_says_so_and_reports_not_embedded(self):
        result, text = self._render_variants(clean=False, annotated=False)
        assert "Session images unavailable; report generated without images." in text
        assert "Best image" not in text
        assert result.images_embedded is False

    def test_a_variant_that_cannot_be_decoded_is_not_counted_as_embedded(self):
        def image_source(variant: str):
            return (_TINY_PNG, "image/png") if variant == "clean" else (b"not an image", "image/png")

        with tempfile.TemporaryDirectory() as tmp:
            result = render_pest_session_pdf(_payload(), image_source, str(Path(tmp) / "bad.pdf"))
        assert result.images_embedded is False

    def test_detector_line_renders_version_word_not_glued_token(self):
        """The Detector row spells out
        "version", never the renderer-glued "v"+value token."""
        raw, squeezed = _render(_payload())
        assert "version 1.0.0" in raw
        assert "v1.0.0" not in squeezed

    def test_allowlist_line_labeled_legibly_with_default_classes(self):
        raw, _ = _render(_payload())
        assert "Scoring allowlist" in raw
        assert "brassica_seedling" in raw
        assert "Target classes" not in raw

    def test_allowlist_renders_a_non_default_configured_list_verbatim(self):
        """A renderer hardcoding today's default cannot pass this."""
        payload = _payload()
        payload["envelope"] = dict(payload["envelope"])
        payload["envelope"]["provenance"] = dict(payload["envelope"]["provenance"])
        payload["envelope"]["provenance"]["target_class_set"] = ["aphid", "x"]
        raw, _ = _render(payload)
        assert "aphid, x" in raw

    def test_engine_provenance_block_renders_when_present(self):
        payload = _payload()
        payload["risk_reasoning"] = dict(payload["risk_reasoning"])
        payload["risk_reasoning"]["engine_provenance"] = [
            "authoritative path: pdm-rule-engine v0.2.0, fusion v1-max-of-bands",
            "comparative path: fuzzy-mamdani-risk v2.0.0, fusion v1-max-of-bands",
        ]
        raw, _ = _render(payload)
        assert "Engine provenance:" in raw
        assert "pdm-rule-engine v0.2.0" in raw
        assert "fuzzy-mamdani-risk v2.0.0" in raw

    def test_engine_provenance_absent_for_old_payloads(self):
        raw, _ = _render(_payload())
        assert "Engine provenance:" not in raw

    def test_engine_provenance_malformed_shapes_never_crash(self):
        for bad in (7, "a bare string", {"engine": "x"}):
            payload = _payload()
            payload["risk_reasoning"] = dict(payload["risk_reasoning"])
            payload["risk_reasoning"]["engine_provenance"] = bad
            raw, _ = _render(payload)
            assert "Engine provenance:" not in raw

    def test_site_label_malformed_type_ignored_never_crashes(self):
        raw, _ = _render(_payload(site_label={"a": 1}))
        assert "45.0" in raw
        assert "{'a'" not in raw

    def test_weather_data_wins_over_contradictory_reason(self):
        payload = _payload(
            weather_summary={
                "source": "open-meteo (history)",
                "observed": {"temperature_2m_max_C": 4.1},
                "unavailable_reason": "should never render",
            }
        )
        raw, _ = _render(payload)
        assert "Temp max: 4.1 C" in raw
        assert "Weather data unavailable" not in raw
        assert "should never render" not in raw

    def test_weather_unavailable_reason_sentence(self):
        payload = _payload(weather_summary={"unavailable_reason": "position unavailable"})
        raw, _ = _render(payload)
        assert "Weather data unavailable for this session:" in raw
        assert "position unavailable" in raw

    def test_populated_weather_renders_figures_and_source(self):
        payload = _payload(
            weather_summary={
                "source": "openmeteo",
                "variables": ["temperature_2m", "relative_humidity_2m"],
                "observed": {
                    "temperature_2m_max_C": 4.1,
                    "temperature_2m_min_C": -1.3,
                    "relative_humidity_2m_avg_pct": 81,
                },
            }
        )
        raw, _ = _render(payload)
        assert "Temp max: 4.1 C" in raw
        assert "Temp min: -1.3 C" in raw
        assert "Humidity avg: 81 %" in raw
        assert "openmeteo" in raw

    def test_one_malformed_side_still_formats_the_other(self):
        """Per-side independent window fallback -- one malformed boundary never drags the other
        back to machine format (mirrors the Reporting service's own test of the same name)."""
        payload = _payload()
        payload["envelope"] = dict(payload["envelope"])
        payload["envelope"]["time_start"] = "not-a-timestamp"
        raw, squeezed = _render(payload)
        assert "13Jan202607:57UTC" in squeezed
        assert "2026-01-13T07:57:49.040000" not in raw

    def test_weather_window_row_formats_the_unicode_arrow_separator(self):
        """`format_window_string`'s composer-shared
        '{A} → {B}' shape, exercised through the mirror's own render path
        (the Reporting service's suite only exercises this via the Reporting service's own core)."""
        payload = _payload()
        payload["risk_reasoning"] = dict(payload["risk_reasoning"])
        payload["risk_reasoning"]["weather_window"] = (
            "2026-01-13T07:23:26.820000+00:00  →  2026-01-13T07:57:49.040000+00:00"
        )
        raw, squeezed = _render(payload)
        assert "WeatherWindow13Jan2026,07:23-07:57UTC" in squeezed
        assert "2026-01-13T07:23:26.820000" not in raw
        assert "2026-01-13T07:57:49.040000" not in raw

    def test_weather_window_row_formats_the_ascii_arrow_separator(self):
        """The '->' separator, tolerated identically
        to '→' per `format_window_string`'s docstring."""
        payload = _payload()
        payload["risk_reasoning"] = dict(payload["risk_reasoning"])
        payload["risk_reasoning"]["weather_window"] = (
            "2026-01-13T07:23:26.820000+00:00 -> 2026-01-13T07:57:49.040000+00:00"
        )
        raw, squeezed = _render(payload)
        assert "WeatherWindow13Jan2026,07:23-07:57UTC" in squeezed
        assert "2026-01-13T07:23:26.820000" not in raw
        assert "2026-01-13T07:57:49.040000" not in raw

    def test_weather_window_single_iso_timestamp_formats_alone(self):
        """a single parseable timestamp (no
        separator) formats on its own."""
        payload = _payload()
        payload["risk_reasoning"] = dict(payload["risk_reasoning"])
        payload["risk_reasoning"]["weather_window"] = "2026-01-13T07:23:26.820000+00:00"
        raw, squeezed = _render(payload)
        assert "WeatherWindow13Jan202607:23UTC" in squeezed
        assert "2026-01-13T07:23:26.820000" not in raw

    def test_weather_window_unrecognized_shape_passes_through_verbatim(self):
        """`format_window_string`'s documented
        fallback -- anything that is neither an arrow-joined pair nor a
        single parseable timestamp passes through VERBATIM, by design."""
        payload = _payload()
        payload["risk_reasoning"] = dict(payload["risk_reasoning"])
        payload["risk_reasoning"]["weather_window"] = "unrecognized-window-shape"
        raw, _ = _render(payload)
        assert "unrecognized-window-shape" in raw

    def test_mirror_logo_asset_matches_the_official_digest(self):
        logo_bytes = (Path(_ASSETS_DIR) / "openagri_logo.png").read_bytes()
        assert hashlib.sha256(logo_bytes).hexdigest() == _OFFICIAL_LOGO_SHA256


class TestForecastOutlookTableAndRecommendationLabel:
    """a zero-row outlook (exactly
    the shape `report_payload.py` composes -- `{"recommendation": ...}`,
    no `days` key at all) suppresses the header-only table but still
    renders the recommendation; a populated outlook still renders its
    table."""

    def test_recommendation_only_outlook_suppresses_the_empty_table(self):
        payload = _payload(
            forecast_outlook={"recommendation": f"{RECOMMENDATION_MARKER}monitor the crop"}
        )
        raw, _ = _render(payload)
        assert f"{RECOMMENDATION_MARKER}monitor the crop" in raw
        assert "Wind m/s" not in raw

    def test_populated_outlook_still_renders_its_table(self):
        payload = _payload(
            forecast_outlook={
                "days": [
                    {
                        "date": "2026-01-14",
                        "temp_max_C": 5.0,
                        "temp_min_C": -2.0,
                        "humidity_pct": 80,
                        "precipitation_mm": 0.0,
                        "wind_speed_ms": 3.0,
                        "risk_level": "Moderate",
                    }
                ],
                "recommendation": f"{RECOMMENDATION_MARKER}monitor the crop",
            }
        )
        raw, _ = _render(payload)
        assert "Wind m/s" in raw
        assert "2026-01-14" in raw
        assert f"{RECOMMENDATION_MARKER}monitor the crop" in raw

    def test_recommendation_renders_exactly_once_without_doubled_label(self):
        """The renderer no longer
        prefixes "Recommendation: " onto the composed sentence, which
        already begins with the pinned RECOMMENDATION_MARKER -- a count
        assertion, since bare presence would still pass a doubled
        render."""
        payload = _payload(
            forecast_outlook={"recommendation": f"{RECOMMENDATION_MARKER}monitor the crop"}
        )
        raw, squeezed = _render(payload)
        sentence_squeezed = "".join(f"{RECOMMENDATION_MARKER}monitor the crop".split())
        assert squeezed.count(sentence_squeezed) == 1
        doubled_squeezed = "".join("Recommendation:".split()) + "".join(RECOMMENDATION_MARKER.split())
        assert doubled_squeezed not in squeezed


class TestEdgeVerdictSentenceUsesThePinnedEdgeSurface:
    """The edge verdict sentence uses the pinned EDGE
    surface (score-band), never the weather-rule surface, and no fabricated
    comparative sentence."""

    def test_edge_composed_payload_renders_the_edge_surface_only(self):
        risk_input = RiskInput(
            session_id="edge-verdict-sentence",
            infestation_score=0.7,
            crop="brassica-oleracea",
            growth_stage=GrowthStage.unknown_for("enum"),
            time_start="2026-01-01T00:00:00Z",
            time_end="2026-01-01T01:00:00Z",
            latitude=45.0,
            longitude=10.0,
        )
        envelope = ReportingEnvelope(
            session_id="edge-verdict-sentence",
            best_image_id="img-1",
            image_count=1,
            time_start="2026-01-01T00:00:00Z",
            time_end="2026-01-01T01:00:00Z",
            latitude=45.0,
            longitude=10.0,
            provenance=DetectorProvenance(
                detector_impl_id="lookup-detector",
                detector_version="1.0.0",
                designation_status="designated",
                aggregation_rule_version="topfrac-v1",
                target_class_set=["brassica_seedling"],
                confidence_floor=0.3,
            ),
        )
        edge_row = RiskAssessmentRecord(
            session_id="edge-verdict-sentence",
            path="edge-score-band",
            status="present",
            level=None,
            fused_verdict="High",
            fusion_rule_version="v1-max-of-bands",
            engine_identity="edge-score-band",
            engine_version="v1",
            weather_window=None,
        )
        payload = compose_report_payload(
            risk_input=risk_input,
            envelope=envelope,
            pdm_assessment=edge_row,
            fuzzy_assessment=None,
            contract_version=CONTRACT_VERSION,
            is_edge=True,
        )

        raw, squeezed = _render(payload)
        assert f"{''.join(FUSED_VERDICT_EDGE_MARKER.split())}High" in squeezed
        assert "".join(FUSED_VERDICT_AUTHORITATIVE_MARKER.split()) not in squeezed
        assert "".join(FUSED_VERDICT_COMPARATIVE_MARKER.split()) not in squeezed


def _layout_lines(payload) -> list:
    """`pdftotext -layout` lines of the rendered report, each with its
    whitespace collapsed."""
    import subprocess

    def image_source(variant: str):
        return (_TINY_PNG, "image/png")

    with tempfile.TemporaryDirectory() as tmp:
        output_path = str(Path(tmp) / "layout_test.pdf")
        render_pest_session_pdf(payload, image_source, output_path)
        out = subprocess.run(
            ["pdftotext", "-layout", output_path, "-"], capture_output=True, text=True, check=True
        ).stdout
    return [" ".join(line.split()) for line in out.splitlines()]


_CLEAN_CAPTION = re.compile(r"Best image(?! with the detected regions)")

_FORMER_VERDICT_PREFIXES = (
    "Authoritative assess" + "ment (",
    "Comparative assess" + "ment (",
)

_FARMER_ORDER = (
    "Session Overview",
    "Risk Assessment",
    "Pest & Crop Context",
    "Risk Reasoning",
    "Weather Summary",
    "Forecast risk outlook: the next 5 days",
    "Session image",
    "Inference Details",
)


class TestMirrorSectionOrderAndVerdictRows:
    """The mirror core lays the report
    out exactly like the Reporting service's (the two cores are byte-mirrored
    below the docstring; these cells prove the mirror's own render path)."""

    def _payload(self, note_lines, engine_provenance):
        payload = _payload(
            pest_crop_context={
                "pest_name": "Cabbage aphid",
                "pest_scientific_name": "Brevicoryne brassicae",
                "pest_eppo_code": "BRVCBR",
            },
            forecast_outlook={"recommendation": "Recommended action: monitor the crop."},
        )
        payload["risk_reasoning"] = {
            "nn_score": 0.55,
            "nn_verdict": "Moderate",
            "nn_note": "\n".join(note_lines),
            "engine_provenance": engine_provenance,
        }
        return payload

    def test_cloud_report_sections_are_in_the_farmers_order_with_verdict_rows(self):
        lines = _layout_lines(
            self._payload(
                [
                    "Weather-rule assessment: High",
                    "Fuzzy assessment: High",
                ],
                [
                    "authoritative path: pdm-rule-engine 1.4.0, fusion v1",
                    "comparative path: fuzzy-mamdani 2.0.0, fusion v1",
                ],
            )
        )
        positions = [lines.index(heading) for heading in _FARMER_ORDER]
        assert positions == sorted(positions) and len(set(positions)) == len(positions)
        assert "Weather-rule assessment: High" in lines
        assert "Fuzzy assessment: High" in lines
        assert not any(former in " ".join(lines) for former in _FORMER_VERDICT_PREFIXES)
        image_at, details_at = lines.index("Session image"), lines.index("Inference Details")
        for line in (
            "authoritative path: pdm-rule-engine 1.4.0, fusion v1",
            "comparative path: fuzzy-mamdani 2.0.0, fusion v1",
        ):
            assert lines.index(line) > details_at > image_at

    def test_no_raw_variable_identifier_reaches_the_extracted_text(self):
        lines = _layout_lines(
            _payload(
                weather_summary={
                    "source": "open-meteo (history)",
                    "variables": ["temperature_2m", "relative_humidity_2m", "precipitation", "soil_moisture_0_to_1cm"],
                    "observed": {"temperature_2m_max_C": 4.1, "relative_humidity_2m_avg_pct": 81},
                }
            )
        )
        assert "Variables temperature, relative humidity, precipitation, soil_moisture_0_to_1cm" in lines
        whole = " ".join(lines)
        for identifier in ("temperature_2m", "relative_humidity_2m"):
            assert identifier not in whole

    def test_the_edge_report_keeps_its_single_record_surface(self):
        lines = _layout_lines(
            self._payload(
                ["Edge assessment (score band): Medium"],
                ["authoritative path: edge-score-band v1"],
            )
        )
        assert "Edge assessment (score band): Medium" in lines
        assert not [line for line in lines if line.startswith("Fuzzy assessment")]
        assert not any(former in " ".join(lines) for former in _FORMER_VERDICT_PREFIXES)
        provenance = [line for line in lines if line.startswith(("authoritative path:", "comparative path:"))]
        assert provenance == ["authoritative path: edge-score-band v1"]
        assert lines.index(provenance[0]) > lines.index("Session image")
        assert lines.index("Inference Details") == max(lines.index(h) for h in _FARMER_ORDER)
