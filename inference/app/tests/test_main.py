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
Unit/integration tests for the inference service's session pipeline.

Coverage targets
-----------------
/healthz smoke
create_session (HTTP upload): format filtering, empty-session
  rejection, best-image selection incl. clean fallback,
  detector provenance recorded, growth-stage/crop defaults
create_session_from_path (path-registration): accepted, format
  filtering, unreadable/traversal/symlink containment rejections
GET /api/v1/sessions/{id}/results: per-image evaluation carrier
push_session: contract-shaped submission, missing-field
  guard, override body, GK/direct routing (Cloud scenario regression)
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["INFERENCE_USING_GATEKEEPER"] = "false"
os.environ["INFERENCE_ORCH_PUSH_ENABLED"] = "false"
os.environ["INFERENCE_STORAGE_DIR"] = "/tmp/inf_test_storage"

from app.config import Settings, get_settings          # noqa: E402
from app.main import app, gatekeeper_auth               # noqa: E402
import app.main as main_module                          # noqa: E402
from app.storage import read_session_metadata           # noqa: E402

_REAL_DETECTION_RESULTS = Path(__file__).parent.parent / "detection_results.json"
_SAMPLE_IMAGES_DIR = Path(__file__).parent / "fixtures" / "sample_images"
_NEEDS_SAMPLE_IMAGES = pytest.mark.skipif(
    not _REAL_DETECTION_RESULTS.exists() or not _SAMPLE_IMAGES_DIR.exists(),
    reason="needs the sample pictures under tests/fixtures/sample_images and detection_results.json",
)


def _no_auth():
    return None


def _make_settings(storage_dir: Path, *, data_root: Path | None = None, **overrides) -> Settings:
    kwargs: dict[str, Any] = dict(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(storage_dir),
        INFERENCE_DATA_ROOT=str(data_root or storage_dir),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _client_for(storage_dir: Path, **overrides) -> TestClient:
    settings = _make_settings(storage_dir, **overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = _no_auth
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


base_client = TestClient(app, raise_server_exceptions=True)


class _ProbeRecorder:
    """Detector double for /healthz: ``probe`` records the thread it ran on
    and raises when told to; ``detect`` can block on an event."""

    impl_id = "probe-test-double"
    version = "0"

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.probe_threads: list[str] = []
        self.detect_started = threading.Event()
        self.detect_release = threading.Event()

    def probe(self) -> None:
        self.probe_threads.append(threading.current_thread().name)
        if self.fail_with is not None:
            raise self.fail_with

    def detect(self, image_ref):
        from app.detectors.base import ImageDetections

        self.detect_started.set()
        assert self.detect_release.wait(timeout=10)
        return ImageDetections(image_id=image_ref.image_id, detections=[])


@pytest.fixture
def health_state(monkeypatch):
    """A fresh HealthState per test: the module-level one carries detections
    recorded by every other test of this file."""
    from app.health import HealthState

    state = HealthState()
    monkeypatch.setattr(main_module, "HEALTH_STATE", state)
    return state


class TestHealthEndpoint:
    def _with_detector(self, monkeypatch, detector, tmp_path, **overrides):
        monkeypatch.setattr(main_module, "_get_detector", lambda impl_name: detector)
        return _client_for(tmp_path, **overrides)

    def test_healthz_returns_200(self, health_state):
        resp = base_client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready", "proof": "probe"}

    def test_null_detector_is_always_proven(self, health_state, tmp_path):
        client = _client_for(tmp_path, INFERENCE_DETECTOR_IMPL="null")
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_working_probe_runs_on_the_detector_thread(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder()
        client = self._with_detector(monkeypatch, detector, tmp_path)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["proof"] == "probe"
        assert len(detector.probe_threads) == 1
        assert detector.probe_threads[0].startswith("detector")

    def test_failing_probe_answers_503_with_the_reason(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder(fail_with=RuntimeError("NVML Unknown Error"))
        client = self._with_detector(monkeypatch, detector, tmp_path)
        resp = client.get("/healthz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert "NVML Unknown Error" in body["reason"]
        assert health_state.consecutive_probe_failures == 1

    def test_probe_that_does_not_finish_is_unhealthy(self, health_state, monkeypatch, tmp_path):
        release = threading.Event()

        class _Hung(_ProbeRecorder):
            def probe(self) -> None:
                release.wait(timeout=5)

        client = self._with_detector(
            monkeypatch, _Hung(), tmp_path, INFERENCE_HEALTH_PROBE_TIMEOUT_SECONDS="0.2"
        )
        try:
            resp = client.get("/healthz")
        finally:
            release.set()
        assert resp.status_code == 503
        assert "did not finish" in resp.json()["reason"]
        assert health_state.consecutive_probe_failures == 1

    def test_recent_detection_is_proof_enough(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder(fail_with=RuntimeError("must not run"))
        client = self._with_detector(monkeypatch, detector, tmp_path)
        health_state.record_detection_ok()
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["proof"] == "recent-detection"
        assert detector.probe_threads == []

    def test_old_detection_is_not_proof(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder()
        client = self._with_detector(
            monkeypatch, detector, tmp_path, INFERENCE_HEALTH_PROOF_MAX_AGE_SECONDS="0.05"
        )
        health_state.record_detection_ok()
        time.sleep(0.1)
        resp = client.get("/healthz")
        assert resp.json()["proof"] == "probe"
        assert len(detector.probe_threads) == 1

    def test_recent_probe_is_proof_enough(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder()
        client = self._with_detector(monkeypatch, detector, tmp_path)
        assert client.get("/healthz").json()["proof"] == "probe"
        assert client.get("/healthz").json()["proof"] == "recent-probe"
        assert len(detector.probe_threads) == 1

    def test_in_flight_detection_is_proof_enough(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder(fail_with=RuntimeError("must not run"))
        client = self._with_detector(monkeypatch, detector, tmp_path)
        results: dict = {}

        def _drive() -> None:
            resp = client.post(
                "/internal/detect-debug",
                files={"file": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
            results["status_code"] = resp.status_code

        t = threading.Thread(target=_drive)
        t.start()
        try:
            assert detector.detect_started.wait(timeout=10)
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json()["proof"] == "in-flight"
            assert detector.probe_threads == []
        finally:
            detector.detect_release.set()
            t.join(timeout=10)
        assert results["status_code"] == 200
        assert health_state.in_flight == 0
        assert health_state.last_detection_ok_at is not None

    def test_probe_in_progress_counts_as_in_flight(self, health_state, monkeypatch, tmp_path):
        started = threading.Event()
        release = threading.Event()

        class _Blocking(_ProbeRecorder):
            def probe(self) -> None:
                super().probe()
                started.set()
                assert release.wait(timeout=10)

        detector = _Blocking()
        client = self._with_detector(monkeypatch, detector, tmp_path)
        results: dict = {}

        def _drive() -> None:
            resp = client.get("/healthz")
            results["status_code"] = resp.status_code
            results["body"] = resp.json()

        t = threading.Thread(target=_drive)
        t.start()
        try:
            assert started.wait(timeout=10)
            second = client.get("/healthz")
            assert second.status_code == 200
            assert second.json() == {"status": "ready", "proof": "in-flight"}
            assert len(detector.probe_threads) == 1
        finally:
            release.set()
            t.join(timeout=10)
        assert results["status_code"] == 200
        assert results["body"] == {"status": "ready", "proof": "probe"}
        assert len(detector.probe_threads) == 1
        assert health_state.in_flight == 0

    def test_passing_probe_after_a_failure_resets_the_count(self, health_state, monkeypatch, tmp_path):
        detector = _ProbeRecorder(fail_with=RuntimeError("lost"))
        client = self._with_detector(monkeypatch, detector, tmp_path)
        assert client.get("/healthz").status_code == 503
        assert health_state.consecutive_probe_failures == 1
        detector.fail_with = None
        assert client.get("/healthz").status_code == 200
        assert health_state.consecutive_probe_failures == 0

    def test_consecutive_failures_end_the_process(self, health_state, monkeypatch, tmp_path):
        exits: list[int] = []
        monkeypatch.setattr(main_module.os, "_exit", lambda code: exits.append(code))
        monkeypatch.setattr(main_module, "_EXIT_DELAY_SECONDS", 0.05)
        detector = _ProbeRecorder(fail_with=RuntimeError("lost"))
        client = self._with_detector(
            monkeypatch, detector, tmp_path, INFERENCE_HEALTH_EXIT_AFTER_FAILURES="2"
        )
        with client:
            first = client.get("/healthz")
            assert first.status_code == 503
            time.sleep(0.3)
            assert exits == []
            second = client.get("/healthz")
            assert second.status_code == 503
            assert second.json()["consecutive_failures"] == 2
            deadline = time.monotonic() + 3
            while not exits and time.monotonic() < deadline:
                time.sleep(0.02)
        assert exits == [3]

    def test_exit_setting_zero_never_exits(self, health_state, monkeypatch, tmp_path):
        exits: list[int] = []
        monkeypatch.setattr(main_module.os, "_exit", lambda code: exits.append(code))
        monkeypatch.setattr(main_module, "_EXIT_DELAY_SECONDS", 0.01)
        detector = _ProbeRecorder(fail_with=RuntimeError("lost"))
        client = self._with_detector(
            monkeypatch, detector, tmp_path, INFERENCE_HEALTH_EXIT_AFTER_FAILURES="0"
        )
        with client:
            for _ in range(4):
                assert client.get("/healthz").status_code == 503
            time.sleep(0.2)
        assert exits == []


class TestCreateSessionHttpUpload:
    def _png(self, tag: bytes = b"\x00") -> bytes:
        return b"\x89PNG\r\n\x1a\n" + tag * 50

    def test_create_session_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.post(
                "/api/v1/sessions",
                data={"latitude": "10.0", "longitude": "20.0"},
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"]
        assert body["best_image_id"]
        assert 0.0 <= body["infestation_score"] <= 1.0

    def test_no_files_returns_422(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.post("/api/v1/sessions")
        assert resp.status_code == 422

    def test_unsupported_format_rejected_supported_accepted(self):
        """A mixed batch: only JPEG/PNG are accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                data={"latitude": "10.0", "longitude": "20.0"},
                files=[
                    ("files", ("ok.png", BytesIO(self._png(b"\x01")), "image/png")),
                    ("files", ("bad.tiff", BytesIO(b"II*\x00fake-tiff"), "image/tiff")),
                ],
            )
            assert resp.status_code == 200
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert len(meta["images"]) == 1
        assert meta["images"][0]["filename"] == "ok.png"
        assert meta["rejected_files"] == ["bad.tiff"]

    def test_all_unsupported_format_is_empty_session_rejection(self):
        """Zero processable images -> defined error, no fabricated score."""
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("bad.tiff", BytesIO(b"II*\x00fake"), "image/tiff")},
            )
        assert resp.status_code == 400
        assert "zero processable images" in resp.json()["detail"].lower()

    def test_session_record_has_detector_provenance(self):
        """Session record identifies detector impl id + version."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        prov = meta["provenance"]
        assert prov["detector_impl_id"] == "lookup-detector"
        assert prov["detector_version"]
        assert prov["designation_status"] == "stub"
        assert prov["aggregation_rule_version"]
        assert isinstance(prov["target_class_set"], list)

    def test_session_record_has_contract_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert meta["contract_version"]

    def test_crop_defaults_when_not_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert meta["crop"] == "brassica-oleracea"

    def test_growth_stage_defaults_to_explicit_unknown(self):
        """Operational default is the explicit unknown marker, not absence."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert meta["growth_stage"]["unknown"] is True
        assert meta["growth_stage"]["representation"] == "enum"

    def test_growth_stage_explicit_enum_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                data={"growth_stage_representation": "enum", "growth_stage_stage_id": "flowering"},
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert meta["growth_stage"]["unknown"] is False
        assert meta["growth_stage"]["stage_id"] == "flowering"

    def test_growth_stage_explicit_bbch_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                data={
                    "growth_stage_representation": "bbch_range",
                    "growth_stage_bbch_min": "10",
                    "growth_stage_bbch_max": "19",
                },
                files={"files": ("img.png", BytesIO(self._png()), "image/png")},
            )
            meta = read_session_metadata(tmp_path, resp.json()["session_id"])
        assert meta["growth_stage"]["bbch_min"] == 10
        assert meta["growth_stage"]["bbch_max"] == 19


@pytest.mark.skipif(
    not _REAL_DETECTION_RESULTS.exists() or not _SAMPLE_IMAGES_DIR.exists(),
    reason="requires the shipped sample-image / detection-results fixtures",
)
class TestPestOnTheSession:
    """The pest (an EPPO code) is a
    session-creation field next to `crop`, persisted in the session
    metadata and carried on the reporting envelope -- never on the risk
    input, which is posted verbatim to the PDM."""

    _PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40

    def _create(self, client: TestClient, **form) -> Any:
        data = {
            "latitude": "48.5",
            "longitude": "21.3",
            "time_start": "2026-01-01T00:00:00Z",
            "time_end": "2026-01-01T01:00:00Z",
        }
        data.update(form)
        return client.post(
            "/api/v1/sessions",
            data=data,
            files={"files": ("img.png", BytesIO(self._PNG), "image/png")},
        )

    def _push_settings(self, tmp_path: Path) -> Settings:
        return Settings(
            INFERENCE_USING_GATEKEEPER="false",
            INFERENCE_ORCH_PUSH_ENABLED="true",
            INFERENCE_ORCH_EDGE_SHARED_SECRET="test-edge-shared-secret",
            INFERENCE_STORAGE_DIR=str(tmp_path),
        )

    @patch("app.main._post_json")
    def test_a_pest_travels_from_the_session_to_the_envelope(self, mock_post):
        mock_post.return_value = (200, '{"status":"accepted"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = self._push_settings(tmp_path)
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)

            created = self._create(client, crop="brassica-oleracea", pest="BRVCBR")
            assert created.status_code == 200, created.text
            session_id = created.json()["session_id"]
            assert read_session_metadata(tmp_path, session_id)["pest"] == "BRVCBR"
            assert client.get(f"/api/v1/sessions/{session_id}").json()["pest"] == "BRVCBR"

            with patch.dict(os.environ, {"INFERENCE_ORCH_BASE_URL": "http://orchestrator:8020"}, clear=False):
                pushed = client.post(f"/api/v1/sessions/{session_id}/push")

        assert pushed.status_code == 200 and pushed.json()["pushed"] is True
        payload = mock_post.call_args[0][1]
        assert payload["contract_version"] == "1.5.0"
        assert payload["reporting_envelope"]["pest"] == "BRVCBR"
        assert "pest" not in payload["risk_input"]

    @patch("app.main._post_json")
    def test_a_submission_without_a_pest_validates_and_carries_none(self, mock_post):
        mock_post.return_value = (200, '{"status":"accepted"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = self._push_settings(tmp_path)
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)

            created = self._create(client, crop="brassica-oleracea")
            session_id = created.json()["session_id"]
            assert "pest" not in read_session_metadata(tmp_path, session_id)
            with patch.dict(os.environ, {"INFERENCE_ORCH_BASE_URL": "http://orchestrator:8020"}, clear=False):
                pushed = client.post(f"/api/v1/sessions/{session_id}/push")

        assert pushed.status_code == 200 and pushed.json()["pushed"] is True
        payload = mock_post.call_args[0][1]
        assert payload["contract_version"] == "1.5.0"
        assert "pest" not in payload["reporting_envelope"]

    def test_a_malformed_pest_is_refused_naming_the_field_and_its_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = self._create(client, pest="brevicoryne brassicae")
            sessions_dir_entries = list(tmp_path.iterdir())

        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any(err["loc"][-1] == "pest" for err in errors)
        assert any("^[A-Z0-9]{1,8}$" in err["msg"] for err in errors)
        assert sessions_dir_entries == []

    def test_a_malformed_pest_is_refused_in_path_registration_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.post(
                "/api/v1/sessions/path", json={"path": "/nowhere", "pest": "brevicoryne brassicae"}
            )
        assert resp.status_code == 422
        assert any(err["loc"][-1] == "pest" for err in resp.json()["detail"])

    def test_the_pest_is_persisted_in_path_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = TestPathRegistration()._make_session_folder(root)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(folder), "latitude": 1.0, "longitude": 2.0, "pest": "BRVCBR"},
            )
            assert resp.status_code == 200, resp.text
            meta = read_session_metadata(root / "storage", resp.json()["session_id"])
        assert meta["pest"] == "BRVCBR"


@_NEEDS_SAMPLE_IMAGES
class TestBestImageSelectionRealFixtures:
    def test_infested_session_best_image_has_highest_score(self):
        clean_name = "sample-11.png"
        hot_name = "sample-03.png"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            files = [
                ("files", (clean_name, (_SAMPLE_IMAGES_DIR / clean_name).read_bytes(), "image/png")),
                ("files", (hot_name, (_SAMPLE_IMAGES_DIR / hot_name).read_bytes(), "image/png")),
            ]
            resp = client.post("/api/v1/sessions", files=files)
            assert resp.status_code == 200
            body = resp.json()
            meta = read_session_metadata(tmp_path, body["session_id"])

        best_record = next(r for r in meta["images"] if r["image_id"] == body["best_image_id"])
        assert best_record["filename"] == hot_name
        assert body["infestation_score"] > 0.0

    def test_clean_session_falls_back_to_first_image(self):
        clean_a = "sample-11.png"
        clean_b = "sample-09.png"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path, INFERENCE_CONFIDENCE_FLOOR="0.999")
            files = [
                ("files", (clean_a, (_SAMPLE_IMAGES_DIR / clean_a).read_bytes(), "image/png")),
                ("files", (clean_b, (_SAMPLE_IMAGES_DIR / clean_b).read_bytes(), "image/png")),
            ]
            resp = client.post("/api/v1/sessions", files=files)
            body = resp.json()
            meta = read_session_metadata(tmp_path, body["session_id"])

        assert body["infestation_score"] == 0.0
        best_record = next(r for r in meta["images"] if r["image_id"] == body["best_image_id"])
        assert best_record["filename"] == clean_a


class TestEvaluationCarrier:
    def _png(self, tag: bytes) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + tag * 50

    def test_one_entry_per_image_in_session_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            files = [
                ("files", ("a.png", BytesIO(self._png(b"\x01")), "image/png")),
                ("files", ("b.png", BytesIO(self._png(b"\x02")), "image/png")),
                ("files", ("c.png", BytesIO(self._png(b"\x03")), "image/png")),
            ]
            created = client.post(
                "/api/v1/sessions", data={"latitude": "10.0", "longitude": "20.0"}, files=files
            ).json()
            resp = client.get(f"/api/v1/sessions/{created['session_id']}/results")
            assert resp.status_code == 200
            body = resp.json()
            meta_images = read_session_metadata(tmp_path, created["session_id"])["images"]

        assert len(body["results"]) == 3
        assert [r["image_id"] for r in body["results"]] == [m["image_id"] for m in meta_images]
        for entry in body["results"]:
            assert 0.0 <= entry["score"] <= 1.0
            assert entry["position"] is None
            assert entry["position_provenance"] is None

    def test_results_preserve_only_supplied_positions_and_centroid_reaches_push_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            files = [
                ("files", ("a.png", BytesIO(self._png(b"\x01")), "image/png")),
                ("files", ("b.png", BytesIO(self._png(b"\x02")), "image/png")),
                ("files", ("c.png", BytesIO(self._png(b"\x03")), "image/png")),
            ]
            metadata = {
                "per_image_positions": {
                    "a.png": {"latitude": 10.0, "longitude": 20.0, "provenance": "replay"},
                    "c.png": {"latitude": 12.0, "longitude": 24.0, "provenance": "replay"},
                }
            }
            created = client.post(
                "/api/v1/sessions",
                data={
                    "latitude": "1.0",
                    "longitude": "2.0",
                    "time_start": "2026-01-01T00:00:00Z",
                    "time_end": "2026-01-01T01:00:00Z",
                    "metadata": json.dumps(metadata),
                },
                files=files,
            ).json()
            results = client.get(f"/api/v1/sessions/{created['session_id']}/results").json()["results"]
            stored = read_session_metadata(tmp_path, created["session_id"])

        assert results[0]["position"] == {"latitude": 10.0, "longitude": 20.0}
        assert results[0]["position_provenance"] == "replay"
        assert results[1]["position"] is None
        assert results[1]["position_provenance"] is None
        assert results[2]["position"] == {"latitude": 12.0, "longitude": 24.0}
        assert results[2]["position_provenance"] == "replay"
        assert stored["latitude"] == 11.0
        assert stored["longitude"] == 22.0
        assert stored["position_provenance"] == "centroid"

    def test_no_per_image_positions_retain_explicit_default_session_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            created = client.post(
                "/api/v1/sessions",
                data={"latitude": "10.0", "longitude": "20.0"},
                files={"files": ("a.png", BytesIO(self._png(b"\x01")), "image/png")},
            ).json()
            stored = read_session_metadata(tmp_path, created["session_id"])

        assert stored["latitude"] == 10.0
        assert stored["longitude"] == 20.0
        assert stored["position_provenance"] == "default"

    def test_no_positions_or_session_coordinates_record_absence_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            created = client.post(
                "/api/v1/sessions",
                files={"files": ("a.png", BytesIO(self._png(b"\\x01")), "image/png")},
            ).json()
            stored = read_session_metadata(tmp_path, created["session_id"])

        assert stored["latitude"] is None
        assert stored["longitude"] is None
        assert stored["position_provenance"] is None

    def test_results_for_unknown_session_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.get("/api/v1/sessions/does-not-exist/results")
        assert resp.status_code == 404


class TestPathRegistration:
    def _write_manifest(self, folder: Path, session_id: str, filenames: list[str]) -> None:
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": session_id,
                    "crop": "brassica-oleracea",
                    "session_start": "2026-01-01T00:00:00Z",
                    "session_end": "2026-01-01T01:00:00Z",
                    "images": [{"filename": f} for f in filenames],
                }
            )
        )

    def _make_session_folder(self, root: Path, name: str = "incoming-session") -> Path:
        folder = root / name
        (folder / "images").mkdir(parents=True)
        (folder / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 40)
        (folder / "images" / "b.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x02" * 40)
        (folder / "images" / "bad.tiff").write_bytes(b"II*\x00fake-tiff")
        self._write_manifest(folder, name, ["a.png", "b.jpg", "bad.tiff"])
        return folder

    def test_contained_path_accepted_with_format_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self._make_session_folder(root)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(folder), "latitude": 1.0, "longitude": 2.0},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"]
        assert 0.0 <= body["infestation_score"] <= 1.0

    def test_rejected_when_path_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(root / "does-not-exist")},
            )
        assert resp.status_code == 400
        assert "cannot be resolved or read" in resp.json()["detail"]

    def test_rejected_on_traversal_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(root / ".." / "outside")},
            )
        assert resp.status_code == 400
        assert "escapes the configured data root" in resp.json()["detail"]

    def test_rejected_on_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            link = root / "escape-link"
            link.symlink_to(outside, target_is_directory=True)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(link)},
            )
        assert resp.status_code == 400
        assert "escapes the configured data root" in resp.json()["detail"]

    def test_contained_symlink_within_root_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            real_folder = self._make_session_folder(root, name="real-session")
            link = root / "alias-session"
            link.symlink_to(real_folder, target_is_directory=True)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post("/api/v1/sessions/path", json={"path": str(link)})
        assert resp.status_code == 200

    def test_unsupported_format_rejected_supported_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self._make_session_folder(root)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(folder), "session_id": "path-sess-1"},
            )
            meta = read_session_metadata(root / "storage", "path-sess-1")
        assert resp.status_code == 200
        assert len(meta["images"]) == 2
        assert meta["rejected_files"] == ["bad.tiff"]
        assert meta["ingestion_mode"] == "path_registration"

    def test_missing_manifest_is_rejected(self):
        """
        Path-mode ingestion reads
        manifest.json — a folder without one is rejected, not silently
        directory-scanned.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "no-manifest-session"
            (folder / "images").mkdir(parents=True)
            (folder / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 40)
            client = _client_for(root / "storage", data_root=root)
            resp = client.post("/api/v1/sessions/path", json={"path": str(folder)})
        assert resp.status_code == 400
        assert "manifest" in resp.json()["detail"].lower()

    def test_all_manifest_images_unsupported_format_is_empty_session_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "empty-session"
            (folder / "images").mkdir(parents=True)
            (folder / "images" / "bad.tiff").write_bytes(b"II*\x00fake-tiff")
            self._write_manifest(folder, "empty-session", ["bad.tiff"])
            client = _client_for(root / "storage", data_root=root)
            resp = client.post("/api/v1/sessions/path", json={"path": str(folder)})
        assert resp.status_code == 400

    def test_manifest_order_preserved_and_extra_file_skipped(self):
        """
        Path-mode ingestion must use the
        manifest's declared image order, not a directory scan, and must
        skip a file present under images/ but not referenced by the
        manifest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "order-session"
            (folder / "images").mkdir(parents=True)
            (folder / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 40)
            (folder / "images" / "c.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x03" * 40)
            (folder / "images" / "z.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x99" * 40)
            self._write_manifest(folder, "order-session", ["c.png", "a.png"])
            client = _client_for(root / "storage", data_root=root)
            resp = client.post(
                "/api/v1/sessions/path",
                json={"path": str(folder), "session_id": "order-session"},
            )
            meta = read_session_metadata(root / "storage", "order-session")
        assert resp.status_code == 200
        assert [r["filename"] for r in meta["images"]] == ["c.png", "a.png"]
        assert meta["rejected_files"] == []
        all_filenames = {r["filename"] for r in meta["images"]} | set(meta["rejected_files"])
        assert "z.png" not in all_filenames


class TestPushSession:
    def _create(self, client: TestClient, **form) -> dict:
        data = {"latitude": "48.5", "longitude": "21.3", "time_start": "2026-01-01T00:00:00Z",
                "time_end": "2026-01-01T01:00:00Z"}
        data.update(form)
        data = {k: v for k, v in data.items() if v is not None}
        resp = client.post(
            "/api/v1/sessions",
            data=data,
            files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_push_missing_time_returns_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            body = self._create(client, latitude=None, longitude=None, time_start=None, time_end=None)
            resp = client.post(f"/api/v1/sessions/{body['session_id']}/push")
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "time_start" in detail

    def test_push_disabled_returns_pushed_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            body = self._create(client)
            resp = client.post(f"/api/v1/sessions/{body['session_id']}/push")
        assert resp.status_code == 200
        assert resp.json()["pushed"] is False

    def test_push_nonexistent_session_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.post("/api/v1/sessions/does-not-exist/push")
        assert resp.status_code == 404

    def test_push_override_body_is_a_session_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            body = self._create(client)
            override = {
                "risk_input": {
                    "session_id": body["session_id"],
                    "infestation_score": 0.5,
                    "crop": "brassica-oleracea",
                    "growth_stage": {"representation": "enum", "unknown": True},
                    "time_start": "2026-01-01T00:00:00Z",
                    "time_end": "2026-01-01T01:00:00Z",
                    "latitude": 1.0,
                    "longitude": 2.0,
                },
                "reporting_envelope": {
                    "session_id": body["session_id"],
                    "best_image_id": body["best_image_id"],
                    "image_count": 1,
                    "time_start": "2026-01-01T00:00:00Z",
                    "time_end": "2026-01-01T01:00:00Z",
                    "latitude": 1.0,
                    "longitude": 2.0,
                    "provenance": {
                        "detector_impl_id": "lookup-detector",
                        "detector_version": "1.0.0",
                        "designation_status": "stub",
                        "aggregation_rule_version": "topfrac-v1",
                        "target_class_set": ["brassica_seedling"],
                        "confidence_floor": 0.3,
                    },
                },
            }
            resp = client.post(f"/api/v1/sessions/{body['session_id']}/push", json=override)
        assert resp.status_code == 200
        assert resp.json()["pushed"] is False

    @patch("app.main._post_json")
    @patch("app.main._get_gatekeeper_token", return_value="fake-gk-jwt")
    def test_push_gk_enabled_sends_contract_shaped_payload(self, _mock_tok, mock_post):
        mock_post.return_value = (200, '{"status":"accepted","report_id":"r-123"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                INFERENCE_USING_GATEKEEPER="true",
                INFERENCE_ORCH_PUSH_ENABLED="true",
                INFERENCE_STORAGE_DIR=str(tmp_path),
                INFERENCE_GATEKEEPER_BASE_URL="http://gatekeeper:8000",
            )
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            body = self._create(client)
            resp = client.post(f"/api/v1/sessions/{body['session_id']}/push")

        assert resp.status_code == 200
        data = resp.json()
        assert data["pushed"] is True
        assert "/api/proxy/orchestrator/" in data["target_url"]

        payload = mock_post.call_args[0][1]
        assert set(payload.keys()) == {"contract_version", "risk_input", "reporting_envelope"}
        assert payload["risk_input"]["latitude"] == 48.5
        assert "best_image_id" not in payload["risk_input"]
        assert payload["reporting_envelope"]["best_image_id"] == body["best_image_id"]

    @patch("app.main._post_json")
    def test_push_gk_disabled_routes_direct_to_orchestrator(self, mock_post):
        mock_post.return_value = (200, '{"status":"accepted"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                INFERENCE_USING_GATEKEEPER="false",
                INFERENCE_ORCH_PUSH_ENABLED="true",
                INFERENCE_ORCH_EDGE_SHARED_SECRET="test-edge-shared-secret",
                INFERENCE_STORAGE_DIR=str(tmp_path),
            )
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            body = self._create(client)
            with patch.dict(os.environ, {"INFERENCE_ORCH_BASE_URL": "http://orchestrator:8020"}, clear=False):
                resp = client.post(f"/api/v1/sessions/{body['session_id']}/push")

        assert resp.status_code == 200
        assert resp.json()["pushed"] is True
        assert "orchestrator:8020" in resp.json()["target_url"]

    @patch("app.main._post_json")
    def test_push_omits_absent_coordinates_from_contract_payload(self, mock_post):
        mock_post.return_value = (200, '{"status":"accepted"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                INFERENCE_USING_GATEKEEPER="false",
                INFERENCE_ORCH_PUSH_ENABLED="true",
                INFERENCE_ORCH_EDGE_SHARED_SECRET="test-edge-shared-secret",
                INFERENCE_STORAGE_DIR=str(tmp_path),
            )
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            body = self._create(client, latitude=None, longitude=None)
            with patch.dict(os.environ, {"INFERENCE_ORCH_BASE_URL": "http://orchestrator:8020"}, clear=False):
                pushed = client.post(f"/api/v1/sessions/{body['session_id']}/push")

        assert pushed.status_code == 200
        payload = mock_post.call_args[0][1]
        omitted_paths = {
            "risk_input.growth_stage.bbch_min",
            "risk_input.growth_stage.bbch_max",
            "risk_input.growth_stage.stage_id",
            "risk_input.latitude",
            "risk_input.longitude",
            "risk_input.position_accuracy_m",
            "reporting_envelope.latitude",
            "reporting_envelope.longitude",
            "reporting_envelope.position_accuracy_m",
            "reporting_envelope.parcel_id",
            "reporting_envelope.provenance.model_version",
            "reporting_envelope.provenance.model_source",
            "reporting_envelope.provenance.model_license",
            "reporting_envelope.provenance.conversion_chain",
        }
        for path in omitted_paths:
            value = payload
            *parents, leaf = path.split(".")
            for parent in parents:
                value = value[parent]
            assert leaf not in value, path

        assert set(payload) == {"contract_version", "risk_input", "reporting_envelope"}
        assert set(payload["risk_input"]) == {
            "session_id", "infestation_score", "crop", "growth_stage", "time_start", "time_end"
        }
        assert payload["risk_input"]["growth_stage"] == {"representation": "enum", "unknown": True}
        assert set(payload["reporting_envelope"]) == {
            "session_id", "best_image_id", "image_count", "time_start", "time_end", "provenance"
        }
        assert set(payload["reporting_envelope"]["provenance"]) == {
            "detector_impl_id", "detector_version", "designation_status", "aggregation_rule_version",
            "target_class_set", "confidence_floor",
        }

    @patch("app.main._post_json")
    def test_push_uses_centroid_coordinates_when_positions_are_available(self, mock_post):
        mock_post.return_value = (200, '{"status":"accepted"}')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                INFERENCE_USING_GATEKEEPER="false",
                INFERENCE_ORCH_PUSH_ENABLED="true",
                INFERENCE_ORCH_EDGE_SHARED_SECRET="test-edge-shared-secret",
                INFERENCE_STORAGE_DIR=str(tmp_path),
            )
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            metadata = json.dumps(
                {
                    "per_image_positions": {
                        "a.png": {"latitude": 8.0, "longitude": 20.0, "provenance": "replay"},
                        "b.png": {"latitude": 10.0, "longitude": 21.0, "provenance": "replay"},
                    }
                }
            )
            created = client.post(
                "/api/v1/sessions",
                data={
                    "latitude": "1.0",
                    "longitude": "2.0",
                    "time_start": "2026-01-01T00:00:00Z",
                    "time_end": "2026-01-01T01:00:00Z",
                    "metadata": metadata,
                },
                files=[
                    ("files", ("a.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"a" * 40), "image/png")),
                    ("files", ("b.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"b" * 40), "image/png")),
                ],
            ).json()
            with patch.dict(os.environ, {"INFERENCE_ORCH_BASE_URL": "http://orchestrator:8020"}, clear=False):
                pushed = client.post(f"/api/v1/sessions/{created['session_id']}/push")

        assert pushed.status_code == 200
        payload = mock_post.call_args[0][1]
        assert payload["risk_input"]["latitude"] == 9.0
        assert payload["risk_input"]["longitude"] == 20.5


@_NEEDS_SAMPLE_IMAGES
class TestBestImagePushDeferredDelivery:
    _FIXTURE_IMAGE = "sample-02.png"

    def _create(self, client: TestClient) -> dict:
        resp = client.post(
            "/api/v1/sessions",
            data={
                "latitude": "10.0",
                "longitude": "20.0",
                "time_start": "2026-01-01T00:00:00Z",
                "time_end": "2026-01-01T01:00:00Z",
            },
            files={
                "files": (
                    self._FIXTURE_IMAGE,
                    (_SAMPLE_IMAGES_DIR / self._FIXTURE_IMAGE).read_bytes(),
                    "image/png",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    @patch("app.main._push_best_image_pair_once")
    def test_exhausted_budget_self_heals_via_background_continuation(self, mock_push):
        observed: dict = {}

        def side_effect(settings, session_id, image_id, clean_bytes, clean_ct, annotated_bytes, annotated_ct,
                         artifact_hash):
            observed["calls"] = observed.get("calls", 0) + 1
            if observed["calls"] <= 4:
                return False, f"forced failure {observed['calls']}"
            meta = read_session_metadata(settings.storage_dir, session_id)
            observed["pending_before_continuation_success"] = meta["images_pending"]
            observed["attempts_before_continuation_success"] = meta["best_image_push_attempts"]
            return True, '{"status":"accepted"}'

        mock_push.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                INFERENCE_USING_GATEKEEPER="false",
                INFERENCE_STORAGE_DIR=str(tmp_path),
                INFERENCE_BEST_IMAGE_PUSH_ENABLED="true",
                BEST_IMAGE_PUSH_MAX_RETRIES=3,
                BEST_IMAGE_PUSH_BACKOFF_BASE_SEC=0.01,
                BEST_IMAGE_PUSH_DEADLINE_SEC=10,
                BEST_IMAGE_PUSH_CONTINUATION_INTERVAL_SEC=0.01,
            )
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            body = self._create(client)

            resp = client.post(f"/api/v1/sessions/{body['session_id']}/push")
            assert resp.status_code == 200

            final_meta = read_session_metadata(tmp_path, body["session_id"])

        assert mock_push.call_count == 5
        assert observed["pending_before_continuation_success"] is True
        assert observed["attempts_before_continuation_success"] == 4
        assert final_meta["images_pending"] is False
        assert final_meta["best_image_push_attempts"] == 5


_RS_STORED_NOT_EMBEDDED = '{"stored":true,"deduplicated":false,"report_updated":false,"images_embedded":false}'
_RS_STORED_EMBEDDED = '{"stored":true,"deduplicated":false,"report_updated":true,"images_embedded":true}'
_RS_STORED_OLDER_RS = '{"stored":true,"deduplicated":false,"report_updated":true}'


@_NEEDS_SAMPLE_IMAGES
class TestBestImagePushConfirmsEmbedding:
    _FIXTURE_IMAGE = TestBestImagePushDeferredDelivery._FIXTURE_IMAGE
    _create = TestBestImagePushDeferredDelivery._create

    @staticmethod
    def _once(settings, body):
        with patch("app.main._post_json", return_value=(200, body)), patch(
            "app.main._get_gatekeeper_token", return_value="fake-gk-jwt"
        ):
            return main_module._push_best_image_pair_once(
                settings, "sess-x", "img-1", b"c", "image/jpeg", b"a", "image/jpeg", "hash-1"
            )

    def _settings(self, tmp_path, **extra):
        return Settings(
            INFERENCE_USING_GATEKEEPER="true",
            INFERENCE_STORAGE_DIR=str(tmp_path),
            INFERENCE_GATEKEEPER_BASE_URL="http://gatekeeper:8000",
            INFERENCE_BEST_IMAGE_PUSH_ENABLED="true",
            BEST_IMAGE_PUSH_MAX_RETRIES=3,
            BEST_IMAGE_PUSH_BACKOFF_BASE_SEC=0.01,
            BEST_IMAGE_PUSH_DEADLINE_SEC=10,
            BEST_IMAGE_PUSH_CONTINUATION_INTERVAL_SEC=0.01,
            **extra,
        )

    def test_stored_with_embedded_false_is_not_a_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            success, detail = self._once(self._settings(Path(tmp)), _RS_STORED_NOT_EMBEDDED)
        assert success is False
        assert "images_embedded" in detail

    def test_stored_with_embedded_true_is_a_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            success, detail = self._once(self._settings(Path(tmp)), _RS_STORED_EMBEDDED)
        assert success is True
        assert json.loads(detail)["images_embedded"] is True

    def test_an_rs_without_the_field_is_never_taken_as_embedded(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            for body in (_RS_STORED_OLDER_RS, '{"images_embedded":"true"}', "not json", "[]", ""):
                success, _ = self._once(settings, body)
                assert success is False, body

    def test_an_http_error_stays_a_failure_with_its_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("app.main._post_json", return_value=(503, "busy")), patch(
                "app.main._get_gatekeeper_token", return_value="fake-gk-jwt"
            ):
                success, detail = main_module._push_best_image_pair_once(
                    self._settings(Path(tmp)), "sess-x", "img-1", b"c", "image/jpeg", b"a",
                    "image/jpeg", "hash-1",
                )
        assert success is False
        assert "HTTP 503" in detail

    def test_state_stays_pending_while_rs_reports_not_embedded_and_clears_when_it_does(self):
        answers = iter([_RS_STORED_NOT_EMBEDDED, _RS_STORED_NOT_EMBEDDED, _RS_STORED_EMBEDDED])
        seen = []

        def _rs(url, payload, headers, timeout):
            meta = read_session_metadata(tmp_path, session_id)
            seen.append((meta["images_pending"], meta["best_image_push_attempts"]))
            return 200, next(answers)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = self._settings(tmp_path)
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            session_id = self._create(client)["session_id"]

            with patch("app.main._post_json", side_effect=_rs), patch(
                "app.main._get_gatekeeper_token", return_value="fake-gk-jwt"
            ):
                main_module._run_best_image_push_sequence(settings, session_id)

            final = read_session_metadata(tmp_path, session_id)

        assert seen == [(True, 0), (True, 1), (True, 2)]
        assert final["images_pending"] is False
        assert final["best_image_push_attempts"] == 3

    def test_the_retry_budget_is_unchanged_and_the_continuation_still_lands_it(self):
        answers = iter([_RS_STORED_NOT_EMBEDDED] * 4 + [_RS_STORED_EMBEDDED])
        calls = []

        def _rs(url, payload, headers, timeout):
            calls.append(url)
            return 200, next(answers)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = self._settings(tmp_path)
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth
            client = TestClient(app, raise_server_exceptions=False)
            session_id = self._create(client)["session_id"]

            with patch("app.main._post_json", side_effect=_rs), patch(
                "app.main._get_gatekeeper_token", return_value="fake-gk-jwt"
            ):
                main_module._run_best_image_push_sequence(settings, session_id)
            final = read_session_metadata(tmp_path, session_id)

        assert len(calls) == 5
        assert final["images_pending"] is False
        assert final["best_image_push_attempts"] == 5


class TestImageServingAndAuth:
    def test_image_bytes_served_after_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
            image_id = resp.json()["best_image_id"]
            img_resp = client.get(f"/image/{image_id}/bytes")
        assert img_resp.status_code == 200
        assert len(img_resp.content) > 0

    def test_image_bytes_nonexistent_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp))
            resp = client.get("/image/does-not-exist/bytes")
        assert resp.status_code == 404

    def test_clean_slash_form_serves_identical_bytes_with_no_redirect(self):
        """The Gatekeeper's proxy
        forwards every fetch with a trailing slash; the route must answer
        that form directly (200, same bytes), never a 307."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
            image_id = resp.json()["best_image_id"]

            bare = client.get(f"/image/{image_id}/clean", follow_redirects=False)
            slash = client.get(f"/image/{image_id}/clean/", follow_redirects=False)

            assert bare.status_code == 200
            assert slash.status_code == 200
            assert slash.content == bare.content


@pytest.mark.skipif(
    not _REAL_DETECTION_RESULTS.exists() or not _SAMPLE_IMAGES_DIR.exists(),
    reason="requires the shipped sample-image / detection-results fixtures",
)
class TestAnnotatedImageSlashForm:
    """The annotated variant is
    rendered from persisted detections (a real image, unlike the clean
    variant, which serves stored bytes verbatim) -- needs the shipped
    fixtures to reach a 200 at all."""

    def test_annotated_slash_form_serves_identical_bytes_with_no_redirect(self):
        hot_name = "sample-03.png"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = _client_for(tmp_path)
            resp = client.post(
                "/api/v1/sessions",
                files=[("files", (hot_name, (_SAMPLE_IMAGES_DIR / hot_name).read_bytes(), "image/png"))],
            )
            image_id = resp.json()["best_image_id"]

            bare = client.get(f"/image/{image_id}/annotated", follow_redirects=False)
            slash = client.get(f"/image/{image_id}/annotated/", follow_redirects=False)

            assert bare.status_code == 200
            assert slash.status_code == 200
            assert slash.content == bare.content

    def test_api_key_valid_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), INFERENCE_EXTERNAL_API_KEY="test-key")
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/sessions",
                headers={"X-API-Key": "test-key"},
                files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
        assert resp.status_code == 200

    def test_api_key_invalid_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), INFERENCE_EXTERNAL_API_KEY="correct-key")
            app.dependency_overrides[get_settings] = lambda: settings
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/sessions",
                headers={"X-API-Key": "wrong-key"},
                files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
        assert resp.status_code == 403


class TestDetectDebugEndpoint:
    """
    ``POST /internal/detect-debug`` is the only surface that returns raw
    canonical detections (label/box/score) rather than the aggregated
    per-image score the session API exposes — useful for checking the box/label/score a deployed pipeline returns.
    """

    def test_empty_lookup_result_returns_empty_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp), INFERENCE_DETECTOR_IMPL="lookup")
            resp = client.post(
                "/internal/detect-debug",
                files={"file": ("no-such-fixture.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["detector_impl_id"] == "lookup-detector"
        assert body["detections"] == []

    def test_populated_lookup_result_returns_raw_boxes(self):
        entries = json.loads(_REAL_DETECTION_RESULTS.read_text())
        filename = next(iter(entries))
        expected_box = entries[filename]["boxes"][0]

        with tempfile.TemporaryDirectory() as tmp:
            client = _client_for(Path(tmp), INFERENCE_DETECTOR_IMPL="lookup")
            resp = client.post(
                "/internal/detect-debug",
                files={"file": (filename, BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["detections"]) >= 1
        got = body["detections"][0]
        assert got["label"] == expected_box["label"]
        assert got["score"] == expected_box["score"]
        assert got["x_min"] == expected_box["x_min"]
