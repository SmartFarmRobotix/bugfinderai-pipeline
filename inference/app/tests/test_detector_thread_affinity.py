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

"""Every detector invocation rides one
dedicated single-worker executor, and a runtime-affine detector refuses
cross-thread calls by name.

Background: the TensorRT detector's lazy
``import pycuda.autoinit`` binds the CUDA context to the importing
thread. Detector calls reach it from three differently-threaded routes
(upload finalization via the default executor pool, PATH finalization on
the server's own threadpool, detect-debug on the event loop), so
whichever runs first strands every other route at the driver
(``invalid device context - no currently active context``). The fix
pins all three routes to one executor whose workers carry the
``detector`` thread-name prefix — the prefix is what these tests
assert, because bare id-equality cannot distinguish the dedicated
executor from any other single-worker pool.
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

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("INFERENCE_USING_GATEKEEPER", "false")
os.environ.setdefault("INFERENCE_ORCH_PUSH_ENABLED", "false")
os.environ.setdefault("INFERENCE_STORAGE_DIR", "/tmp/inf_test_storage")

from app import main as main_module  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.detectors import dfine_tensorrt as trt_module  # noqa: E402
from app.detectors.base import ImageDetections, ImageRef  # noqa: E402
from app.detectors.dfine_tensorrt import DFineTensorRTDetector  # noqa: E402
from app.main import app, gatekeeper_auth  # noqa: E402

RESPONSIVENESS_BOUND_SECONDS = 1.0
DEDICATED_PREFIX = "detector"


def _no_auth():
    return None


def _make_settings(storage_dir: Path, *, data_root: Path | None = None, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(storage_dir),
        INFERENCE_DATA_ROOT=str(data_root or storage_dir),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _client_for(storage_dir: Path, **overrides: Any) -> TestClient:
    settings = _make_settings(storage_dir, **overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = _no_auth
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()
    main_module._DETECTOR_CACHE.clear()


class _ThreadRecordingDetector:
    """Records the executing thread's name and ident on every detect."""

    impl_id = "thread-recorder-test-double"
    version = "0"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        t = threading.current_thread()
        self.calls.append((t.name, t.ident or -1))
        return ImageDetections(image_id=image_ref.image_id, detections=[])


def _write_manifest(folder: Path, session_id: str, filenames: list[str]) -> None:
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


def _make_session_folder(root: Path, name: str) -> Path:
    folder = root / name
    (folder / "images").mkdir(parents=True)
    (folder / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 40)
    _write_manifest(folder, name, ["a.png"])
    return folder


def _post_upload(client: TestClient, session_id: str) -> None:
    resp = client.post(
        "/api/v1/sessions",
        data={"session_id": session_id},
        files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
    )
    assert resp.status_code == 200, resp.text


def _post_path(client: TestClient, folder: Path) -> None:
    resp = client.post("/api/v1/sessions/path", json={"path": str(folder)})
    assert resp.status_code == 200, resp.text


def _post_detect_debug(client: TestClient) -> None:
    resp = client.post(
        "/internal/detect-debug",
        files={"file": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
    )
    assert resp.status_code == 200, resp.text


class TestDedicatedDetectorThread:
    def test_all_three_routes_ride_the_dedicated_thread(self, monkeypatch):
        """Upload finalization, PATH finalization, and detect-debug each
        invoke the detector twice; every recorded thread name carries the
        dedicated executor's prefix and every ident is the same single
        thread. Pre-change this fails on all three counts: the upload
        lands on the default executor pool, PATH on the server's own
        threadpool, detect-debug on the event-loop thread."""
        recorder = _ThreadRecordingDetector()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monkeypatch.setattr(main_module, "_get_detector", lambda impl_name: recorder)
            with _client_for(root / "storage", data_root=root) as client:
                _post_upload(client, "affinity-upload-1")
                _post_upload(client, "affinity-upload-2")
                _post_path(client, _make_session_folder(root, "affinity-path-1"))
                _post_path(client, _make_session_folder(root, "affinity-path-2"))
                _post_detect_debug(client)
                _post_detect_debug(client)

        assert len(recorder.calls) >= 6, recorder.calls
        names = {name for name, _ in recorder.calls}
        idents = {ident for _, ident in recorder.calls}
        offenders = [n for n in names if not n.startswith(DEDICATED_PREFIX)]
        assert not offenders, (
            f"detector ran on non-dedicated thread(s) {offenders} — every invocation must ride "
            f"the single '{DEDICATED_PREFIX}'-prefixed executor thread (calls: {recorder.calls})"
        )
        assert len(idents) == 1, (
            f"detector ran on {len(idents)} distinct threads {idents} — the executor must be "
            f"single-worker so the CUDA context stays valid (calls: {recorder.calls})"
        )


class _SlowDetector:
    impl_id = "slow-test-double"
    version = "0"

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        time.sleep(self._delay)
        return ImageDetections(image_id=image_ref.image_id, detections=[])


class TestDetectDebugResponsiveness:
    def test_healthz_answers_during_slow_detect_debug(self, monkeypatch):
        """Pre-change, detect_debug_endpoint runs detector.detect
        synchronously inside the async handler, so a 2.0s detect blocks
        the event loop and /healthz starves past the 1.0s bound."""
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            monkeypatch.setattr(main_module, "_get_detector", lambda impl_name: _SlowDetector(delay=2.0))
            with _client_for(storage_dir) as client:
                results: dict = {}

                def _drive() -> None:
                    resp = client.post(
                        "/internal/detect-debug",
                        files={"file": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
                    )
                    results["status_code"] = resp.status_code

                t = threading.Thread(target=_drive)
                t.start()
                time.sleep(0.3)

                start = time.monotonic()
                health = client.get("/healthz")
                elapsed = time.monotonic() - start
                assert health.status_code == 200
                assert elapsed < RESPONSIVENESS_BOUND_SECONDS, (
                    f"/healthz took {elapsed:.2f}s while detect-debug was in flight — the debug "
                    "surface must not block the event loop"
                )

                t.join(timeout=10)
                assert not t.is_alive()
                assert results["status_code"] == 200


class _Marker(Exception):
    """Raised by the patched preprocess to prove the guard was traversed."""


class TestAffinityGuard:
    """The typed cross-thread refusal, unit-tested without CUDA by
    building the detector via __new__ and injecting a fake runtime —
    the guard must act before any runtime use."""

    def _bare_detector(self) -> DFineTensorRTDetector:
        det = object.__new__(DFineTensorRTDetector)
        det._runtime = ("trt", "cuda", "engine", "context")
        det._runtime_owner_ident = None
        det._imgsz = 640
        det._tiling = None
        det._class_names = ["x"]
        return det

    def test_same_thread_call_proceeds_past_the_guard(self, monkeypatch):
        det = self._bare_detector()
        det._runtime_owner_ident = threading.get_ident()
        monkeypatch.setattr(trt_module, "preprocess_image_dfine", lambda *a, **k: (_ for _ in ()).throw(_Marker()))
        with pytest.raises(_Marker):
            det.detect(ImageRef(image_id="i", filename="i.png", path=Path("/nonexistent")))

    def test_cross_thread_call_is_refused_naming_both_ids(self):
        det = self._bare_detector()
        owner = threading.get_ident() + 12345
        det._runtime_owner_ident = owner
        with pytest.raises(trt_module.TensorRTThreadAffinityError) as exc:
            det.detect(ImageRef(image_id="i", filename="i.png", path=Path("/nonexistent")))
        msg = str(exc.value)
        assert str(owner) in msg and str(threading.get_ident()) in msg, msg

    def test_unset_owner_is_refused_by_name(self):
        det = self._bare_detector()
        det._runtime_owner_ident = None
        with pytest.raises(trt_module.TensorRTThreadAffinityError) as exc:
            det.detect(ImageRef(image_id="i", filename="i.png", path=Path("/nonexistent")))
        assert "no owning thread" in str(exc.value), str(exc.value)
