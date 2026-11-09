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
The ingest path
stays responsive during slow uploads and finalization.

Two blocking operations are offloaded off the event loop: the
detector run inside ``_finalize_session`` and the disk write inside
``save_upload_file``. Pre-change, both run synchronously on the event
loop itself, so a 2.0s injected delay in either one blocks the loop for
2.0s -- provably exceeding the 1.0s ``RESPONSIVENESS_BOUND_SECONDS``
bound below for both ``/healthz`` and the probe sink, since a blocked
loop cannot even dispatch to the threadpool. Post-change, each delayed
operation runs off the loop, so the bound holds.
"""
from __future__ import annotations

import asyncio
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
from app.detectors.base import ImageDetections, ImageRef  # noqa: E402
from app.main import app, gatekeeper_auth  # noqa: E402

RESPONSIVENESS_BOUND_SECONDS = 1.0


def _no_auth():
    return None


def _make_settings(storage_dir: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(storage_dir),
        INFERENCE_DATA_ROOT=str(storage_dir),
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


class _SlowDetector:
    """A deliberately slow detector double -- the first of the two
    blocking operations that are offloaded."""

    impl_id = "slow-test-double"
    version = "0"

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def detect(self, image_ref: ImageRef) -> ImageDetections:
        time.sleep(self._delay)
        return ImageDetections(image_id=image_ref.image_id, detections=[])


def _post_one_image_session(client: TestClient, session_id: str, results: dict) -> None:
    resp = client.post(
        "/api/v1/sessions",
        data={
            "session_id": session_id,
            "latitude": "48.5",
            "longitude": "21.3",
            "time_start": "2026-01-01T00:00:00Z",
            "time_end": "2026-01-01T01:00:00Z",
        },
        files={"files": ("img.png", BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), "image/png")},
    )
    results["status_code"] = resp.status_code
    results["body"] = resp.text


def _assert_health_and_sink_answer_promptly(client: TestClient) -> None:
    start = time.monotonic()
    health_resp = client.get("/healthz")
    health_elapsed = time.monotonic() - start
    assert health_resp.status_code == 200
    assert health_elapsed < RESPONSIVENESS_BOUND_SECONDS, (
        f"/healthz took {health_elapsed:.2f}s -- must answer within a single-digit-second bound "
        "independent of finalization/upload progress"
    )

    start = time.monotonic()
    sink_resp = client.post("/api/v1/probe/sink", content=b"probe-bytes")
    sink_elapsed = time.monotonic() - start
    assert sink_resp.status_code == 200
    assert sink_resp.json()["received_bytes"] == len(b"probe-bytes")
    assert sink_elapsed < RESPONSIVENESS_BOUND_SECONDS, (
        f"the probe sink took {sink_elapsed:.2f}s -- must answer within a single-digit-second bound"
    )


class TestHealthAndSinkAnswerDuringFinalization:
    def test_slow_detector_does_not_starve_health_or_sink(self, monkeypatch):
        """Injection (i): a deliberately slow detector double (2.0s delay,
        see ``_SlowDetector``) finalizing one session. Pre-change,
        ``detect`` runs synchronously inside ``_finalize_session`` on the
        event loop, so the 2.0s delay blocks the loop -- provably exceeding
        the 1.0s ``RESPONSIVENESS_BOUND_SECONDS`` bound for both
        ``/healthz`` and the probe sink. Post-change, ``detect`` runs off
        the loop."""
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            monkeypatch.setattr(main_module, "_get_detector", lambda impl_name: _SlowDetector(delay=2.0))

            with _client_for(storage_dir) as client:
                results: dict = {}
                t = threading.Thread(target=_post_one_image_session, args=(client, "sess-slow-detect", results))
                t.start()
                time.sleep(0.3)

                _assert_health_and_sink_answer_promptly(client)

                t.join(timeout=10)
                assert not t.is_alive(), "the slow upload must still complete"
                assert results["status_code"] == 200, results.get("body")

    def test_slow_storage_write_does_not_starve_health_or_sink(self, monkeypatch):
        """
        Injection (ii): a deliberately slow storage-write double (2.0s
        delay in ``_slow_write_bytes``) persisting one session's files.
        Offloading only the detector would pass a detector-only test while
        a slow disk still starves both endpoints. Pre-change,
        ``save_upload_file``'s ``Path.write_bytes`` call runs synchronously
        on the event loop, so the 2.0s delay blocks the loop -- provably
        exceeding the 1.0s ``RESPONSIVENESS_BOUND_SECONDS`` bound for both
        ``/healthz`` and the probe sink. Post-change, the write runs off
        the loop.
        """
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)

            real_write_bytes = Path.write_bytes

            def _slow_write_bytes(self: Path, data: bytes) -> int:
                time.sleep(2.0)
                return real_write_bytes(self, data)

            monkeypatch.setattr(Path, "write_bytes", _slow_write_bytes)

            with _client_for(storage_dir) as client:
                results: dict = {}
                t = threading.Thread(target=_post_one_image_session, args=(client, "sess-slow-write", results))
                t.start()
                time.sleep(0.3)

                _assert_health_and_sink_answer_promptly(client)

                t.join(timeout=10)
                assert not t.is_alive(), "the slow upload must still complete"
                assert results["status_code"] == 200, results.get("body")


class TestProbeSinkAnswersDuringCrawlingUpload:
    def test_paced_request_stream_still_serves_health_and_sink_mid_transfer(self):
        """While a paced request stream feeds the session endpoint,
        both a probe-sink call and a health call answer within bound. Pinned
        at the framework's own streaming layer (the ASGI app awaits each
        chunk from ``receive``) -- this is what makes a slow, in-flight
        UPLOAD distinct from a slow FINALIZATION.

        Driven with a real ``httpx.AsyncClient`` over ``httpx.ASGITransport``
        and an async-generator request body (genuine ``await asyncio.sleep``
        between chunks), NOT Starlette's synchronous ``TestClient``: that
        client's own bridging code (``testclient.py``'s ``receive``) calls
        ``httpx.Request.read``, which EAGERLY materializes any request
        body -- including a paced generator -- into one `bytes` blob in a
        single synchronous call before the ASGI app ever starts reading it,
        which would silently defeat the pacing entirely and block the
        event loop for the generator's full duration in one shot.
        """
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            settings = _make_settings(storage_dir)
            app.dependency_overrides[get_settings] = lambda: settings
            app.dependency_overrides[gatekeeper_auth] = _no_auth

            boundary = "test-boundary-paced-upload"
            image_bytes = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 20_000)
            head = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="session_id"\r\n\r\n'
                "sess-crawling-upload\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="latitude"\r\n\r\n'
                "48.5\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="longitude"\r\n\r\n'
                "21.3\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="time_start"\r\n\r\n'
                "2026-01-01T00:00:00Z\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="time_end"\r\n\r\n'
                "2026-01-01T01:00:00Z\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="files"; filename="img.png"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode()
            tail = f"\r\n--{boundary}--\r\n".encode()

            async def paced_body():
                yield head
                chunk_size = 2000
                for i in range(0, len(image_bytes), chunk_size):
                    await asyncio.sleep(0.15)
                    yield image_bytes[i : i + chunk_size]
                await asyncio.sleep(0.15)
                yield tail

            async def scenario() -> dict:
                results: dict = {}
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    async def do_paced_upload():
                        resp = await client.post(
                            "/api/v1/sessions",
                            content=paced_body(),
                            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                        )
                        results["status_code"] = resp.status_code
                        results["body"] = resp.text

                    upload_task = asyncio.create_task(do_paced_upload())
                    await asyncio.sleep(0.3)

                    start = time.monotonic()
                    health_resp = await client.get("/healthz")
                    results["health_status"] = health_resp.status_code
                    results["health_elapsed"] = time.monotonic() - start

                    start = time.monotonic()
                    sink_resp = await client.post("/api/v1/probe/sink", content=b"probe-bytes")
                    results["sink_status"] = sink_resp.status_code
                    results["sink_received_bytes"] = sink_resp.json()["received_bytes"]
                    results["sink_elapsed"] = time.monotonic() - start

                    await asyncio.wait_for(upload_task, timeout=15)
                return results

            try:
                results = asyncio.run(scenario())
            finally:
                app.dependency_overrides.clear()

        assert results["health_status"] == 200
        assert results["health_elapsed"] < RESPONSIVENESS_BOUND_SECONDS, (
            f"/healthz took {results['health_elapsed']:.2f}s during a crawling upload"
        )
        assert results["sink_status"] == 200
        assert results["sink_received_bytes"] == len(b"probe-bytes")
        assert results["sink_elapsed"] < RESPONSIVENESS_BOUND_SECONDS, (
            f"the probe sink took {results['sink_elapsed']:.2f}s during a crawling upload"
        )
        assert results["status_code"] == 200, results.get("body")
