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
Real-session best-image pair resolution: the pre-generated fixture store keeps precedence, and when
it has no pair for the image, the REAL-SESSION path resolves — clean is
the stored original, annotated is rendered from the metadata's
persisted best_image_detections. A session without persisted
detections fails closed (404), never a fabricated annotated image.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

os.environ.setdefault("INFERENCE_USING_GATEKEEPER", "false")
os.environ.setdefault("INFERENCE_ORCH_PUSH_ENABLED", "false")
os.environ.setdefault("INFERENCE_STORAGE_DIR", "/tmp/inf_test_storage")

from app import main as main_module  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import _render_annotated_from_detections, _resolve_pair_raw  # noqa: E402
from app.storage import ensure_dir, write_session_metadata  # noqa: E402

DETS = [
    {"label": "aphid", "score": 0.91, "x_min": 0.0625, "y_min": 0.1, "x_max": 0.375, "y_max": 0.4583333},
    {"label": "aphid", "score": 0.44, "x_min": 0.4375, "y_min": 0.1666667, "x_max": 0.6875, "y_max": 0.5833333},
]


def _is_overlay_colour(rgb) -> bool:
    """The overlay colour #eb6834 (235, 104, 52) family. Thin strokes lose
    chroma to JPEG subsampling, so this asserts the warm-orange family, not
    the exact value (the exact value is asserted on a thick stroke below)."""
    r, g, b = rgb
    return r >= 170 and 70 <= g <= 160 and r - b >= 60


def _real_session(tmp: Path, session_id: str, *, detections):
    """A stored session with one real JPEG and metadata shaped like
    _finalize_session's payload (images records + best_image_id)."""
    sdir = tmp / "sessions" / session_id
    ensure_dir(sdir)
    img_path = sdir / "frame-0001.jpg"
    Image.new("RGB", (160, 120), (40, 120, 40)).save(img_path, format="JPEG")
    payload = {
        "session_id": session_id,
        "best_image_id": "img-1",
        "images": [{"image_id": "img-1", "filename": "frame-0001.jpg", "path": str(img_path)}],
    }
    if detections is not None:
        payload["best_image_detections"] = detections
    write_session_metadata(tmp, session_id, payload)
    return payload


def _settings(tmp: Path) -> Settings:
    return Settings(
        INFERENCE_USING_GATEKEEPER="false",
        INFERENCE_ORCH_PUSH_ENABLED="false",
        INFERENCE_STORAGE_DIR=str(tmp),
        INFERENCE_DATA_ROOT=str(tmp),
    )


def test_real_session_pair_resolves_with_rendered_annotations() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-real", detections=DETS)
        clean_raw, annotated_raw = _resolve_pair_raw(_settings(tmp), "sess-real", "img-1", payload)

        clean_img = Image.open(io.BytesIO(clean_raw))
        annotated_img = Image.open(io.BytesIO(annotated_raw))
        assert clean_img.size == annotated_img.size == (160, 120)
        assert annotated_raw != clean_raw
        px = annotated_img.convert("RGB").getpixel((11, 40))
        assert _is_overlay_colour(px), px


def test_empty_detections_render_honestly_as_boxless() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-empty", detections=[])
        clean_raw, annotated_raw = _resolve_pair_raw(_settings(tmp), "sess-empty", "img-1", payload)
        assert Image.open(io.BytesIO(annotated_raw)).size == (160, 120)


def test_missing_persisted_detections_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-old", detections=None)
        with pytest.raises(HTTPException) as exc:
            _resolve_pair_raw(_settings(tmp), "sess-old", "img-1", payload)
        assert exc.value.status_code == 404
        assert "best_image_detections" in exc.value.detail


def test_fixture_store_keeps_precedence(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as fixtures:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-fixture", detections=DETS)
        fx = Path(fixtures)
        for prefix, color in (("clean", (1, 2, 3)), ("annotated", (200, 5, 5))):
            img = Image.new("RGB", (8, 8), color)
            img.save(fx / f"{prefix}_frame-0001.png", format="PNG")
        monkeypatch.setattr(main_module, "_ANNOTATED_IMAGES_DIR", fx)
        clean_raw, annotated_raw = _resolve_pair_raw(_settings(tmp), "sess-fixture", "img-1", payload)
        assert Image.open(io.BytesIO(clean_raw)).size == (8, 8), "fixture pair must win over the real-session render"


def test_render_is_pure_pil_and_marks_every_detection() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        img_path = tmp / "orig.jpg"
        Image.new("RGB", (400, 300), (30, 30, 30)).save(img_path, format="JPEG")
        out = _render_annotated_from_detections(img_path, DETS)
        rendered = Image.open(io.BytesIO(out)).convert("RGB")
        for det in DETS:
            x = round(det["x_min"] * 400) + 1
            y = round((det["y_min"] + det["y_max"]) / 2 * 300)
            assert _is_overlay_colour(rendered.getpixel((x, y))), (det["label"], x, y, rendered.getpixel((x, y)))


def test_fixture_pair_is_byte_for_byte_and_clean_equals_stored_original(monkeypatch) -> None:
    """Byte-level assertions, not shape checks."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as fixtures:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-bytes", detections=DETS)
        clean_raw, _ = _resolve_pair_raw(_settings(tmp), "sess-bytes", "img-1", payload)
        stored = Path(payload["images"][0]["path"]).read_bytes()
        assert clean_raw == stored
        fx = Path(fixtures)
        for prefix, color in (("clean", (1, 2, 3)), ("annotated", (200, 5, 5))):
            Image.new("RGB", (8, 8), color).save(fx / f"{prefix}_frame-0001.png", format="PNG")
        monkeypatch.setattr(main_module, "_ANNOTATED_IMAGES_DIR", fx)
        clean_raw2, annotated_raw2 = _resolve_pair_raw(_settings(tmp), "sess-bytes", "img-1", payload)
        assert clean_raw2 == (fx / "clean_frame-0001.png").read_bytes()
        assert annotated_raw2 == (fx / "annotated_frame-0001.png").read_bytes()


def test_malformed_persisted_detection_is_refused_by_name() -> None:
    """a malformed self-written record refuses,
    never a partial annotation."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bad = [{"label": "aphid", "score": 0.5, "x_min": 0.1}]
        payload = _real_session(tmp, "sess-bad", detections=bad)
        with pytest.raises(HTTPException) as exc:
            _resolve_pair_raw(_settings(tmp), "sess-bad", "img-1", payload)
        assert exc.value.status_code == 422
        assert "malformed" in exc.value.detail


def test_real_session_render_never_reenters_the_detector(monkeypatch) -> None:
    """The render path must not touch the
    detector registry — a re-entry would reintroduce the thread/device
    dependency at push time."""
    def _bomb(*a, **k):
        raise AssertionError("detector re-entered during pair rendering")
    monkeypatch.setattr(main_module, "_get_detector", _bomb)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-nodet", detections=DETS)
        clean_raw, annotated_raw = _resolve_pair_raw(_settings(tmp), "sess-nodet", "img-1", payload)
        assert annotated_raw and clean_raw


def _big_original(tmp: Path) -> Path:
    path = tmp / "big.jpg"
    Image.new("RGB", (4000, 3000), (40, 120, 40)).save(path, format="JPEG", quality=95)
    return path


def test_boxes_land_where_the_normalised_detections_are() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.873, "x_min": 0.5, "y_min": 0.5, "x_max": 0.75, "y_max": 0.75}
        out = _render_annotated_from_detections(original, [det])
        before = np.asarray(Image.open(original).convert("RGB")).astype(int)
        after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB")).astype(int)
        assert before.shape == after.shape == (3000, 4000, 3)

        changed = np.abs(before - after).max(axis=2) > 12
        ys, xs = np.nonzero(changed)
        assert len(xs), "nothing was drawn"
        margin = 16
        stroke = 10
        assert xs.min() >= 2000 - margin and xs.max() <= 3000 + margin
        assert ys.min() >= 1500 - margin and ys.max() <= 2250 + margin
        assert not changed[: 1500 - margin].any()
        interior = changed[1500 + stroke + margin : 2250 - stroke - margin, 2000 + stroke + margin : 3000 - stroke - margin]
        assert not interior.any()
        for x, y in ((2005, 1900), (2995, 1900), (2500, 1505), (2500, 2245)):
            assert _is_overlay_colour(tuple(after[y, x])), (x, y, tuple(after[y, x]))
        assert np.abs(before[:100, :100] - after[:100, :100]).max() <= 3


def test_stroke_is_one_exact_colour_and_proportional_to_the_longer_side() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.5, "x_min": 0.25, "y_min": 0.4, "x_max": 0.75, "y_max": 0.9}
        out = _render_annotated_from_detections(original, [det])
        after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB")).astype(int)
        for x in (1002, 1006):
            r, g, b = after[2100, x]
            assert abs(r - 235) <= 6 and abs(g - 104) <= 6 and abs(b - 52) <= 6, (x, r, g, b)
        assert tuple(after[2100, 1030]) != tuple(after[2100, 1002])
        assert abs(after[2100, 990][1] - 120) <= 6


def test_no_text_is_drawn_at_the_boxes(monkeypatch) -> None:
    from PIL import ImageDraw

    texts: list[str] = []
    real_text = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        texts.append(text)
        return real_text(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.8, "x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}
        _render_annotated_from_detections(original, [det])
    assert texts == []


def test_box_at_the_image_edge_is_drawn_inside_the_image() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.5, "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        out = _render_annotated_from_detections(original, [det])
        after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB")).astype(int)
        assert after.shape == (3000, 4000, 3)
        assert _is_overlay_colour(tuple(after[5, 2000]))
        assert _is_overlay_colour(tuple(after[1500, 3995]))


@pytest.mark.parametrize(
    "field,value",
    [("x_max", 4056), ("x_min", -0.1), ("y_max", 1.0001), ("y_min", 12.0)],
)
def test_out_of_range_coordinates_are_refused_by_field_name(field, value) -> None:
    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.5, "x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
        det[field] = value
        with pytest.raises(HTTPException) as exc:
            _render_annotated_from_detections(original, [det])
        assert exc.value.status_code == 422
        assert field in exc.value.detail
        assert "[0, 1]" in exc.value.detail


@pytest.mark.parametrize(
    "box,names",
    [
        ({"x_min": 0.5, "x_max": 0.5}, ("x_min", "x_max")),
        ({"y_min": 0.6, "y_max": 0.2}, ("y_min", "y_max")),
    ],
)
def test_degenerate_or_inverted_boxes_are_refused_by_field_names(box, names) -> None:
    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.5, "x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5, **box}
        with pytest.raises(HTTPException) as exc:
            _render_annotated_from_detections(original, [det])
        assert exc.value.status_code == 422
        assert all(n in exc.value.detail for n in names)


@pytest.mark.parametrize("bad", ["0.5", None, True, float("nan")])
def test_non_numeric_or_nan_coordinate_is_refused(bad) -> None:
    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        det = {"label": "aphid", "score": 0.5, "x_min": bad, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
        with pytest.raises(HTTPException) as exc:
            _render_annotated_from_detections(original, [det])
        assert exc.value.status_code == 422
        assert "malformed" in exc.value.detail or "x_min" in exc.value.detail


def test_one_bad_detection_produces_no_image_at_all() -> None:
    with tempfile.TemporaryDirectory() as d:
        original = _big_original(Path(d))
        good = {"label": "aphid", "score": 0.5, "x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}
        bad = {**good, "x_max": 4056}
        with pytest.raises(HTTPException):
            _render_annotated_from_detections(original, [good, bad])


from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app, gatekeeper_auth  # noqa: E402
from app.image_compress import compress_pair_to_cap  # noqa: E402


def _client(tmp: Path) -> TestClient:
    settings = _settings(tmp)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[gatekeeper_auth] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _fixture_dir(fixtures: str) -> Path:
    fx = Path(fixtures)
    for prefix, color in (("clean", (1, 2, 3)), ("annotated", (200, 5, 5))):
        Image.new("RGB", (8, 8), color).save(fx / f"{prefix}_frame-0001.png", format="PNG")
    return fx


def test_real_session_variants_are_served_by_both_routes() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-routes", detections=DETS)
        client = _client(tmp)

        clean = client.get("/image/img-1/clean")
        annotated = client.get("/image/img-1/annotated")

        assert clean.status_code == 200 and annotated.status_code == 200
        assert clean.headers["content-type"] == "image/jpeg"
        assert annotated.headers["content-type"] == "image/jpeg"
        assert clean.content == Path(payload["images"][0]["path"]).read_bytes()
        assert Image.open(io.BytesIO(annotated.content)).size == (160, 120)
        assert annotated.content != clean.content


def test_png_original_is_served_with_its_own_media_type() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-png", detections=DETS)
        png_path = Path(payload["images"][0]["path"]).with_suffix(".png")
        Image.new("RGB", (160, 120), (40, 120, 40)).save(png_path, format="PNG")
        payload["images"][0].update(filename=png_path.name, path=str(png_path))
        write_session_metadata(tmp, "sess-png", payload)

        clean = _client(tmp).get("/image/img-1/clean")

        assert clean.status_code == 200
        assert clean.headers["content-type"] == "image/png"
        assert clean.content == png_path.read_bytes()


def test_fixture_pair_keeps_precedence_on_both_routes_and_reads_no_detections(monkeypatch) -> None:
    class _Spy(dict):
        reads: list[str] = []

        def get(self, key, *a):
            if key == "best_image_detections":
                _Spy.reads.append(key)
            return super().get(key, *a)

        def __getitem__(self, key):
            if key == "best_image_detections":
                _Spy.reads.append(key)
            return super().__getitem__(key)

    def _bomb(*a, **k):
        raise AssertionError("overlay rendered although the fixture pair exists")

    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as fixtures:
        tmp = Path(d)
        _real_session(tmp, "sess-fx-routes", detections=DETS)
        fx = _fixture_dir(fixtures)
        monkeypatch.setattr(main_module, "_ANNOTATED_IMAGES_DIR", fx)
        monkeypatch.setattr(main_module, "_render_annotated_from_detections", _bomb)
        real_read = main_module.read_session_metadata
        monkeypatch.setattr(main_module, "read_session_metadata", lambda *a, **k: _Spy(real_read(*a, **k)))
        client = _client(tmp)

        clean = client.get("/image/img-1/clean")
        annotated = client.get("/image/img-1/annotated")

        assert clean.status_code == 200 and annotated.status_code == 200
        assert clean.content == (fx / "clean_frame-0001.png").read_bytes()
        assert annotated.content == (fx / "annotated_frame-0001.png").read_bytes()
        assert clean.headers["content-type"] == annotated.headers["content-type"] == "image/png"
        assert _Spy.reads == [], "persisted detections were read for a fixture pair"


def test_real_session_without_detections_refuses_annotated_by_session_and_serves_clean() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-nodets", detections=None)
        client = _client(tmp)

        annotated = client.get("/image/img-1/annotated")
        clean = client.get("/image/img-1/clean")

        assert annotated.status_code == 404
        assert "sess-nodets" in annotated.json()["detail"]
        assert clean.status_code == 200
        assert clean.content == Path(payload["images"][0]["path"]).read_bytes()


def test_annotated_route_refuses_an_image_that_is_not_the_sessions_best_image() -> None:
    """Detections are persisted for the best image only; another image of
    the session must not be drawn with them."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        payload = _real_session(tmp, "sess-other", detections=DETS)
        second = Path(d) / "sessions" / "sess-other" / "second.jpg"
        Image.new("RGB", (160, 120), (90, 90, 90)).save(second, format="JPEG")
        payload["images"].append({"image_id": "img-2", "filename": "second.jpg", "path": str(second)})
        write_session_metadata(tmp, "sess-other", payload)
        client = _client(tmp)

        assert client.get("/image/img-2/annotated").status_code == 404
        assert "sess-other" in client.get("/image/img-2/annotated").json()["detail"]
        assert client.get("/image/img-2/clean").status_code == 200


def test_malformed_persisted_detections_refuse_the_annotated_route_by_field() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _real_session(tmp, "sess-px", detections=[{"label": "aphid", "score": 0.5, "x_min": 10.0, "y_min": 12.0, "x_max": 4056, "y_max": 55.0}])
        client = _client(tmp)

        annotated = client.get("/image/img-1/annotated")

        assert annotated.status_code == 422
        assert "x_min" in annotated.json()["detail"]
        assert client.get("/image/img-1/clean").status_code == 200


def test_unknown_image_is_404_on_both_routes() -> None:
    with tempfile.TemporaryDirectory() as d:
        client = _client(Path(d))
        assert client.get("/image/nope/clean").status_code == 404
        assert client.get("/image/nope/annotated").status_code == 404


def test_pair_push_carries_the_same_bytes_as_the_two_routes(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _real_session(tmp, "sess-same", detections=DETS)
        client = _client(tmp)
        route_clean = client.get("/image/img-1/clean").content
        route_annotated = client.get("/image/img-1/annotated").content

        payload = main_module.read_session_metadata(tmp, "sess-same")
        assert _resolve_pair_raw(_settings(tmp), "sess-same", "img-1", payload) == (route_clean, route_annotated)

        sent: dict = {}

        def _capture(settings, session_id, image_id, clean_bytes, clean_ct, annotated_bytes, annotated_ct, artifact_hash):
            sent.update(clean=clean_bytes, annotated=annotated_bytes)
            return True, '{"status":"accepted"}'

        monkeypatch.setattr(main_module, "_push_best_image_pair_once", _capture)
        settings = _settings(tmp)
        main_module._run_best_image_push_sequence(settings, "sess-same")

        expected = compress_pair_to_cap(
            route_clean,
            route_annotated,
            settings.best_image_max_bytes_per_image,
            settings.best_image_max_bytes_combined,
        )
        assert (sent["clean"], sent["annotated"]) == (expected[0], expected[2])


def test_pair_push_keeps_its_all_or_nothing_404(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _real_session(tmp, "sess-half", detections=None)
        calls: list = []
        monkeypatch.setattr(main_module, "_push_best_image_pair_once", lambda *a, **k: calls.append(a) or (True, "{}"))

        main_module._run_best_image_push_sequence(_settings(tmp), "sess-half")

        assert calls == [], "a pair without an annotated image must not be pushed"
        assert _client(tmp).get("/image/img-1/clean").status_code == 200
