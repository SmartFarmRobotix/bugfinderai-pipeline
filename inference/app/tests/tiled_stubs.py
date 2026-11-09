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
Shared hermetic stubs for the tiled-inference tests: artifact + provenance writers, a scripted
per-tile output source, a stub ONNX Runtime session, and a fake TensorRT
runtime. No weights, no engine, no GPU.

FAKE-FIDELITY. Each fake models the vendor's DOCUMENTED contract, not what
the detector under test happens to expect:

* ``StubInferenceSession`` -- ONNX Runtime Python API,
  https://onnxruntime.ai/docs/api/python/api_summary.html
  (``InferenceSession.run(output_names, input_feed)``,
  ``get_inputs/get_outputs`` returning ``NodeArg`` with ``name``,
  ``shape`` (symbolic dims as strings) and ``type``). ``run`` refuses, with
  ORT's own ``InvalidArgument``, an input feed whose names differ from the
  model's declared inputs, whose dtype differs from the declared type, or
  whose fixed dimensions differ from the declared shape -- and accepts any
  value along a symbolic (dynamic) dimension. The D-FINE export declares
  ``images [N,3,S,S]`` float and ``orig_target_sizes [N,2]`` int64 with a
  dynamic batch axis, so batch N > 1 is legal.
* ``FakeEngine`` / ``FakeContext`` -- TensorRT Python API,
  https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/
  (``ICudaEngine.get_tensor_shape`` reports -1 for a dynamic dimension;
  ``ICudaEngine.get_tensor_profile_shape(name, profile_index)`` returns the
  (min, opt, max) shapes of an input's optimization profile;
  ``IExecutionContext.set_input_shape`` returns ``False`` for a shape
  outside the profile; ``get_tensor_shape`` on an output stays -1 until
  every input shape is set; ``execute_async_v3`` returns ``False`` when it
  cannot run).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from app.detectors.dfine_common import DFINE_STOCK_COCO_CLASS_NAMES

ABSENT = object()

N_QUERIES = 300

REGION_CLASS = "aphid_infestation_region"


class ArtifactFiles(NamedTuple):
    path: Path
    provenance_path: Path
    manifest_path: Path
    sha256: str


def class_map_hash(class_names: Sequence[str]) -> str:
    """Independent re-computation of the canonical class-map hash (same
    rationale as the fixtures in test_dfine_ort.py: never the SUT's own)."""
    return hashlib.sha256("\n".join(class_names).encode("utf-8")).hexdigest()


def write_artifact(
    tmp_path: Path,
    *,
    kind: str = "onnx",
    imgsz: Any = 1280,
    tiling: Any = ABSENT,
    class_names: Optional[Sequence[str]] = (REGION_CLASS,),
    execution_target: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    artifact_bytes: Optional[bytes] = None,
) -> ArtifactFiles:
    """A fake artifact file, its provenance manifest, and a designation
    manifest naming it. ``imgsz``/``tiling`` may be ``ABSENT`` (key not
    written) or any JSON value; ``class_names=None`` writes the stock COCO
    shape (no ``class_names`` key, the stock list's hash)."""
    execution_target = execution_target or ("jetson" if kind == "engine" else "cloud")
    payload = artifact_bytes if artifact_bytes is not None else f"fixture-{kind}-{imgsz!r}-{tiling!r}".encode()
    path = tmp_path / ("model.engine" if kind == "engine" else "model.onnx")
    path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()

    provenance: Dict[str, Any] = {
        "artifact_kind": kind,
        "execution_target": execution_target,
        "file_sha256": sha256,
        "file_path": f"{kind}s/{path.name}" if kind == "engine" else f"onnx/{path.name}",
        "pt_provenance": {
            "model_name": "dfine-l",
            "source": "dfine-finetuned-example",
            "license": "Apache-2.0",
            "file_sha256": "ptsha000",
        },
    }
    if kind == "engine":
        provenance["source_onnx_sha256"] = "onnxsha000"
        provenance["source_onnx_provenance_file"] = "provenance/model.onnx.json"
    if imgsz is not ABSENT:
        provenance["imgsz"] = imgsz
    if tiling is not ABSENT:
        provenance["tiling"] = tiling
    if class_names is None:
        provenance["class_map_hash"] = class_map_hash(DFINE_STOCK_COCO_CLASS_NAMES)
    else:
        provenance["class_names"] = list(class_names)
        provenance["class_map_hash"] = class_map_hash(class_names)
    provenance.update(extra or {})

    provenance_path = tmp_path / f"{path.name}.provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    manifest_path = tmp_path / "designation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "targets": {execution_target: {"artifacts": {sha256: {"designation_status": "integration-only"}}}},
            }
        )
    )
    return ArtifactFiles(path, provenance_path, manifest_path, sha256)


def write_frame(path: Path, width: int, height: int, seed: int = 3) -> Path:
    """A synthetic RGB frame with non-uniform content (a colour gradient plus
    low-amplitude noise, kept in uint8 so a 4056x3040 frame stays cheap),
    JPEG or PNG by suffix."""
    rng = np.random.default_rng(seed)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (np.arange(width, dtype=np.uint32) * 247 // max(width - 1, 1)).astype(np.uint8)[None, :]
    frame[..., 1] = (np.arange(height, dtype=np.uint32) * 247 // max(height - 1, 1)).astype(np.uint8)[:, None]
    frame[..., 2] = ((np.arange(width)[None, :] + np.arange(height)[:, None]) % 247).astype(np.uint8)
    frame += rng.integers(0, 8, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(frame, "RGB").save(path)
    return path


class TileScript:
    """The per-tile raw outputs a stub runtime returns, consumed in call
    order: the k-th tile the detector presents (row-major, batch chunks in
    sequence) receives ``tiles[k]``. Because the detector's tile order is
    what places each tile's boxes at its frame offset, an out-of-order
    detector produces wrong regions rather than passing silently."""

    def __init__(self, tiles: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self._tiles = list(tiles)
        self.cursor = 0

    def take(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cursor + n > len(self._tiles):
            raise AssertionError(
                f"tile script exhausted: {n} more tile(s) asked at cursor {self.cursor} of {len(self._tiles)}"
            )
        rows = self._tiles[self.cursor : self.cursor + n]
        self.cursor += n
        return (
            np.stack([r[0] for r in rows]).astype(np.int64),
            np.stack([r[1] for r in rows]).astype(np.float32),
            np.stack([r[2] for r in rows]).astype(np.float32),
        )


def pad_tile(labels: Sequence[int], boxes: Sequence[Sequence[float]], scores: Sequence[float]):
    """One tile's detections padded to the full top-300 the runtime returns
    (padding rows: label 0, zero box, score 0.0)."""
    n = len(scores)
    out_labels = np.zeros(N_QUERIES, dtype=np.int64)
    out_boxes = np.zeros((N_QUERIES, 4), dtype=np.float32)
    out_scores = np.zeros(N_QUERIES, dtype=np.float32)
    out_labels[:n] = labels
    if n:
        out_boxes[:n] = np.asarray(boxes, dtype=np.float32)
    out_scores[:n] = scores
    return out_labels, out_boxes, out_scores


def script_from_raw(raw: Dict[str, Any], *source_files: str) -> TileScript:
    """The golden raw-tile fixture's tiles of the named source images, each
    image's tiles in tile order, images in the order named, as a stub script
    (the images are then presented to the detector in that same order)."""
    padded = []
    for source_file in source_files:
        tiles = [t for t in raw["tiles"] if t["source_file"] == source_file]
        tiles.sort(key=lambda t: t["tile_index"])
        padded.extend(pad_tile(t["labels"], t["boxes"], t["scores"]) for t in tiles)
    return TileScript(padded)


def clip_script(script: TileScript, limit: int) -> TileScript:
    """A copy of ``script`` whose boxes are clipped into ``[0, limit]`` -- a
    runtime whose outputs lie inside the tile."""
    return TileScript([(l, np.clip(b, 0, limit), s) for l, b, s in script._tiles])


def load_fixture(name: str) -> Dict[str, Any]:
    return json.loads((Path(__file__).resolve().parent / "fixtures" / "tiled" / name).read_text())


def sha256_rows(array: np.ndarray) -> List[str]:
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in array]


class _NodeArg:
    def __init__(self, name: str, shape: list, type_: str) -> None:
        self.name, self.shape, self.type = name, shape, type_


class StubInferenceSession:
    """See the module docstring for the ONNX Runtime contract it models."""

    def __init__(self, model_path: str, providers: Sequence[str], script: TileScript, imgsz: int) -> None:
        self.model_path = model_path
        self.requested_providers = list(providers)
        self._script = script
        self._inputs = [
            _NodeArg("images", ["N", 3, imgsz, imgsz], "tensor(float)"),
            _NodeArg("orig_target_sizes", ["N", 2], "tensor(int64)"),
        ]
        self._outputs = [
            _NodeArg("labels", ["N", N_QUERIES], "tensor(int64)"),
            _NodeArg("boxes", ["N", N_QUERIES, 4], "tensor(float)"),
            _NodeArg("scores", ["N", N_QUERIES], "tensor(float)"),
        ]
        self.calls: List[Dict[str, Any]] = []

    def get_providers(self) -> List[str]:
        return list(self.requested_providers)

    def get_inputs(self) -> List[_NodeArg]:
        return list(self._inputs)

    def get_outputs(self) -> List[_NodeArg]:
        return list(self._outputs)

    def run(self, output_names, input_feed, run_options=None):
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

        declared = {arg.name: arg for arg in self._inputs}
        if set(input_feed) != set(declared):
            raise InvalidArgument(
                f"input feed names {sorted(input_feed)!r} differ from the model's inputs {sorted(declared)!r}"
            )
        for name, array in input_feed.items():
            arg = declared[name]
            want = np.float32 if arg.type == "tensor(float)" else np.int64
            if np.asarray(array).dtype != want:
                raise InvalidArgument(f"Unexpected input data type for {name}: {np.asarray(array).dtype}, expected {want}")
            shape = np.asarray(array).shape
            if len(shape) != len(arg.shape) or any(
                isinstance(dim, int) and dim != got for dim, got in zip(arg.shape, shape)
            ):
                raise InvalidArgument(f"Got invalid dimensions for input: {name}, got {list(shape)}, expected {arg.shape}")
        images = input_feed["images"]
        n = images.shape[0]
        if input_feed["orig_target_sizes"].shape[0] != n:
            raise InvalidArgument("images and orig_target_sizes disagree on the batch dimension")
        self.calls.append(
            {
                "images_shape": tuple(images.shape),
                "images_dtype": images.dtype,
                "images_sha256": sha256_rows(images),
                "orig_target_sizes": input_feed["orig_target_sizes"].tolist(),
                "output_names": list(output_names),
            }
        )
        labels, boxes, scores = self._script.take(n)
        by_name = {"labels": labels, "boxes": boxes, "scores": scores}
        return [by_name[name] for name in output_names]


class StubInferenceSessionFactory:
    """Callable stand-in for the ``onnxruntime.InferenceSession`` class."""

    def __init__(self, script: TileScript, imgsz: int) -> None:
        self.script, self.imgsz = script, imgsz
        self.sessions: List[StubInferenceSession] = []

    def __call__(self, model_path, providers):
        session = StubInferenceSession(model_path, providers, self.script, self.imgsz)
        self.sessions.append(session)
        return session

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return [call for session in self.sessions for call in session.calls]


def install_ort_stub(monkeypatch, script: TileScript, imgsz: int) -> StubInferenceSessionFactory:
    import onnxruntime as ort

    factory = StubInferenceSessionFactory(script, imgsz)
    monkeypatch.setattr(ort, "InferenceSession", factory)
    return factory


class FakeTensorIOMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class FakeTrtModule:
    TensorIOMode = FakeTensorIOMode

    @staticmethod
    def nptype(dtype):
        return dtype


class FakeDeviceMem:
    _registry: Dict[int, "FakeDeviceMem"] = {}

    def __init__(self, nbytes: int) -> None:
        self.buffer = bytearray(nbytes)
        FakeDeviceMem._registry[id(self)] = self

    def __int__(self) -> int:
        return id(self)


class FakeStream:
    handle = 0xF00D

    def synchronize(self) -> None:
        pass


class FakeCudaModule:
    def mem_alloc(self, nbytes: int) -> FakeDeviceMem:
        return FakeDeviceMem(nbytes)

    def memcpy_htod_async(self, dst: FakeDeviceMem, src: np.ndarray, stream) -> None:
        arr = np.ascontiguousarray(src)
        dst.buffer[: arr.nbytes] = arr.tobytes()

    def memcpy_dtoh_async(self, dst: np.ndarray, src: FakeDeviceMem, stream) -> None:
        dst[...] = np.frombuffer(bytes(src.buffer[: dst.nbytes]), dtype=dst.dtype).reshape(dst.shape)

    def Stream(self) -> FakeStream:
        return FakeStream()


class FakeEngine:
    """An engine built from the dynamic-batch D-FINE export: the batch
    dimension of every tensor reports -1, and each input's optimization
    profile is ``(min, opt, max)`` batch. I/O tensors are declared in a
    scrambled order so the detector must bind by name."""

    def __init__(self, imgsz: int, batch_profile: Tuple[int, int, int] = (1, 1, 1)) -> None:
        self.imgsz = imgsz
        self.batch_profile = batch_profile
        self._tensors = [
            ("orig_target_sizes", FakeTensorIOMode.INPUT, np.int64, (-1, 2)),
            ("boxes", FakeTensorIOMode.OUTPUT, np.float32, (-1, N_QUERIES, 4)),
            ("images", FakeTensorIOMode.INPUT, np.float32, (-1, 3, imgsz, imgsz)),
            ("labels", FakeTensorIOMode.OUTPUT, np.int64, (-1, N_QUERIES)),
            ("scores", FakeTensorIOMode.OUTPUT, np.float32, (-1, N_QUERIES)),
        ]

    @property
    def num_io_tensors(self) -> int:
        return len(self._tensors)

    def get_tensor_name(self, i: int) -> str:
        return self._tensors[i][0]

    def _row(self, name: str):
        return next(t for t in self._tensors if t[0] == name)

    def get_tensor_mode(self, name: str) -> str:
        return self._row(name)[1]

    def get_tensor_dtype(self, name: str):
        return self._row(name)[2]

    def get_tensor_shape(self, name: str) -> tuple:
        return self._row(name)[3]

    def get_tensor_profile_shape(self, name: str, profile_index: int):
        assert profile_index == 0
        assert self.get_tensor_mode(name) == FakeTensorIOMode.INPUT, "profile shapes exist for inputs only"
        tail = tuple(self._row(name)[3][1:])
        return [(b,) + tail for b in self.batch_profile]


class FakeContext:
    """Records every run's batch size in ``runs`` and the per-tile input
    hashes in ``images_sha256``; outputs come from the shared script."""

    def __init__(self, engine: FakeEngine, script: TileScript) -> None:
        self._engine = engine
        self._script = script
        self._mem: Dict[str, FakeDeviceMem] = {}
        self._input_shapes: Dict[str, tuple] = {}
        self.runs: List[int] = []
        self.shapes_set: List[Tuple[str, tuple]] = []
        self.images_sha256: List[str] = []
        self.orig_target_sizes: List[list] = []

    def set_input_shape(self, name: str, shape) -> bool:
        shape = tuple(shape)
        lo, _, hi = (p[0] for p in self._engine.get_tensor_profile_shape(name, 0))
        declared = self._engine.get_tensor_shape(name)
        if not lo <= shape[0] <= hi or shape[1:] != declared[1:]:
            return False
        self._input_shapes[name] = shape
        self.shapes_set.append((name, shape))
        return True

    def _inputs_set(self) -> bool:
        return len(self._input_shapes) == 2

    def get_tensor_shape(self, name: str) -> tuple:
        declared = self._engine.get_tensor_shape(name)
        if self._engine.get_tensor_mode(name) == FakeTensorIOMode.INPUT:
            return self._input_shapes.get(name, declared)
        if not self._inputs_set():
            return declared
        return (self._input_shapes["images"][0],) + tuple(declared[1:])

    def set_tensor_address(self, name: str, addr: int) -> None:
        self._mem[name] = FakeDeviceMem._registry[addr]

    def execute_async_v3(self, stream_handle) -> bool:
        if not self._inputs_set():
            return False
        n = self._input_shapes["images"][0]
        images = np.frombuffer(bytes(self._mem["images"].buffer), dtype=np.float32).reshape(
            (n, 3, self._engine.imgsz, self._engine.imgsz)
        )
        sizes = np.frombuffer(bytes(self._mem["orig_target_sizes"].buffer[: n * 16]), dtype=np.int64).reshape(n, 2)
        self.runs.append(n)
        self.images_sha256.extend(sha256_rows(images))
        self.orig_target_sizes.extend(sizes.tolist())
        labels, boxes, scores = self._script.take(n)
        for name, array in (("labels", labels), ("boxes", boxes), ("scores", scores)):
            arr = np.ascontiguousarray(array)
            self._mem[name].buffer[: arr.nbytes] = arr.tobytes()
        return True


def install_trt_runtime(
    detector, script: TileScript, imgsz: int, batch_profile: Tuple[int, int, int] = (1, 1, 1)
) -> FakeContext:
    """Give a constructed ``DFineTensorRTDetector`` a fake runtime owned by
    the calling thread (its CUDA-context affinity rule), and return the
    context so a test can read the runs it saw."""
    engine = FakeEngine(imgsz, batch_profile)
    context = FakeContext(engine, script)
    detector._runtime = (FakeTrtModule, FakeCudaModule(), engine, context)
    detector._runtime_owner_ident = threading.get_ident()
    return context
