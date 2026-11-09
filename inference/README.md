# Inference service

A FastAPI service that runs the detector over a session of pictures and scores the session.

**Session intake.** `POST /api/v1/sessions` with the pictures as multipart files (Cloud scenario), or
`POST /api/v1/sessions/path` with the path of a session folder the service can read (Mixed and Edge
scenarios: the feeder and the service share one volume, so no picture bytes cross the network). Optional
fields: `session_id`, `parcel_id`, `latitude`, `longitude`, `time_start`, `time_end`, `crop`, `pest` (an
EPPO code), the growth stage, `metadata`. A path intake names a session folder under `INFERENCE_DATA_ROOT`
(default `/code/app/data`) that holds `manifest.json` and `images/`. Only JPEG and PNG pictures are accepted.

**Detection.** The detector is selected by `INFERENCE_DETECTOR_IMPL`: `ort` (ONNX Runtime, CPU or CUDA,
`INFERENCE_ONNX_MODEL_PATH`), `tensorrt` (a Jetson engine, `INFERENCE_ENGINE_PATH`), `lookup` (the default: a fixed stub for
tests) or `null`. The model file has a provenance record beside it (`<model>.provenance.json`) and a
`designation-manifest.json`; the service verifies the SHA-256 of the model against the record and refuses a
model it cannot verify. A record with a `tiling` entry selects the tiled path: the picture is cut into
tiles of the recorded size and stride, every tile is run, and the detections are merged across tiles by a
fixed union rule (`app/detectors/region_merge.py`). `INFERENCE_DETECTOR_PROFILE` names the expectations the
loaded model must meet: the released ONNX file runs under `legacy-full-frame` (which accepts any model), a
Jetson engine built from it under `edge-s-fp16`.

**Scoring and pushes.** Each picture gets a score from the detections of the target classes
(`INFERENCE_TARGET_CLASS_SET`, `INFERENCE_CONFIDENCE_FLOOR`); the session's score and its best picture are
recorded. `POST /api/v1/sessions/{id}/push` (the feeder calls it after a submission) sends the session result
to the orchestrator when `INFERENCE_ORCH_PUSH_ENABLED` is set, through the Gatekeeper proxy (service name
`INFERENCE_ORCH_SERVICE_NAME`, default `orchestrator`) or directly to `INFERENCE_ORCH_BASE_URL` with the
`X-Edge-Shared-Secret` header (`INFERENCE_ORCH_EDGE_SHARED_SECRET`), and
with `INFERENCE_BEST_IMAGE_PUSH_ENABLED` the best picture and its annotated copy to the Reporting service,
always through the Gatekeeper proxy.

**Other routes.** `GET /api/v1/sessions/{id}` (the record with its provenance), `GET
/api/v1/sessions/{id}/results` (per-picture scores), `GET /api/v1/sessions/{id}/accounting` (the bytes the
service sent for that session), `GET /image/{id}/bytes`, `/clean` and `/annotated`, `POST /api/v1/probe/sink`
(the feeder's uplink probe target), `GET /healthz`. The routes under `/internal/` (`acceptance-smoke`,
`detect-debug`, `housekeeping/run`) have no authentication and belong on an internal network.

**Authentication.** Through the Gatekeeper (`INFERENCE_USING_GATEKEEPER=true` with
`INFERENCE_GATEKEEPER_BASE_URL`, `_USERNAME`, `_PASSWORD`; a token), or a pre-shared key in the `X-API-Key` header
(`INFERENCE_EXTERNAL_API_KEY`), which is how the feeder authenticates. With neither configured, the routes
are open.

**Run.** See the compose files under `deploy/`. Images: `Dockerfile` (CPU), `Dockerfile.cloud-gpu` (CUDA),
`Dockerfile.jetson-gpu` (TensorRT on JetPack 6.2). Tests: `python -m pytest` from this directory; they need
no model file and no GPU.
