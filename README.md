# BugFinderAI pest-session pipeline

A pipeline that runs a pest-detection session for a field robot and turns it into a farmer-facing report
through the OpenAgri services. It was built in the OpenAgri open call project SIP12 (BugFinderAI) by Smart Farm
Robotix, and it was run in the field in three scenarios:

- **Cloud.** The pictures of a session go to an inference service in the cloud. The detections, the risk
  assessment and the PDF report are made there.
- **Mixed.** The detection runs on the robot. Only the detections and one best-image pair go to the cloud,
  where the assessment and the report are made.
- **Edge.** Everything runs on the robot. The report is rendered on the device and synchronised to the cloud
  later, when a link is available.

The session feeder measures the uplink at the start of a session and picks the scenario by itself, or runs a
fixed one.

## What is in this repository

| Directory | What it is |
|---|---|
| `orchestrator/` | The session orchestrator (FastAPI). It receives the detections of a session, assesses the pest risk by two paths (the weather rules of the OpenAgri Pest and Disease Management service, and a fuzzy rule base), composes the session context in the OpenAgri common semantic model and calls the OpenAgri Reporting service for the PDF report. Its edge profile assesses without weather, renders the report locally and keeps a synchronisation queue. |
| `inference/` | The inference service (FastAPI). It runs the D-FINE detector over the tiles of each picture on ONNX Runtime (CPU or CUDA) or on a TensorRT engine (Jetson), scores the session, records its own outbound traffic per session, and pushes the results to the orchestrator and the best image pair to Reporting. |
| `feeder/` | The session feeder (command-line tool). It takes a session folder, probes the uplink, selects the scenario and submits the session to the inference service. |
| `deploy/` | Docker Compose files of the three scenarios: `cloud/` (the OpenAgri services, the orchestrator and the inference service on a cloud host), `mixed/` and `edge/` (the robot side), each with a `.env.example`. |
| `MODEL.md` | The model card of the trained network. The network itself is a release asset. |

The four OpenAgri services the pipeline talks to (Gatekeeper, Pest and Disease Management, Weather, Reporting)
are upstream projects. The pipeline runs against the SmartFarmRobotix forks of these services, branch
`sip12`, which carry the fixes and additions the pipeline needs:
[Gatekeeper](https://github.com/SmartFarmRobotix/OpenAgri-GateKeeper),
[Pest and Disease Management](https://github.com/SmartFarmRobotix/OpenAgri-PestAndDiseaseManagement),
[Weather](https://github.com/SmartFarmRobotix/OpenAgri-WeatherService),
[Reporting](https://github.com/SmartFarmRobotix/OpenAgri-ReportingService).
The forks are git submodules of this repository under `deploy/services/`, pinned at the published commit, so a
clone with `--recurse-submodules` brings everything the compose files build.

## The trained network

The GitHub release of this repository (tag `v1.0.0`, under Releases) carries the detector: `dfine-s-aphid-v2.onnx` (D-FINE S, one class, 1280×1280 input), its
provenance record `dfine-s-aphid-v2.onnx.provenance.json` and `designation-manifest.json`. Download the three
files into one directory and rename the network and its record:

```
model/
  model.onnx (the downloaded dfine-s-aphid-v2.onnx)
  model.onnx.provenance.json (the downloaded dfine-s-aphid-v2.onnx.provenance.json)
  designation-manifest.json
```

The inference service verifies the SHA-256 of the network against the record before it loads it. See
`MODEL.md` for what the network does and how well.

## Quick start on one machine (CPU)

This runs the Cloud scenario on one computer without a GPU: the four OpenAgri services, the orchestrator, the
inference service on the CPU and the feeder.

1. Clone with the forks: `git clone --recurse-submodules <this repository>`, or in an existing clone
   `git submodule update --init`. The compose file builds the four OpenAgri service images from
   `deploy/services/`.
2. `cd deploy/cloud && cp .env.example .env` and fill the values marked REQUIRED: the secrets, the database
   passwords, the pre-shared key of the inference service, the model directory and the directory that holds
   your `sessions/<id>/` folders.
3. Start the services: `docker compose -f compose.yml up -d`. Wait until every service shows healthy in
   `docker compose ps`. The Gatekeeper's web interface is on port 8001.
4. Start the inference service: `docker compose -f compose.inference.yml --profile cpu up -d`.
5. Submit a session: `docker compose -f compose.inference.yml --profile feeder run --rm feeder`. The feeder
   prints the session id and writes `result.json` and `accounting.json` into the output directory.
6. The report. On the orchestrator, with a Gatekeeper token in `Authorization: Bearer …`,
   `GET http://localhost:8021/api/v1/sessions/<session id>/report` returns the report id and its state, and
   `GET /api/v1/sessions/<session id>` the session record with the two assessments. The PDF itself is in the
   Reporting service, through the Gatekeeper proxy with the same token:
   `GET http://localhost:8001/api/proxy/reporting/api/v1/openagri-report/<report id>/`.

With a CUDA GPU use `--profile gpu` in step 4.

## A session folder

The feeder reads session folders of this shape:

```
sessions/
  <session_id>/
    manifest.json
    images/
      *.jpg | *.png
```

`manifest.json` names the session, the crop, the start and end time and the pictures (file name and time
stamp, optional position). `feeder/app/manifest.py` is the schema. Pictures are expected at 4056×3040 pixels;
the service tiles them at 1280×1280 with a stride of 1024. A folder is replayed once; the feeder keeps a marker
file of what it has submitted, and `REPLAY_SESSION_ID` selects a folder explicitly.

## On the robot

`deploy/mixed/compose.yml` runs the inference service and the feeder on the device. With the `jetson` profile
the inference service is the TensorRT image; with the `cpu` profile it is the ONNX Runtime image (on the robot
it runs the `cloud` execution target, so the designation manifest lists the ONNX file under `cloud`). The
feeder in `auto` mode probes the uplink: at 5 Mbit/s or more it sends the session to the cloud inference
service (Cloud), below that it submits to the device (Mixed). A probe failure selects Edge, which needs the
orchestrator of the edge stack: in this stack the session is submitted to the device and its result cannot
be pushed, so a device that must work without any link runs `deploy/edge/compose.yml` instead.

`deploy/edge/compose.yml` adds the orchestrator in its edge profile on an internal network. The report is
rendered on the device and queued. When a link is available, `docker compose --profile sync run --rm sync`
uploads the queued reports to the cloud Reporting service and asks the cloud orchestrator to re-assess the
sessions with weather data.

The TensorRT engine for a Jetson is built from the released ONNX file on the device with `trtexec`
(TensorRT 10.3, JetPack 6.2; fp16; batch profile 1 to 12 at 3×1280×1280). Save it as `model/model.engine`
with a provenance record of its own, `model/model.engine.provenance.json`: the engine's SHA-256 as
`file_sha256`, `execution_target` `jetson`, `variant` `s`, `precision` `fp16`, `imgsz` 1280, `class_names`,
`class_map_hash` and the same `tiling` as the ONNX record. The engine's SHA-256 must be listed under `jetson`
in `designation-manifest.json`. The inference service refuses an engine it cannot verify.

## Development

Each service has a `requirements.txt`; the tests are under `feeder/tests`, `inference/app/tests` and
`orchestrator/app/tests` (`python -m pytest` from the service directory, in a virtual environment with the
dev requirements where a `requirements-dev.txt` exists). The tests run without a GPU, without the network and
without the model file; the orchestrator's report tests need poppler-utils (`pdftotext` and `pdfimages`).

## Limits

- The network was trained on a small set of pictures of one crop; see `MODEL.md`. It is a proof of the
  pipeline, not a deployable detector.
- The orchestrator's fuzzy rule base and the Pest and Disease Management rules cover the crops and pests of
  the project.
- The pipeline was run on a Jetson Orin Nano 8 GB and on x86 hosts; other devices were not tried.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`. The trained network is a derivative of D-FINE's public COCO
checkpoint (Apache-2.0). Three parts of `orchestrator/app/report_pdf/` are under other licences and are listed
in `NOTICE`: the PDF builder derived from the OpenAgri Reporting service (EUPL-1.2), the GNU FreeFont files
(GPL with the font-embedding exception) and the two logos.

## Funding

This software was developed in BugFinderAI (SIP12), a sub-project of the open call of the OpenAgri project
(https://horizon-openagri.eu/), which has received funding from the EU's Horizon Europe research and innovation
programme under Grant Agreement no. 101134083. Views and opinions expressed are those of the authors only and do
not necessarily reflect those of the European Union or the granting authority; neither can be held responsible
for them.
