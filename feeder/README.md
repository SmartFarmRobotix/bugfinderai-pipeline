# Session feeder

A command-line tool that submits one stored session to the inference service and exits.

**Session folders.** `REPLAY_DATA_ROOT/sessions/<session_id>/` with a `manifest.json` (session id, crop,
start and end time, the list of pictures with their time stamps and optional positions; the schema is
`app/manifest.py`) and an `images/` directory of JPEG or PNG files. The feeder takes the first folder, by name, that it
has not submitted yet (name the folders so that this order is chronological; it starts over when all are
done), or the one named by `REPLAY_SESSION_ID`, and records what it submitted in a marker file.

**Profiles.** `REPLAY_DEPLOYMENT_PROFILE`: `cloud` uploads the pictures over HTTP (`x86_ci`, the default,
does the same for development); `mixed` and `edge` register the folder path with an inference service on the
same device; `auto` measures the uplink first and picks one of the three for this session: at 5 Mbit/s or
more Cloud, below that Mixed, and Edge when the probe fails (connection refused, time-out, bad answer or no
samples). The probe measures the round trip with a one-byte upload, then uploads payloads that double from
32 KiB until one lasts a second (at most 1 MiB), and takes the median rate corrected for the round trip.

**Outputs.** For every submission (and every probe in `auto` mode) the feeder writes its network accounting
(bytes sent per session) under `<data root>/accounting/` and its marker file under the data root, so the data
root must be writable. If `REPLAY_ACCOUNTING_FILE` is set, it also mirrors the accounting record there. If
`REPLAY_RESULT_FILE` is set, it writes the push result (session id, PDM session id, report id, pushed) there
after a push, and always in `auto` mode, with the probe decision. A `--dry-run` writes nothing.

**Capture folders.** `python -m app.ingest <date directory> --session-id <id> --crop <crop> --data-root <data root>`
turns a camera's capture directory (epoch-named folders, each with an `rgb.jpg`) into a session folder: the
folder epochs become the time stamps, GPS coordinates are read from the EXIF data when present, and the
free-text EXIF annotations land in a `capture-exif.json` sidecar beside the manifest.

**Run.** `cp .env.example .env`, then `python -m app.main` (or `--dry-run` to see the selection without
submitting; set `REPLAY_DATA_ROOT` and `REPLAY_INFERENCE_BASE_URL` for a run outside Docker), or the `feeder` service of the compose files under `deploy/`. Tests: `python -m pytest`.
