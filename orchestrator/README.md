# Session orchestrator

A FastAPI service. It receives the result of a detection session from the inference service, assesses the
pest risk and produces the report.

**Two risk paths.** The scalar result of the session (infestation score, crop, growth stage, position, time)
goes to the OpenAgri Pest and Disease Management service for its weather-rule assessment, and in parallel
through a fuzzy rule base (`app/fuzzy_threats/`) that combines the infestation score with the environmental
pressure. The two verdicts are fused (`app/fusion.py`), the session context is composed in the OpenAgri
common semantic model, and the OpenAgri Reporting service renders the PDF report.

**Profiles.** `ORCH_DEPLOYMENT_PROFILE=cloud` or `mixed`: the intake is authenticated
through the Gatekeeper, the two paths run, Reporting renders the report. `ORCH_DEPLOYMENT_PROFILE=edge`: no
Gatekeeper, the intake is authenticated by a shared secret, the assessment is the score band only, the PDF is
rendered locally (`app/report_pdf/`) and the session is queued (`app/sync_queue.py`). `python sync_worker.py`, run from `app/`
(the `sync` service of `deploy/edge/compose.yml`), uploads the queue to the cloud Reporting service and asks
the cloud orchestrator to re-assess the session with weather data.

**Routes.** `POST /api/v1/sessions/` (intake), `GET /api/v1/sessions/` and `GET /api/v1/sessions/{id}` (the
session records with the assessments and the report id), `GET /api/v1/sessions/{id}/assessments`,
`GET /api/v1/sessions/{id}/intake-state`, `GET /api/v1/sessions/{id}/report` (the report id and state; the PDF
is in the Reporting service), `POST /internal/housekeeping/run` (edge profile only), `GET /ready`,
`GET /health`. The `/api/v1` routes need a Gatekeeper token in the cloud and mixed profiles, and the
`X-Edge-Shared-Secret` header in the edge profile.

**Run.** In Docker: copy `../deploy/cloud/.env.example` to `../deploy/cloud/.env`, fill the values marked
REQUIRED, then `docker compose -f ../deploy/cloud/compose.yml up orchestrator`. For a development run outside
Docker: `cp .env.example .env`, fill the Gatekeeper credentials (or the edge secret), then
`pip install -r requirements-dev.txt`, `mkdir -p app/data`,
`export ORCH_DATABASE_URL=sqlite:///./app/data/orchestrator.db`, `alembic upgrade head` and
`uvicorn --app-dir app main:app --port 8020`. The database is SQLite by default (`alembic/` holds the
migrations). Tests: `python -m pytest` (the report tests need poppler-utils: `pdftotext` and `pdfimages`).

**Semantic model examples.** `examples/ocsm/` holds two records in the OpenAgri common semantic model (JSON-LD
against the published OCSM context) for the pest and crop pairs the rule bases cover: cabbage with the cabbage
aphid, and cucumber with the green peach aphid and the melon aphid; the first file carries powdery mildew of
grape as a second pair. The session context the orchestrator composes refers to pests and crops in these terms.
