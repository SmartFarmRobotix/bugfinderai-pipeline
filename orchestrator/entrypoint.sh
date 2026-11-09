#!/bin/bash
set -e

# The edge profile has no Gatekeeper and
# must not be blocked from starting for lacking Gatekeeper credentials it
# will never use -- this shell-level check must mirror config.py's own
# profile-conditional validator, not enforce GK creds unconditionally.
if [ "${ORCH_DEPLOYMENT_PROFILE:-cloud}" != "edge" ]; then
  if [ -z "${ORCH_GATEKEEPER_USERNAME}" ]; then
    echo "ORCH_GATEKEEPER_USERNAME is not set in env"
    exit 1
  fi

  if [ -z "${ORCH_GATEKEEPER_PASSWORD}" ]; then
    echo "ORCH_GATEKEEPER_PASSWORD is not set in env"
    exit 1
  fi
fi

mkdir -p "$(dirname "${ORCH_DATABASE_URL#sqlite:///}")" 2>/dev/null || true

echo "Starting db migrations"
alembic upgrade head
echo "Finished db migrations"

echo "Starting Uvicorn server"
exec uvicorn --host 0.0.0.0 --port "$1" --app-dir=app 'main:app'
