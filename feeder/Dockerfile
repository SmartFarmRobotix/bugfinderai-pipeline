FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/app/

# The session folders are mounted at REPLAY_DATA_ROOT (default /app/data):
# <data root>/sessions/<session_id>/manifest.json + images/*.jpg|png
# One-shot command: the feeder selects one session, submits it and exits.
CMD ["python", "-m", "app.main"]
