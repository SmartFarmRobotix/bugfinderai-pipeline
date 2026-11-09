FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_VERSION_TO_INSTALL="24.0"

RUN apt-get update && \
    apt-get install -y \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -U pip==${PIP_VERSION_TO_INSTALL} && \
    rm -rf /tmp/pip* /root/.cache

WORKDIR /code

COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
