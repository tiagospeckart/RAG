FROM python:3.11.9-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.1

# Generate workable requirements.txt from Poetry dependencies
FROM base AS poetry-build

WORKDIR /rag

RUN pip install "poetry==$POETRY_VERSION"

RUN python -m pip install --no-cache-dir --upgrade poetry

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM base AS pip-build
WORKDIR /wheels
COPY --from=poetry-build /rag/requirements.txt .
RUN pip install -U pip  \
    && pip wheel -r requirements.txt

FROM base
COPY --from=pip-build /wheels /wheels

RUN pip install -U pip \
    && pip install \
    --no-index \
    -r /wheels/requirements.txt \
    -f /wheels \
    && rm -rf /wheels

COPY app app
COPY settings.yaml settings.yaml

ENTRYPOINT ["python", "-m", "app"]