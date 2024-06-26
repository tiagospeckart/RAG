FROM python:3.11.9-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.1

# Generate workable requirements.txt from Poetry dependencies
FROM base AS poetry-build

WORKDIR /app

RUN pip install "poetry==$POETRY_VERSION"

RUN python -m pip install --no-cache-dir --upgrade poetry

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM base AS pip-build
WORKDIR /wheels
COPY --from=poetry-build /app/requirements.txt .
RUN pip install -U pip  \
    && pip wheel -r requirements.txt

FROM base
COPY --from=pip-build /wheels /wheels

COPY ./data/docs/*.md ./data/docs/

RUN pip install -U pip \
    && pip install \
    --no-index \
    -r /wheels/requirements.txt \
    -f /wheels \
    && rm -rf /wheels

ADD . .

CMD ["poetry run python -m app"]