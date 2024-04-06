FROM python:3.11-slim

# Install poetry
RUN pip install poetry

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock* ./.env ./settings.yaml ./
COPY ./app ./app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
