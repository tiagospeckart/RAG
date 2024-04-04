FROM python:3.11-slim

# Install poetry
RUN pip install poetry

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./.env ./
COPY ./app ./app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
