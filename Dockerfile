# Generate workable requirements.txt from Poetry dependencies
FROM python:3.11-slim AS requirements

RUN python -m pip install --no-cache-dir --upgrade poetry

WORKDIR /rag

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM python:3.11-slim as builder

WORKDIR /rag

# Copy requirements.txt from the 'requirements' stage
COPY --from=requirements /rag/requirements.txt ./requirements.txt
ADD . .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]