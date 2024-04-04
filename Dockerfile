FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN apt-get update && apt-get install -y git

ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./.env ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

CMD langchain serve
