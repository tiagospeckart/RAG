# RAG

Project for developing a Retrieval-Augmented Generation with LLMs and data ingested from documents.

## Tech Stack

- Python 3.11
- Poetry: package management
- FastAPI: REST API
- Uvicorn: ASGI server that powers FastAPI for serving HTTP requests.
- LangChain: cahin orchestration
- Pydantic: data validation and settings management using Python type annotations.

## Installation

### Locally

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environment configuration. You can use your preferred Python environment setup easily, like `conda` or `venv`, and `poetry` will use your environment. If no environmet is setup, `poetry` will create a `.venv` folder in this project's root folder.

Run `pipx install poetry`

#### Prerequisites

- Any version of `python 3.11` in your `$PATH`
- Configure your own environment variables. You can use an `.env` file using the `example.env` as reference.
  - Missing fields may result in validation errors

#### Steps

- Run `poetry install` to download the dependencies
- Run run it with `poetry run python app`

### With Docker

Run `docker compose up`

## About

This project has been strongly influenced and supported by other amazing projects like PrivateGPT and LangChain.
