# RAG: Retrieval-Augmented Generation with LLMs

## Introduction

RAG is a project aimed at developing a Retrieval-Augmented Generation system utilizing Large Language Models (LLMs) and data ingested from documents.

## Tech Stack

- **Python 3.11**: Programming language used for development.
- **Poetry**: Dependency management and virtual environment configuration.
- **FastAPI**: Framework for building REST APIs.
- **Uvicorn**: ASGI server powering FastAPI for serving HTTP requests.
- **LangChain**: Chain orchestration tool.
- **Pydantic**: Data validation and settings management using Python type annotations.

## Installation

### Local Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environment configuration. You can use your preferred Python environment setup easily, like `conda` or `venv`, and `poetry` will use your current environment. If no environmet is found, poetry will create a `.venv` folder in this project's root folder.

#### Prerequisites

- Latest version of Poetry: `pipx install poetry`
- Ensure Python 3.11 is available in your `$PATH`
- Configure environment variables using an `.env` file (refer to `example.env`).
  - Missing fields may result in validation errors

#### Steps

1. Clone the repository and navigate to the project directory.
2. Run `poetry install` to install project dependencies.
3. Set up your environment variables in the `.env` file.
4. Start the application with `make run`, which runs `poetry run python -m app`.

### Docker Installation

Run `docker compose up -d`

- **Note**: This may take several minutes, as the current build uses local embedding models that require a lot of dependencies.

While running container, you can upload a `Markdown` file at the data volume.

## About

This project has been strongly influenced and supported by other amazing projects like PrivateGPT and LangChain.
