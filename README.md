# RAG

Project for developing a Retrieval-Augmented Generation with LLMs and data ingested from documents.

## Tech Stack

- Python 3.11
- Poetry: package management
- FastAPI: REST API 
- Uvicorn: ASGI server that powers FastAPI for serving HTTP requests.
- LangChain
- Pydantic: data validation and settings management using Python type annotations.

## Installation

### Locally

#### Prerequisites

- Any version of `python 3.11` in your `$PATH`
- Configure your own environment variables. You can use an `.env` file using the `example.env` as reference.
  - Missing fields may result in validation errors!

_Optional_ 
- Use a `conda` environment 

#### Steps

- Run `poetry install` to download the dependencies
- Then run it with `poetry run python app`

### With Docker

Run `docker compose up`

## About

This project has been strongly influenced and supported by other amazing projects like PrivateGPT and LangChain.

