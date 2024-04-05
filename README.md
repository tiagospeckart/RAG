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

- Use the `example.env` file for creating your configuration. Missing fields may result in validation errors.
- Run `poetry install` to download the dependencies
- Then run it with `poetry run python app`

## With Docker

Run `docker compose up`

## About

This project has been strongly influenced and supported by other amazing projects like PrivateGPT and LangChain.

