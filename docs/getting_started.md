# Getting Started

This guide walks you through setting up a local development environment for
Gradient Mechanics.

## Setup

The project is developed using [uv](https://docs.astral.sh/uv/). After you have
installed `uv`, create a virtual environment and download all dependencies:

```sh
uv venv
uv sync
```

## Running tests

```sh
uv run pytest
```

## Building


### Wheel
Build the wheel:

```sh
uv build
```

### Docs

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) is used for documentation. You can see a live preview of the docs while editing by running:

```sh
uv run mkdocs serve
```

Doc package is built by running:

```sh
uv run mkdocs build
```
