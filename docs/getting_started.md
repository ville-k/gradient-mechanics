# Getting Started

## Setup

This project is developed using [uv](https://docs.astral.sh/uv/).

After you've installed uv, you can create a venv using:

```sh
uv venv
```

Download all dependencies, including dev-dependencies:

```sh
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
