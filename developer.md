# Developer Guide

## 1. Tech Stack
- Language: Python 3.13+
- Tooling: uv (env + deps), ruff (lint/format)
- Notebooks: marimo
- Core libraries (planned):
  - SQL parsing: sqlglot
  - Similarity: scikit-learn (TF-IDF, cosine)
  - BM25: rank-bm25
  - Clustering: scikit-learn (Agglomerative, DBSCAN, KMeans)
  - Embeddings (optional): sentence-transformers
- CLI: Typer
- AWS SDK: boto3
- Testing: pytest

## 2. Repository Layout (Proposed)
- `src/athena_optimizer/`
  - `ingestion/`
  - `normalization/`
  - `scoring/`
  - `similarity/`
  - `clustering/`
  - `recommendations/`
  - `reporting/`
  - `models/`
- `tests/`
- `examples/`
- `notebooks/`

## 3. Development Practices
- Keep modules small, composable, and unit-testable.
- Use interfaces/protocols for interchangeable strategies.
- Prefer pure functions where possible.
- Avoid hidden global state.
- Keep config in a single module and allow overrides via CLI or env.

## 4. Coding Standards
- Use type hints for public interfaces.
- Prefer dataclasses or pydantic models for data structures.
- Keep functions under ~50 lines unless justified.
- Avoid one-letter variable names except for indices.
- Use ASCII characters unless the file already uses Unicode.

## 5. Testing
- Write unit tests for each module and major heuristic.
- Add regression tests for known query patterns.
- Ensure deterministic clustering behavior by fixed random seeds.
- Run `ruff check .` and `ruff format .` before releases.
- Run `pytest` before releases.

## 6. Versioning and Releases
- Follow semantic versioning.
- Tag releases with changelog notes.

## 7. Contribution Workflow
- Create a feature branch per change.
- Keep PRs focused and small.
- Include tests for new functionality.
- Avoid committing generated files.

## 8. Local Development
- Setup:
  - `uv venv .venv`
  - `source .venv/bin/activate`
  - `uv pip install -e ".[dev]"`
- Run lint/format:
  - `ruff check .`
  - `ruff format .`
- Run tests:
  - `pytest`
- AWS auth (local):
  - `aws sso login --profile <name>`
- Notebooks:
  - `marimo edit`
  - `marimo run notebooks/<name>.py`

## 9. Code Review Checklist
- Does the change include tests?
- Are module boundaries respected?
- Is the behavior deterministic where required?
- Are recommendations explainable and traceable?
- Is documentation updated if behavior changes?
