# Development Guide

## Prerequisites

[uv](https://docs.astral.sh/uv/) is required. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Clone the repository and install all dependencies (runtime + dev):

```bash
git clone https://github.com/packmad/StatisticsForScientists.git
cd StatisticsForScientists
uv sync
```

Install the git hooks so linting and type-checking run automatically:

```bash
uv run pre-commit install --hook-type pre-commit
uv run pre-commit install --hook-type pre-push
```

To run all checks manually:

```bash
uv run pre-commit run --all-files                        # commit-stage
uv run pre-commit run --all-files --hook-stage push 
```

## Running the tests

```bash
uv run pytest
```

## Releasing a new version

This project follows [Semantic Versioning](https://packaging.python.org/en/latest/discussions/versioning/) (`major.minor.patch`):

| Part | When to increment |
|---|---|
| `major` | incompatible API changes |
| `minor` | new functionality, backwards-compatible |
| `patch` | backwards-compatible bug fixes |

Bump the version, commit, and tag in one command:

```bash
uv run bump-my-version bump patch   # 1.0.0 -> 1.0.1
uv run bump-my-version bump minor   # 1.0.0 -> 1.1.0
uv run bump-my-version bump major   # 1.0.0 -> 2.0.0
```

This updates `pyproject.toml` and `stats4science/version.py`, runs `uv lock` to sync the lockfile, commits all three files, and creates a `v{version}` git tag. Push the tag to trigger the publish CI:

```bash
git push origin main --tags
```
