# GitHub Configuration

This directory contains CI/CD workflows, actions, and scripts for the project.

## Quick Start

Install pre-commit hooks after cloning:

```bash
uv sync
uv run pre-commit install
```

### `uv` dev deps for this:

```toml
# reset of dependencies
    "pytest==8.4.2",
    "pytest-asyncio==1.2.0",
    "pyright==1.1.408",
    "ruff==0.8.4",
    "pre-commit>=4.5.1",
    "pytest-cov>=7.0.0",
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) for code quality checks. Hooks run automatically on `git commit`.

| Hook                        | Description                               |
| --------------------------- | ----------------------------------------- |
| `trailing-whitespace`       | Remove trailing whitespace                |
| `end-of-file-fixer`         | Ensure files end with newline             |
| `check-yaml` / `check-toml` | Validate config files                     |
| `check-added-large-files`   | Block files > 1MB                         |
| `detect-secrets`            | Prevent accidental secret commits         |
| `ruff`                      | Python linting with auto-fix              |
| `ruff-format`               | Python formatting                         |
| `pyright`                   | Type checking                             |
| `update-requirements`       | Sync requirements.txt from pyproject.toml |

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

## CI Workflow

The `python-ci.yml` workflow runs on pushes and PRs to `main` and `develop`:

- **Pyright** - Type checking
- **Ruff** - Linting
- **Pytest** - Tests with coverage
- **Pre-commit** - All hooks validation

## Directory Structure

```
.github/
  actions/
    check-linear-issue/    # Verify PRs link to Linear issues
  scripts/
    update-requirements.sh # Sync requirements.txt from pyproject.toml
  workflows/
    python-ci.yml          # Main CI pipeline
  .coveragerc              # Coverage configuration
  .secrets.baseline        # detect-secrets baseline
```

## Secrets Baseline

If `detect-secrets` flags a false positive, update the baseline:

```bash
uv run detect-secrets scan --baseline .github/.secrets.baseline
```
