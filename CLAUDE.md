# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`netra-sdk` (PyPI name; import as `netra`) is a Python SDK for AI application observability. It wraps OpenTelemetry to trace LLM calls, vector DB operations, HTTP clients, and more, and layers on session tracking, custom spans/metrics, and HTTP clients for the Netra backend (dashboards, evaluations, prompts, datasets, usage, simulations).

Python `>=3.10,<3.14`. The shipped package is `netra`.

## Commands

Dependency and env management is **Poetry** (`poetry.lock` is the source of truth; `uv.lock` is also present).

```bash
./setup_dev.sh                       # install with [dev,test] extras + pre-commit hooks (also installs commit-msg & pre-push hooks)

poetry run pytest                    # full suite
poetry run pytest tests/test_netra_init.py            # single file
poetry run pytest tests/test_netra_init.py::TestName::test_case   # single test
poetry run pytest -m unit            # by marker: unit | integration | thread_safety | slow
poetry run pytest --cov=netra --cov-report=html       # coverage -> htmlcov/

pre-commit run --all-files           # black, isort, autoflake, mypy (strict), yaml/whitespace hooks
poetry run mypy netra                # type-check (strict mode; tests/examples excluded)
```

Line length is **120** (black, isort profile=black, flake8). mypy runs in `strict` mode over `netra/` only. Commits must follow **Conventional Commits** (enforced by commitizen via the commit-msg hook); versioning is PEP 621 in `pyproject.toml` and bumped by commitizen (`tag_format = v{version}`).

## Architecture

### Initialization and configuration
- `Netra` (`netra/__init__.py`) is the single public entry point — a thread-safe singleton configured once via `Netra.init(...)` (guarded by an `RLock`; repeat calls are ignored). It wires the `Tracer`, instrumentations, metrics, session manager, and the backend API clients.
- `Config` (`netra/config.py`) resolves every setting with this precedence: **code params passed to `init()` → `NETRA_*` env vars → standard `OTEL_*` env vars → SDK defaults**. When changing config behavior, preserve this ordering.

### Tracing pipeline
- `Tracer` (`netra/tracer.py`) builds the OTel `TracerProvider`, chooses the exporter (`TrialAwareOTLPExporter`/`FilteringSpanExporter` in `netra/exporters/`, falling back to Console), and installs a chain of custom span processors from `netra/processors/`. Each processor has a narrow job — e.g. `SpanIOProcessor` normalizes `input`/`output` and remaps `traceloop.*` attributes, `RootInstrumentFilterProcessor` drops root spans (and their whole subtree) from instrumentations not allowed to produce root spans, `ScrubbingSpanProcessor` handles scrubbing, `SessionSpanProcessor` stamps session/user/tenant context.
- Root vs. non-root instruments are configured **independently**: `instruments` controls non-root spans, `root_instruments` controls which instrumentations may emit root-level spans. Both default to the `DEFAULT_INSTRUMENTS` / `DEFAULT_INSTRUMENTS_FOR_ROOT` sets in `netra/instrumentation/instruments.py`.

### Instrumentation
- Built on top of **traceloop-sdk**. Each `InstrumentSet` member in `netra/instrumentation/instruments.py` is tagged `_Origin.TRACELOOP` or `_Origin.CUSTOM`; `partition_by_origin` (`netra/instrumentation/wiring/selection.py`) routes `CUSTOM` members to Netra's own registry and never delegates them to traceloop, so the same call sites are never patched twice. Netra ships its own implementations for several providers (subdirs under `netra/instrumentation/libraries/`, e.g. `openai/`, `cohere/`, `weaviate/`, `claude_agent_sdk/`). When adding/overriding an instrumentation, add the custom impl, tag its member `_Origin.CUSTOM` in `instruments.py`, and add it to `CUSTOM_INSTRUMENTORS` (`wiring/registry.py`).

### Decorators
- `netra/decorators.py` exposes `@workflow`, `@agent`, `@task`, `@span`. They wrap sync + async functions, sync/async generators, and streaming responses (see `_wrap_*` helpers), and can decorate a class to instrument all its (non-internal) methods.

### Custom spans
- `SpanWrapper` + `UsageModel` / `ActionModel` / `SpanType` (`netra/span_wrapper.py`) back `Netra.start_span(...)` for manually tracing external calls with usage, actions, and events.

### Backend API clients
- `dashboard/`, `evaluation/`, `prompts/`, `models/`, `usage/`, `simulation/` each follow the same shape: `api.py` (public class, constructed with `Config`, exposed as a `Netra` attribute), `client.py` (HTTP client), and usually `models.py` (typed request/response models). Reuse this `api` + `client` + `models` split when adding a new backend feature.

## Testing notes
- `pytest.ini`: `testpaths=tests`, `--strict-markers` + `--strict-config` are on, so any new marker must be declared there. Markers: `unit`, `integration`, `thread_safety`, `slow`. Per-test timeout is 300s. `DeprecationWarning`/`PendingDeprecationWarning`/pydantic `UserWarning` are filtered out.
- Instrumentation tests (`tests/test_*_instrumentation.py`) verify Netra's custom provider instrumentations; thread-safety tests exist because `Netra`/`Config` are process-global singletons.
