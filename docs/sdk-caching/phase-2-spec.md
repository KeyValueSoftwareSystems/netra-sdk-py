---
name: SDK Caching Phase 2 — netra-sdk-py
status: ready-to-implement
repo: netra-sdk-py
phase: 2
derived_from: docs/superpowers/specs/2026-07-10-sdk-caching-phase-2-design.md
---

# Phase 2 Spec — `netra-sdk-py` (Models caching)

> **Implement from this document** for the Python SDK PR.  
> **Design:** [Phase 2 design](../../superpowers/specs/2026-07-10-sdk-caching-phase-2-design.md)  
> **Shared Phase 1 contract:** [sdk-caching spec](./spec.md)  
> **Sibling:** [JS Phase 2 spec](./phase-2-js.md)

## Goal

Add opt-in in-memory caching to the **existing** `Netra.models.get_model_pricing()` API, matching Phase 1 prompts caching and the JS Phase 2 behavior.

This PR does **not** add a new Models HTTP surface — `netra/models/` already calls `GET /sdk/models`.

## Non-negotiable conventions (match existing code)

| Convention | Follow |
|---|---|
| Module layout | Keep `netra/models/{api,client,__init__}.py` — extend `api.py` only for cache wiring |
| HTTP | Leave `ModelsHttpClient` behavior intact unless a bug blocks correct unwrap |
| Logging | `logging.getLogger(__name__)` / existing `netra.models` messages |
| Naming | snake_case: `use_cache`, `cache_ttl` |
| Cache | Reuse `netra.cache.TTLCache` (thread-safe). Do **not** add a second cache type |
| Init | `Models` already constructed in `Netra.init()` — no new init flags |
| Shutdown | Best-effort `clear_cache()` next to prompts in `Netra.shutdown()` |
| Tests | `pytest` + `unittest.mock`; mirror `tests/test_prompts_cache.py` |
| Style | Type hints, docstrings on public methods; conventional commits per `CONTRIBUTING.md` |

## Current behavior (preserve)

```python
# netra/models/api.py today
def get_model_pricing(self, name: Optional[str] = None) -> List[Any] | Any:
    result = self._client.get_model_pricing(name=name)
    # unwraps ApiResponse { data: [...] } → list
    ...
```

Client returns `{}` on failure; API may return `[]` on bad shape, or the raw non-dict result.

**Backward compatibility:** Existing callers with only `name` (or no args) must behave identically — always HTTP, no cache.

## Public API (after change)

```python
MODEL_PRICING_CACHE_TTL_SECONDS = 300  # module-level constant in api.py (or models/constants)

def get_model_pricing(
    self,
    name: Optional[str] = None,
    use_cache: bool = False,
    cache_ttl: Optional[int] = None,
) -> Any:
    ...

def clear_cache(self) -> None:
    """Clear all cached model pricing entries."""
```

| Param | Default | Meaning |
|---|---|---|
| `name` | `None` | Optional filter; cache key uses `"all"` when omitted |
| `use_cache` | `False` | Opt-in read/write cache |
| `cache_ttl` | `None` | Per-call TTL seconds; when `None` and caching, use **300** (Models-owned), **not** `Config.cache_ttl_seconds` |

### Behavior matrix

| `use_cache` | `cache_ttl` | Behavior |
|---|---|---|
| `False` | any | Always HTTP. No cache read/write. |
| `True` | `None` | Read cache → miss → fetch → store with **300s** |
| `True` | `N` | Read cache → miss → fetch → store with `N` seconds (`TTLCache` already skips write when `ttl <= 0`) |

### Cache key

```text
model:pricing:{name or "all"}
```

### What to cache / not cache

| Result | Cache? |
|---|---|
| Non-empty `list` | Yes |
| Empty `list` `[]` | Yes (successful empty result) |
| `None` | No |
| `{}` from client failure | No (treat as failure — same spirit as prompts) |
| Non-list unexpected value after unwrap | No (keep current error logging / return path; do not store) |

Match prompts’ guard style:

```python
# prompts today
if use_cache and result is not None and result != {}:
    self._cache.set(...)
```

For models, prefer:

```python
if use_cache and isinstance(result, list):
    self._cache.set(cache_key, result, cache_ttl)
```

so only successful list payloads are stored (including `[]`).

## Files to change

| File | Change |
|---|---|
| `netra/models/api.py` | Instantiate `TTLCache(default_ttl=MODEL_PRICING_CACHE_TTL_SECONDS)`; wire `get_model_pricing`; add `clear_cache()` |
| `netra/__init__.py` | In `shutdown()`, clear `models` cache when present (alongside prompts) |
| `tests/test_models_cache.py` | **New** — mirror prompts cache tests for models |
| `tests/test_netra_init.py` | Extend only if shutdown/init assertions need models cache coverage |

### Do not change (unless required for correctness)

- `netra/models/client.py` HTTP path / timeout (`NETRA_MODELS_TIMEOUT`, default 10s)
- `netra/cache.py` API
- Prompts default TTL / `cache_ttl_seconds` semantics
- JS SDK (separate PR / repo)

## Implementation sketch

```python
from netra.cache import TTLCache

MODEL_PRICING_CACHE_TTL_SECONDS = 300


class Models:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = ModelsHttpClient(config)
        self._cache: TTLCache[Any] = TTLCache(default_ttl=MODEL_PRICING_CACHE_TTL_SECONDS)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_model_pricing(
        self,
        name: Optional[str] = None,
        use_cache: bool = False,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        cache_key = f"model:pricing:{name or 'all'}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = self._client.get_model_pricing(name=name)

        # Existing unwrap logic (keep as-is)...
        items = ...  # list or failure sentinel

        if use_cache and isinstance(items, list):
            self._cache.set(cache_key, items, cache_ttl)

        return items
```

**Important:** Construct cache with `MODEL_PRICING_CACHE_TTL_SECONDS`, not `cfg.cache_ttl_seconds`.

### Shutdown

```python
if hasattr(cls, "models") and cls.models is not None:
    try:
        cls.models.clear_cache()
    except Exception:
        pass
```

Place next to the existing prompts `clear_cache()` block.

## Acceptance criteria

### Functional

- [ ] Callers without `use_cache` unchanged (HTTP every time)
- [ ] `use_cache=True` second call with same `name` skips HTTP
- [ ] `name=None` / omitted uses key `model:pricing:all`
- [ ] Different names → different keys
- [ ] `None` / `{}` / non-list failures not cached
- [ ] Empty list `[]` may be cached
- [ ] Default TTL 300s when `cache_ttl` omitted
- [ ] Per-call `cache_ttl` overrides; `cache_ttl=0` does not retain entry (`TTLCache` behavior)
- [ ] `clear_cache()` and `Netra.shutdown()` force refetch on next cached call
- [ ] Prompts still use `cache_ttl_seconds` (no regression)

### Tests (minimum — mirror `test_prompts_cache.py`)

- [ ] `use_cache` omitted → HTTP every time
- [ ] `use_cache=True` → second call skips HTTP
- [ ] Different `name` → separate entries
- [ ] Failure (`{}` / `None`) not cached
- [ ] `use_cache=False` with `cache_ttl` ignores cache
- [ ] Per-call TTL expiry via mocked `time.monotonic`
- [ ] `clear_cache` forces HTTP
- [ ] Shutdown clears models cache (init/shutdown fixture pattern from prompts tests)

### Clean code / reuse checklist

- [ ] Reuse `TTLCache` only — no new cache class
- [ ] Named constant for 300s TTL
- [ ] Public `clear_cache()` — no private `_cache` access from `Netra.shutdown()`
- [ ] Preserve existing response unwrap logic; don’t rewrite client unless broken
- [ ] Docstrings updated for new params

## Usage example

```python
Netra.init()

Netra.models.get_model_pricing()
Netra.models.get_model_pricing("gpt-4o")

Netra.models.get_model_pricing(use_cache=True)
Netra.models.get_model_pricing("gpt-4o", use_cache=True, cache_ttl=60)
```

## Out of scope

- Backend Redis / invalidation
- Changing Models HTTP client URL or auth
- Adding JS Models module (see sibling spec)
- New init config keys for models TTL
