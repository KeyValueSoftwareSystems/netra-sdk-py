"""Shared constants for the redteam module."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PREFIX = "netra.redteam"

# ---------------------------------------------------------------------------
# Span / tracing
# ---------------------------------------------------------------------------
SPAN_NAME = "Netra.Redteam.Turn"

# ---------------------------------------------------------------------------
# Concurrency / payload limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_CONCURRENCY = 5
# Matches SubmitRedteamTurnDto's MaxLength(100000) on output/error exactly — a hard backstop
# against a pathological agent response (not a product-level content cap), so it never trims
# real content the backend would otherwise accept.
MAX_AGENT_RESPONSE_CHARS = 100000
# Matches SubmitRedteamTurnDto's `@Max(1000)` on turnIndex.
MAX_TURN_INDEX = 1000
RESULTS_PAGE_LIMIT = 200

# ---------------------------------------------------------------------------
# API endpoints (relative to the "redteam/sdk" base path)
# ---------------------------------------------------------------------------
URL_CREATE_RUN = "/redteam/sdk/runs"
URL_GET_PROMPTS = "/redteam/sdk/runs/{run_id}/prompts"
URL_SUBMIT_TURN = "/redteam/sdk/runs/{run_id}/turns"
URL_GET_PROGRESS = "/redteam/sdk/runs/{run_id}/progress"
URL_GET_RESULTS = "/redteam/sdk/runs/{run_id}/results"
URL_GET_RISK_SCORE = "/redteam/sdk/configs/{config_id}/risk-score"
URL_CANCEL_RUN = "/redteam/sdk/runs/{run_id}/cancel"
TELEMETRY_SUFFIX = "/telemetry"

# ---------------------------------------------------------------------------
# HTTP client timeout
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_S = 20.0
ENV_TIMEOUT = "NETRA_REDTEAM_TIMEOUT"

# ---------------------------------------------------------------------------
# Generation-gating poll (client re-POSTs createRun while status="generating")
# ---------------------------------------------------------------------------
DEFAULT_GENERATION_POLL_INTERVAL_S = 2.0
ENV_GENERATION_POLL_INTERVAL = "NETRA_REDTEAM_GENERATION_POLL_INTERVAL"

DEFAULT_GENERATION_TIMEOUT_S = 300.0
ENV_GENERATION_TIMEOUT = "NETRA_REDTEAM_GENERATION_TIMEOUT"
