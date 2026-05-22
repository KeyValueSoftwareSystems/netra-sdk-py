"""Shared constants for the evaluation module."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PREFIX = "netra.evaluation"

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MIN_CONCURRENCY = 1
DEFAULT_CONCURRENCY = 5

# ---------------------------------------------------------------------------
# Span / tracing
# ---------------------------------------------------------------------------
SPAN_NAME_PREFIX = "TestRun"

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
URL_CREATE_DATASET = "/evaluations/dataset"
URL_DATASET_ITEMS = "/evaluations/dataset/{dataset_id}/items"
URL_GET_DATASET = "/evaluations/dataset/{dataset_id}"
URL_CREATE_RUN = "/evaluations/test_run"
URL_RUN_ITEM = "/evaluations/run/{run_id}/item"
URL_LOCAL_EVALUATIONS = "/evaluations/run/{run_id}/item/{test_run_item_id}/local-evaluations"
URL_RUN_STATUS = "/evaluations/run/{run_id}/status"
URL_GET_RUN = "/evaluations/run/{run_id}"
URL_SPAN = "sdk/traces/spans/{span_id}"
TELEMETRY_SUFFIX = "/telemetry"

# ---------------------------------------------------------------------------
# HTTP client timeouts
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 10.0
ENV_TIMEOUT = "NETRA_EVALUATION_TIMEOUT"
