"""Shared constants for the simulation module."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PREFIX = "netra.simulation"

# ---------------------------------------------------------------------------
# Span / tracing
# ---------------------------------------------------------------------------
SPAN_NAME = "Netra.Simulation.TestRun"

# ---------------------------------------------------------------------------
# Conversation limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_TURNS = 50

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
URL_CREATE_RUN = "/evaluations/test_run/multi-turn"
URL_INITIALIZE_RUN = "/evaluations/test_run/multi-turn/initialize"
URL_FIRST_TURN = "/evaluations/run/{run_id}/item/{run_item_id}/first-turn"
URL_AGENT_RESPONSE = "/evaluations/turn/agent-response"
URL_RUN_ITEM_STATUS = "/evaluations/run/{run_id}/item/{run_item_id}/status"
URL_RUN_STATUS = "/evaluations/run/{run_id}/status"
TELEMETRY_SUFFIX = "/telemetry"

# ---------------------------------------------------------------------------
# HTTP client timeouts
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 500.0
ENV_TIMEOUT = "NETRA_SIMULATION_TIMEOUT"

# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------
DEFAULT_FILE_DOWNLOAD_TIMEOUT = 30.0
ENV_FILE_DOWNLOAD_TIMEOUT = "NETRA_SIMULATION_FILE_DOWNLOAD_TIMEOUT"
MAX_FILE_DOWNLOAD_WORKERS = 8
