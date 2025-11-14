from __future__ import annotations

import logging

_LOG_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def configure_package_logging(debug_mode: bool) -> None:
    """Configure logging for the netra package and related dependencies."""

    pkg_logger = logging.getLogger("netra")
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    otel_trace_logger = logging.getLogger("opentelemetry.trace")
    otel_instr_logger = logging.getLogger("opentelemetry.instrumentation")
    otel_instr_instrumentor_logger = logging.getLogger("opentelemetry.instrumentation.instrumentor")

    # Always clear Netra handlers so we can reconfigure cleanly
    pkg_logger.handlers.clear()

    if debug_mode:
        # Netra: verbose
        pkg_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(_LOG_FORMATTER)
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False

        # HTTP clients: visible but not too noisy
        httpx_logger.setLevel(logging.INFO)
        httpcore_logger.setLevel(logging.INFO)
        httpx_logger.propagate = True
        httpcore_logger.propagate = True

        # OpenTelemetry: allow warnings and above
        otel_trace_logger.setLevel(logging.WARNING)
        otel_instr_logger.setLevel(logging.WARNING)
        otel_instr_instrumentor_logger.setLevel(logging.WARNING)
        otel_trace_logger.propagate = True
        otel_instr_logger.propagate = True
        otel_instr_instrumentor_logger.propagate = True
    else:
        # Netra: completely silent unless user opts in
        pkg_logger.setLevel(logging.CRITICAL)
        pkg_logger.addHandler(logging.NullHandler())
        pkg_logger.propagate = False

        # HTTP clients: keep their own warnings local, do not bubble via Netra
        httpx_logger.setLevel(logging.WARNING)
        httpcore_logger.setLevel(logging.WARNING)
        httpx_logger.propagate = False
        httpcore_logger.propagate = False

        # OpenTelemetry: suppress logs triggered by Netra usage
        otel_trace_logger.setLevel(logging.ERROR)
        otel_instr_logger.setLevel(logging.ERROR)
        otel_instr_instrumentor_logger.setLevel(logging.ERROR)
        otel_trace_logger.propagate = False
        otel_instr_logger.propagate = False
        otel_instr_instrumentor_logger.propagate = False
