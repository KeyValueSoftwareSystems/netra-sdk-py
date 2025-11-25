import logging
import threading
import time
from typing import Optional, Set

logger = logging.getLogger(__name__)

_trial_status_lock = threading.Lock()
_trial_blocked_at: Optional[float] = None
_blocked_trace_ids: Set[str] = set()

TRIAL_BLOCK_DURATION_SECONDS = 15 * 60 


def set_trial_blocked(blocked: bool) -> None:
    """Set the trial blocked status with automatic expiration after 15 minutes.

    When called with blocked=True, starts a timer. All span exports will be blocked
    for 15 minutes. After 15 minutes, exports automatically resume even if this
    function is not called again.

    Args:
        blocked: True to start the 15-minute blocking period, False to reset
    """
    global _trial_blocked_at
    with _trial_status_lock:
        if blocked:
            if _trial_blocked_at is None:
                # Start a new 15-minute blocking period
                _trial_blocked_at = time.time()
                logger.warning(
                    "Trial/quota exhausted: blocking span export for %d seconds (15 minutes)",
                    TRIAL_BLOCK_DURATION_SECONDS
                )
            else:
                elapsed = time.time() - _trial_blocked_at
                remaining = TRIAL_BLOCK_DURATION_SECONDS - elapsed
                logger.debug(
                    "Trial already blocked: %d seconds remaining",
                    max(0, int(remaining))
                )
        else:
            if _trial_blocked_at is not None:
                logger.info("Trial blocking manually reset")
            _trial_blocked_at = None


def is_trial_blocked() -> bool:
    """Check if trial is currently blocked.

    Automatically returns False after 15 minutes have passed, even if
    set_trial_blocked(True) was never called again.

    Returns:
        True if currently within the 15-minute blocking period, False otherwise
    """
    global _trial_blocked_at
    
    with _trial_status_lock:
        if _trial_blocked_at is None:
            return False
        
        # Check if 15 minutes have passed since blocking started
        elapsed = time.time() - _trial_blocked_at
        if elapsed >= TRIAL_BLOCK_DURATION_SECONDS:
            _trial_blocked_at = None
            logger.info("Trial blocking period (15 minutes) expired, resuming exports")
            return False
        
        return True


def add_blocked_trace_id(trace_id: str) -> None:
    """Add a trace ID to the blocked list.

    Trace IDs that started during the blocking period should be added to this list.
    All spans from these trace IDs will be filtered out, even after the 15-minute
    block expires. Only new trace IDs created after block expiration will be exported.

    Args:
        trace_id: The trace ID to block (format: hex string)
    """
    with _trial_status_lock:
        _blocked_trace_ids.add(trace_id)
        logger.debug("Added trace ID to blocked list: %s (total blocked: %d)", trace_id, len(_blocked_trace_ids))


def is_trace_id_blocked(trace_id: str) -> bool:
    """Check if a trace ID is in the blocked list.

    Args:
        trace_id: The trace ID to check

    Returns:
        True if this trace ID should be filtered, False otherwise
    """
    with _trial_status_lock:
        return trace_id in _blocked_trace_ids
