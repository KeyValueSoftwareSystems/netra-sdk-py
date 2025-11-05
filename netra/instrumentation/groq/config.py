from typing import Any, Callable, Dict, Optional


class Config:
    exception_logger: Optional[Callable[[BaseException], None]] = None
    get_common_metrics_attributes: Callable[[], Dict[str, Any]] = lambda: {}
    use_legacy_attributes = True
