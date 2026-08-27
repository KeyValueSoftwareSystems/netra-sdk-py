from typing import Callable, Optional


class Config:
    exception_logger: Optional[Callable[[Exception], None]] = None
