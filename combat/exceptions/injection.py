# File: combat/exceptions/prompt_injection.py

from typing import List, Optional


class InjectionException(Exception):
    """
    Raised when prompt injection is detected in input and blocking is enabled.

    Attributes:
        message (str): Human-readable explanation of why blocking occurred.
        has_violation (bool): True if prompt injection was detected in the provided text.
        violations (List[str]): List of violation types that were detected.
        risk_score (float): Risk score from the prompt injection scanner (0.0 to 1.0).
        is_blocked (bool): True if blocking is enabled and prompt injection was detected.
    """

    def __init__(
        self,
        message: str = "Input blocked due to detected injection.",
        has_violation: bool = True,
        violations: Optional[List[str]] = None,
        is_blocked: bool = True,
    ) -> None:
        # Always pass the message to the base Exception constructor
        super().__init__(message)

        # Store structured attributes
        self.has_violation: bool = has_violation
        self.violations: List[str] = violations or []
        self.is_blocked: bool = is_blocked
