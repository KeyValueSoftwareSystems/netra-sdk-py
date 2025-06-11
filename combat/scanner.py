"""
Scanner module for Combat SDK to implement various scanning capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from combat.exceptions import InjectionException


class Scanner(ABC):
    """
    Abstract base class for scanner implementations.

    Scanners can analyze and process input prompts for various purposes
    such as security checks, content moderation, etc.
    """

    @abstractmethod
    def scan(self, prompt: str, is_blocked: bool = False) -> Tuple[str, bool, float]:
        """
        Scan the input prompt and return the sanitized prompt, validity flag, and risk score.

        Args:
            prompt: The input prompt to scan
            is_blocked: If True, raises PromptInjectionBlockedException when violations are detected

        Returns:
            Tuple containing:
                - sanitized_prompt: The potentially modified prompt after scanning
                - is_valid: Boolean indicating if the prompt passed the scan
                - risk_score: A score between 0.0 and 1.0 indicating the risk level
        """


class PromptInjection(Scanner):
    """
    A scanner implementation that detects and handles prompt injection attempts.

    This scanner uses llm_guard's PromptInjection scanner under the hood.
    """

    def __init__(self, threshold: float = 0.5, match_type: Any = None):
        """
        Initialize the PromptInjection scanner.

        Args:
            threshold: The threshold value (between 0.0 and 1.0) above which a prompt is considered risky
            match_type: The type of matching to use (from llm_guard.input_scanners.prompt_injection.MatchType)
        """
        from llm_guard.input_scanners import \
            PromptInjection as LLMGuardPromptInjection
        from llm_guard.input_scanners.prompt_injection import MatchType

        self.threshold = threshold
        if match_type is None:
            match_type = MatchType.FULL

        self.scanner = LLMGuardPromptInjection(threshold=threshold, match_type=match_type)

    def scan(self, prompt: str, is_blocked: bool = False) -> Tuple[str, bool, float]:
        """
        Scan the input prompt for potential prompt injection attempts.

        Args:
            prompt: The input prompt to scan

        Returns:
            Tuple containing:
                - sanitized_prompt: The potentially modified prompt after scanning
                - is_valid: Boolean indicating if the prompt passed the scan
                - risk_score: A score between 0.0 and 1.0 indicating the risk level
        """
        sanitized_prompt, is_valid, risk_score = self.scanner.scan(prompt)
        if not is_valid:
            raise InjectionException(
                message=f"Input blocked: detected prompt injection",
                has_violation=True,
                violations=["prompt_injection"],
                is_blocked=is_blocked,
            )
        return sanitized_prompt, is_valid, risk_score
