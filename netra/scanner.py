"""
Scanner module for Netra SDK to implement various scanning capabilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from netra.exceptions import InjectionException

logger = logging.getLogger(__name__)


class Scanner(ABC):
    """
    Abstract base class for scanner implementations.

    Scanners can analyze and process input prompts for various purposes
    such as security checks, content moderation, etc.
    """

    @abstractmethod
    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        """
        Scan the input prompt and return the sanitized prompt, validity flag, and risk score.

        Args:
            prompt: The input prompt to scan

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

    def __init__(self, threshold: float = 0.5, match_type: Optional[str] = None):
        """
        Initialize the PromptInjection scanner.

        Args:
            threshold: The threshold value (between 0.0 and 1.0) above which a prompt is considered risky
            match_type: The type of matching to use
                (from llm_guard.input_scanners.prompt_injection.MatchType)
        """
        self.threshold = threshold
        self.scanner = None
        self.llm_guard_available = False

        try:
            from llm_guard.input_scanners import PromptInjection as LLMGuardPromptInjection
            from llm_guard.input_scanners.prompt_injection import MatchType

            if match_type is None:
                match_type = MatchType.FULL

            self.scanner = LLMGuardPromptInjection(threshold=threshold, match_type=match_type)
            self.llm_guard_available = True
        except ImportError:
            logger.warning(
                "llm-guard package is not installed. Prompt injection scanning will be limited. "
                "To enable full functionality, install with: pip install 'netra-sdk[llm_guard]'"
            )

    def scan(self, prompt: str) -> Tuple[str, bool, float]:
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
        if not self.llm_guard_available or self.scanner is None:
            # Simple fallback when llm-guard is not available
            # Always pass validation but log a warning
            logger.warning(
                "Using fallback prompt injection detection (llm-guard not available). "
                "Install the llm_guard optional dependency for full protection."
            )
            return prompt, True, 0.0

        # Use llm_guard's scanner to check for prompt injection
        assert self.scanner is not None  # This helps mypy understand self.scanner is not None here
        sanitized_prompt, is_valid, risk_score = self.scanner.scan(prompt)
        if not is_valid:
            raise InjectionException(
                message="Input blocked: detected prompt injection",
                has_violation=True,
                violations=["prompt_injection"],
            )
        return sanitized_prompt, is_valid, risk_score
