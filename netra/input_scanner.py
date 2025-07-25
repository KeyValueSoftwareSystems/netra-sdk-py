"""
Input Scanner module for Netra SDK to implement LLM guard scanning options.

This module provides a unified interface for scanning input prompts using
various scanner implementations.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from netra import Netra
from netra.exceptions import InjectionException
from netra.scanner import Scanner

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """
    Result of running input scanning on prompts.

    Attributes:
        has_violation: True if any violations were detected
        violations: List of violation types that were detected
        is_blocked: True if the input should be blocked
        violation_actions: Dictionary mapping action types to lists of violations
    """

    has_violation: bool = False
    violations: List[str] = field(default_factory=list)
    is_blocked: bool = False
    violation_actions: Dict[str, List[str]] = field(default_factory=dict)


class ScannerType(Enum):
    """
    Enum representing the available scanner types.
    """

    PROMPT_INJECTION = "prompt_injection"


class InputScanner:
    """
    A factory class for creating input scanners.
    """

    def __init__(
        self,
        scanner_types: List[Union[str, ScannerType]] = [ScannerType.PROMPT_INJECTION],
        model_configuration: Optional[Dict[str, Any]] = None,
    ):
        self.scanner_types = scanner_types
        self.model_configuration = model_configuration

    @staticmethod
    def _get_scanner(scanner_type: Union[str, ScannerType], **kwargs: Any) -> Scanner:
        """
        Factory function to get a scanner instance based on the specified type.

        Args:
            scanner_type: The type of scanner to create (e.g., "prompt_injection" or ScannerType.PROMPT_INJECTION)
            **kwargs: Additional parameters to pass to the scanner constructor

        Returns:
            Scanner: An instance of the appropriate scanner

        Raises:
            ValueError: If the specified scanner type is not supported
        """
        if isinstance(scanner_type, ScannerType):
            scanner_type = scanner_type.value

        if scanner_type == ScannerType.PROMPT_INJECTION.value:
            match_type = None
            try:
                # Try to import from llm_guard if available
                from llm_guard.input_scanners.prompt_injection import MatchType

                match_type = kwargs.get("match_type", MatchType.FULL)
            except ImportError:
                logger.warning(
                    "llm-guard package is not installed. Using default match type. "
                    "To enable full functionality, install with: pip install 'netra-sdk[llm_guard]'"
                )

            from netra.scanner import PromptInjection

            threshold_value = kwargs.get("threshold", 0.5)
            if not isinstance(threshold_value, (int, float)):
                logger.info(f"Invalid threshold value: {threshold_value}")
                threshold = 0.5
            else:
                threshold = float(threshold_value)

            # Extract model configuration if provided
            model_configuration = kwargs.get("model_configuration")

            return PromptInjection(threshold=threshold, match_type=match_type, model_configuration=model_configuration)
        else:
            raise ValueError(f"Unsupported scanner type: {scanner_type}")

    def scan(self, prompt: str, is_blocked: bool = False) -> ScanResult:
        violations_detected = []
        for scanner_type in self.scanner_types:
            try:
                scanner = self._get_scanner(scanner_type, model_configuration=self.model_configuration)
                scanner.scan(prompt)
            except ValueError as e:
                raise ValueError(f"Invalid value type: {e}")
            except InjectionException as error:
                violations_detected.append(error.violations[0])

        # Create dynamic violation actions mapping based on detected violations and blocking status
        violations_actions = {}
        if violations_detected:
            if is_blocked:
                violations_actions["BLOCK"] = violations_detected
            else:
                violations_actions["FLAG"] = violations_detected

            Netra.set_custom_event(
                event_name="violation_detected",
                attributes={
                    "has_violation": True,
                    "violations": violations_detected,
                    "is_blocked": is_blocked,
                    "violation_actions": json.dumps(violations_actions),
                },
            )

        if is_blocked and violations_detected:
            raise InjectionException(
                message=f"Input blocked: detected {', '.join(violations_detected)}.",
                has_violation=True,
                violations=violations_detected,
                is_blocked=True,
                violation_actions=violations_actions,
            )

        return ScanResult(
            has_violation=bool(violations_detected),
            violations=violations_detected,
            violation_actions=violations_actions,
            is_blocked=bool(is_blocked and violations_detected),
        )
