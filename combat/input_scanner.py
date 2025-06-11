"""
Input Scanner module for Combat SDK to implement LLM guard scanning options.

This module provides a unified interface for scanning input prompts using
various scanner implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum

from combat.scanner import Scanner
from combat import Combat
from combat.exceptions import InjectionException


@dataclass
class ScanResult:
    """
    Result of running input scanning on prompts.

    Attributes:
        has_violation: True if any violations were detected
        violations: List of violation types that were detected
        is_blocked: True if the input should be blocked
    """
    has_violation: bool = False
    violations: List[str] = field(default_factory=list)
    is_blocked: bool = False


class ScannerType(Enum):
    """
    Enum representing the available scanner types.
    """
    PROMPT_INJECTION = "prompt_injection"


def get_scanner(scanner_type: Union[str, ScannerType], **kwargs) -> Scanner:
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
        from combat.scanner import PromptInjection
        from llm_guard.input_scanners.prompt_injection import MatchType

        match_type = kwargs.get("match_type", MatchType.FULL)
        threshold = kwargs.get("threshold", 0.5)

        return PromptInjection(threshold=threshold, match_type=match_type)
    else:
        raise ValueError(f"Unsupported scanner type: {scanner_type}")


def scan(prompt: str, types: List[Union[str, ScannerType]], is_blocked: bool = False) -> ScanResult:
    """
    Scan the input prompt for potential issues based on specified types of scans.

    Args:
        prompt: The input prompt to scan
        types: A list of scan types to perform (e.g., ["prompt_injection"] or [ScannerType.PROMPT_INJECTION])
        is_blocked: If True, raises PromptInjectionException when violations are detected

    Returns:
        ScanResult: An object containing:
            - has_violation: True if any violations were detected
            - violations: List of violation types that were detected
            - is_blocked: True if the input should be blocked

    Raises:
        PromptInjectionBlockedException: If violations are detected and is_blocked is True
    """
    violations_detected = []
    for scanner_type in types:
        try:

            scanner = get_scanner(scanner_type)
            scanner.scan(prompt)

        except ValueError as e:
            raise ValueError(f"Invalid value type: {e}")

        except InjectionException as error:
            violations_detected.append(error.violations[0])

    if violations_detected:
        Combat.set_custom_event(event_name="violation_detected", attributes={
            "has_violation": True, "violations": violations_detected, "is_blocked": is_blocked})

    if is_blocked and violations_detected:
        raise InjectionException(
            message=f"Input blocked: detected {', '.join(violations_detected)}.",
            has_violation=True,
            violations=violations_detected,
            is_blocked=True
        )

    return ScanResult(
        has_violation=bool(violations_detected),
        violations=violations_detected,
        is_blocked=is_blocked
    )
