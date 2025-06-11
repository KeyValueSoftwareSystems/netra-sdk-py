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

    Returns:
        ScanResult: An object containing:
            - has_violation: True if any violations were detected
            - violations: List of violation types that were detected
            - is_blocked: True if the input should be blocked
    """
    scan_result = ScanResult()

    for scanner_type in types:
        try:
            # Get the appropriate scanner for this type
            scanner = get_scanner(scanner_type)

            # Perform the scan
            sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)

            # Update the result if there's a violation
            if not is_valid:
                scan_result.has_violation = True

                # Convert ScannerType enum to string if needed
                violation_type = scanner_type.value if isinstance(scanner_type, ScannerType) else scanner_type
                scan_result.violations.append(violation_type)

            scan_result.is_blocked = is_blocked
        except ValueError as e:
            continue
    if scan_result.has_violation:
        Combat.set_custom_event(event_name="violation_detected", attributes={"has_violation": scan_result.has_violation, "violations": scan_result.violations, "is_blocked": scan_result.is_blocked})

    return scan_result
