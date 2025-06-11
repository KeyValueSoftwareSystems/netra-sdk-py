# File: promptops_sdk/exceptions.py

from typing import Any, Dict, List, Optional, Union


class PIIBlockedException(Exception):
    """
    Raised when PII is detected in input and blocking is enabled.

    Attributes:
        message (str): Human-readable explanation of why blocking occurred.
        has_pii (bool): True if PII was detected in the provided text.
        entity_counts (Dict[str, int]): Mapping from PII label to number of occurrences.
        masked_text (Union[str, List[Dict[str, str]], List[Any], None]): Input text after masking PII spans.
            Can be a string for simple inputs, a list of dicts for chat messages,
            or a list of BaseMessage objects for LangChain inputs.
        blocked (bool): True if blocking is enabled and PII was detected.
    """

    def __init__(
        self,
        message: str = "Input blocked due to detected PII.",
        has_pii: bool = True,
        entity_counts: Optional[Dict[str, int]] = None,
        masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = None,
        blocked: bool = True,
    ) -> None:
        # Always pass the message to the base Exception constructor
        super().__init__(message)

        # Store structured attributes
        self.has_pii: bool = has_pii
        self.entity_counts: Dict[str, int] = entity_counts or {}
        self.masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = masked_text
        self.blocked: bool = blocked
