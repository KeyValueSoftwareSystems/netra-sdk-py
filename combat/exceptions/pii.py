# File: combat/exceptions/pii.py

from typing import Any, Dict, List, Optional, Union


class PIIBlockedException(Exception):
    """
    Raised when PII is detected in input and blocking is enabled.

    Attributes:
        message (str): Human-readable explanation of why blocking occurred.
        has_pii (bool): True if PII was detected in the provided text.
        pii_entities (Dict[str, int]): Mapping from PII label to number of occurrences.
        masked_text (Union[str, List[Dict[str, str]], List[Any], None]): Input text after masking PII spans.
            Can be a string for simple inputs, a list of dicts for chat messages,
            or a list of BaseMessage objects for LangChain inputs.
        blocked (bool): True if blocking is enabled and PII was detected.
        pii_actions (Dict[str, List[str]]): Dictionary mapping action types to lists of PII entities.
    """

    def __init__(
        self,
        message: str = "Input blocked due to detected PII.",
        has_pii: bool = True,
        pii_entities: Optional[Dict[str, int]] = None,
        masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = None,
        blocked: bool = True,
        pii_actions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Always pass the message to the base Exception constructor
        super().__init__(message)

        # Store structured attributes
        self.has_pii: bool = has_pii
        self.pii_entities: Dict[str, int] = pii_entities or {}
        self.masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = masked_text
        self.blocked: bool = blocked
        self.pii_actions: Dict[str, List[str]] = pii_actions or {}
