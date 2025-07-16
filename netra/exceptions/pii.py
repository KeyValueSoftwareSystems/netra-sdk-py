# File: netra/exceptions/pii.py

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
        is_blocked (bool): True if blocking is enabled and PII was detected.
        pii_actions (Dict[str, List[str]]): Dictionary mapping action types to lists of PII entities.
        original_text (Union[str, List[Dict[str, str]], List[str], List[Any], None]): The original text used to call the detect() method.
            Can be a string, list of strings, list of dictionaries, or any other type.
        hashed_entities (Dict[str, str]): Dictionary mapping hashed entity values to their original values.
            Only populated when using Anonymizer for masking.
    """

    def __init__(
        self,
        message: str = "Input blocked due to detected PII.",
        has_pii: bool = True,
        pii_entities: Optional[Dict[str, int]] = None,
        masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = None,
        pii_actions: Optional[Dict[Any, List[str]]] = None,
        is_blocked: bool = True,
        original_text: Optional[Union[str, List[Dict[str, str]], List[str], List[Any]]] = None,
        hashed_entities: Optional[Dict[str, str]] = None,
    ) -> None:
        # Always pass the message to the base Exception constructor
        super().__init__(message)

        # Store structured attributes
        self.has_pii: bool = has_pii
        self.pii_entities: Dict[str, int] = pii_entities or {}
        self.masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = masked_text
        self.pii_actions: Dict[Any, List[str]] = pii_actions or {}
        self.is_blocked: bool = is_blocked
        self.original_text: Optional[Union[str, List[Dict[str, str]], List[str], List[Any]]] = original_text
        self.hashed_entities: Dict[str, str] = hashed_entities or {}
