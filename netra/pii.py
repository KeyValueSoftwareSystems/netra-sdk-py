# File: netra/pii.py
import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Pattern, Tuple, Union, cast

from netra import Netra
from netra.anonymizer import Anonymizer
from netra.exceptions import PIIBlockedException

EMAIL_PATTERN: Pattern[str] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN: Pattern[str] = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN: Pattern[str] = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_PATTERN: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

DEFAULT_PII_PATTERNS: Dict[str, Pattern[str]] = {
    "EMAIL": EMAIL_PATTERN,
    "PHONE": PHONE_PATTERN,
    "CREDIT_CARD": CREDIT_CARD_PATTERN,
    "SSN": SSN_PATTERN,
}

DEFAULT_ENTITIES: List[str] = [
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "NRP",
    "LOCATION",
    "PHONE_NUMBER",
    "MEDICAL_LICENSE",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
    "UK_NHS",
    "UK_NINO",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    "IN_PAN",
    "IN_AADHAAR",
    "IN_VEHICLE_REGISTRATION",
    "IN_VOTER",
    "IN_PASSPORT",
]


@dataclass(frozen=True)
class PIIDetectionResult:
    """
    Result of running PII detection on input text.
    Attributes:
        has_pii: True if any PII matches were found.
        pii_entities: Dictionary mapping PII label -> count of occurrences.
        masked_text: Input text with PII spans replaced/masked.
            Can be a string for simple inputs, a list of dicts for chat messages,
            or a list of BaseMessage objects for LangChain inputs.
        original_text: The original text used to call the detect() method.
            Can be a string, list of strings, list of dictionaries, or any other type.
        is_blocked: True if block_on_pii is enabled and has_pii is True.
        is_masked: True if any text was replaced to mask PII.
        pii_actions: Dictionary mapping action types to lists of PII entities.
        hashed_entities: Dictionary mapping hashed entity values to their original values.
    """

    has_pii: bool = False
    pii_entities: Dict[str, int] = field(default_factory=dict)
    masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = None
    original_text: Optional[Union[str, List[Dict[str, str]], List[str], List[Any]]] = None
    is_blocked: bool = False
    is_masked: bool = False
    pii_actions: Dict[Any, List[str]] = field(default_factory=dict)
    hashed_entities: Dict[str, str] = field(default_factory=dict)


class PIIDetector(ABC):
    """
    Abstract base for all PII detectors. Provides common iteration/
    aggregation logic, while requiring subclasses to implement _detect_single_message().
    """

    def __init__(self, action_type: Literal["BLOCK", "FLAG", "MASK"] = "FLAG") -> None:
        """
        Initialize the PII detector.

        Args:
            action_type: Action to take when PII is detected. Options are:
                - "BLOCK": Raise PIIBlockedException when PII is detected
                - "FLAG": Detect PII but don't block or mask
                - "MASK": Replace PII with mask tokens (default)
        """
        self._action_type: Literal["BLOCK", "FLAG", "MASK"] = action_type

    @abstractmethod
    def _detect_pii(self, text: str) -> Tuple[bool, Counter[str], str, Dict[str, str]]:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text, entities)
        """

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text before PII detection.

        Args:
            text: The input text to preprocess.

        Returns:
            Preprocessed text ready for PII detection.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Trim whitespace
        text = text.strip()

        return text

    def _mask_spans(self, text: str, spans: Dict[str, List[Tuple[int, int]]]) -> str:
        """
        Mask identified PII spans in the text.

        Args:
            text: The original text containing PII.
            spans: Dictionary mapping PII label to list of (start, end) spans.

        Returns:
            Text with PII spans replaced by mask tokens.
        """
        # Convert spans to a flat list of (start, end, label) tuples
        all_spans = []
        for label, span_list in spans.items():
            for start, end in span_list:
                all_spans.append((start, end, label))

        # Sort spans by start position (in reverse order to avoid index shifting)
        all_spans.sort(reverse=True)

        # Apply masking
        result = text
        for start, end, label in all_spans:
            mask = f"[{label}]"
            result = result[:start] + mask + result[end:]

        return result

    def detect(self, input_data: Union[str, List[Dict[str, str]], List[str], List[Any]]) -> PIIDetectionResult:
        """
        Public entry point. Accepts either:
        1. A single string
        2. A list of dictionaries with string values (e.g. chat messages)
        3. A list of strings
        4. A list of LangChain BaseMessage objects (detected by duck typing)

        Args:
            input_data: The input data to detect PII in

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        try:
            return self._process_input_data(input_data)
        except PIIBlockedException as e:
            return self._handle_pii_exception(e)

    def _process_input_data(
        self, input_data: Union[str, List[Dict[str, str]], List[str], List[Any]]
    ) -> PIIDetectionResult:
        """
        Process input data based on its type and route to appropriate detection method.

        Args:
            input_data: The input data to detect PII in

        Returns:
            PIIDetectionResult: The detection result containing PII information

        Raises:
            ValueError: If input type is not supported
        """
        if isinstance(input_data, str):
            return self._detect_single_message(input_data)

        if isinstance(input_data, list):
            return self._process_list_input(input_data)

        raise ValueError(f"Unsupported input type: {type(input_data).__name__}")

    def _process_list_input(self, input_list: List[Any]) -> PIIDetectionResult:
        """
        Process list input by determining the list type and routing to appropriate method.

        Args:
            input_list: List of items to process

        Returns:
            PIIDetectionResult: The detection result containing PII information

        Raises:
            ValueError: If list item type is not supported
        """
        if not input_list:
            return PIIDetectionResult(original_text=input_list)

        first_item = input_list[0]

        if isinstance(first_item, dict):
            return self._detect_chat_messages(cast(List[Dict[str, str]], input_list))

        if isinstance(first_item, str):
            return self._detect_string_list(cast(List[str], input_list))

        if self._is_langchain_message(first_item):
            return self._process_langchain_messages(input_list)

        raise ValueError(f"Unsupported input type in list: {type(first_item).__name__}")

    def _is_langchain_message(self, item: Any) -> bool:
        """
        Check if an item is a LangChain BaseMessage-like object using duck typing.

        Args:
            item: The item to check

        Returns:
            True if the item has the expected LangChain message attributes
        """
        return hasattr(item, "content") and hasattr(item, "type")

    def _process_langchain_messages(self, messages: List[Any]) -> PIIDetectionResult:
        """
        Process LangChain BaseMessage-like objects by extracting their content.

        Args:
            messages: List of LangChain BaseMessage-like objects

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        contents = [msg.content for msg in messages if hasattr(msg, "content")]
        return self._detect_string_list(contents)

    def _handle_pii_exception(self, exception: PIIBlockedException) -> PIIDetectionResult:
        """
        Handle PIIBlockedException based on the configured action type.

        Args:
            exception: The PIIBlockedException that was raised

        Returns:
            PIIDetectionResult: Appropriate result based on action type

        Raises:
            PIIBlockedException: Re-raised if action type is BLOCK
        """
        pii_actions = self._create_pii_actions(exception)
        attributes = self._build_trace_attributes(exception, pii_actions)

        # Log the PII detection event
        Netra.set_custom_event(event_name="pii_detected", attributes=attributes)

        # Handle different action types
        if self._action_type == "BLOCK":
            raise exception

        return self._create_detection_result(exception, pii_actions)

    def _create_pii_actions(self, exception: PIIBlockedException) -> Dict[str, List[str]]:
        """
        Create pii_actions dictionary based on action type and detected entities.

        Args:
            exception: The PIIBlockedException containing detected entities

        Returns:
            Dictionary mapping action type to list of PII entity types
        """
        return {self._action_type: list(exception.pii_entities.keys())}

    def _build_trace_attributes(
        self, exception: PIIBlockedException, pii_actions: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Build attributes dictionary for tracing/logging the PII detection event.

        Args:
            exception: The PIIBlockedException containing PII information
            pii_actions: Dictionary of PII actions to be taken

        Returns:
            Dictionary of attributes for the trace event
        """
        attributes = {
            "has_pii": exception.has_pii,
            "pii_entities": json.dumps(exception.pii_entities),
            "is_blocked": self._action_type == "BLOCK",
            "is_masked": self._action_type == "MASK",
            "pii_actions": json.dumps(pii_actions),
        }

        # Add masked_text to attributes only for MASK action type
        if self._action_type == "MASK":
            attributes["masked_text"] = self._serialize_masked_text(exception.masked_text)

        return attributes

    def _serialize_masked_text(self, masked_text: Any) -> str:
        """
        Serialize masked text to string format for tracing attributes.

        Args:
            masked_text: The masked text in various possible formats

        Returns:
            String representation of the masked text
        """
        if isinstance(masked_text, (dict, list)):
            return json.dumps(masked_text)
        return str(masked_text)

    def _create_detection_result(
        self, exception: PIIBlockedException, pii_actions: Dict[str, List[str]]
    ) -> PIIDetectionResult:
        """
        Create PIIDetectionResult based on action type and exception data.

        Args:
            exception: The PIIBlockedException containing PII information
            pii_actions: Dictionary of PII actions taken

        Returns:
            PIIDetectionResult with appropriate fields set based on action type
        """
        if self._action_type == "MASK":
            return PIIDetectionResult(
                has_pii=exception.has_pii,
                pii_entities=exception.pii_entities,
                original_text=exception.original_text,
                pii_actions=pii_actions,
                masked_text=exception.masked_text,
                is_blocked=False,
                is_masked=True,
                hashed_entities=exception.hashed_entities,
            )

        # For FLAG action type
        return PIIDetectionResult(
            has_pii=exception.has_pii,
            pii_entities=exception.pii_entities,
            original_text=exception.original_text,
            pii_actions=pii_actions,
            masked_text=None,
            is_blocked=False,
            is_masked=False,
            hashed_entities=exception.hashed_entities,
        )

    def _detect_single_message(self, text: str) -> PIIDetectionResult:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        has_pii, counts, masked_text, entities = self._detect_pii(text)

        if has_pii:
            # Create pii_actions based on the action type and detected entities
            pii_actions = {self._action_type: list(counts.keys())}
            raise PIIBlockedException(
                message="PII detected; blocking enabled.",
                has_pii=has_pii,
                pii_entities=dict(counts),
                masked_text=masked_text,
                pii_actions=pii_actions,
                is_blocked=True,
                original_text=text,
                hashed_entities=entities,
            )

        return PIIDetectionResult(
            has_pii=has_pii,
            pii_entities={},
            masked_text=None,  # No PII detected, so no masked text needed
            original_text=text,
            is_blocked=False,
            is_masked=False,
            pii_actions={},  # No PII detected, so no actions needed
            hashed_entities=entities,
        )

    def _detect_chat_messages(self, chat_messages: List[Dict[str, str]]) -> PIIDetectionResult:
        """
        Detect PII in a list of chat messages.

        Args:
            chat_messages: List of chat message dictionaries with 'role' and 'message' keys

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        overall_has_pii = False
        total_counts: Counter[str] = Counter()
        masked_list: List[Dict[str, str]] = []
        merged_hashed_entities: Dict[str, str] = {}

        for message in chat_messages:
            role = message.get("role", "unknown")
            text = message.get("content", "")

            try:
                self._detect_single_message(text)
                # If we get here, no PII was detected
                masked_list.append({"role": role, "content": text})
            except PIIBlockedException as e:
                # PII was detected
                overall_has_pii = True
                total_counts.update(e.pii_entities)
                # Merge hashed entities from this message
                merged_hashed_entities.update(e.hashed_entities)
                # Convert masked_text to string if it's not already to prevent type errors
                masked_text_str = str(e.masked_text) if e.masked_text is not None else ""
                masked_list.append({"role": role, "content": masked_text_str})

        if overall_has_pii:
            # Create pii_actions based on the action type and detected entities
            pii_actions = {self._action_type: list(total_counts.keys())}
            raise PIIBlockedException(
                message="PII detected in one or more messages; blocking enabled.",
                has_pii=overall_has_pii,
                pii_entities=dict(total_counts),
                masked_text=masked_list,
                pii_actions=pii_actions,
                is_blocked=True,
                hashed_entities=merged_hashed_entities,
            )

        return PIIDetectionResult(
            has_pii=False,
            pii_entities={},
            masked_text=None,
            original_text=chat_messages,
            is_blocked=False,
            is_masked=False,
            pii_actions={},  # No PII detected, so no actions needed
            hashed_entities={},
        )

    def _detect_string_list(self, string_list: List[str]) -> PIIDetectionResult:
        """
        Detect PII in a list of strings.

        Args:
            string_list: List of strings to detect PII in

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        overall_has_pii = False
        total_counts: Counter[str] = Counter()
        masked_list: List[str] = []
        merged_hashed_entities: Dict[str, str] = {}

        for text in string_list:
            try:
                self._detect_single_message(text)
                # If we get here, no PII was detected
                masked_list.append(text)
            except PIIBlockedException as e:
                # PII was detected
                overall_has_pii = True
                total_counts.update(e.pii_entities)
                # Merge hashed entities from this string
                merged_hashed_entities.update(e.hashed_entities)
                # Ensure we're appending a string to the string list
                masked_text_str = str(e.masked_text) if e.masked_text is not None else ""
                masked_list.append(masked_text_str)

        if overall_has_pii:
            # Create pii_actions based on the action type and detected entities
            pii_actions = {self._action_type: list(total_counts.keys())}
            raise PIIBlockedException(
                message="PII detected in one or more messages; blocking enabled.",
                has_pii=overall_has_pii,
                pii_entities=dict(total_counts),
                masked_text=masked_list,
                pii_actions=pii_actions,
                is_blocked=True,
                hashed_entities=merged_hashed_entities,
            )

        return PIIDetectionResult(
            has_pii=False,
            pii_entities={},
            masked_text=None,
            original_text=string_list,
            is_blocked=False,
            is_masked=False,
            pii_actions={},  # No PII detected, so no actions needed
            hashed_entities={},
        )


class RegexPIIDetector(PIIDetector):
    """
    Regex-based PII detector. Overrides _detect_single_message to handle a plain string.
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, Pattern[str]]] = None,
        action_type: Literal["BLOCK", "FLAG", "MASK"] = "MASK",
    ) -> None:
        if action_type is None:
            env_action = os.getenv("NETRA_ACTION_TYPE", "MASK")
            # Ensure action_type is one of the valid literal values
            if env_action not in ["BLOCK", "FLAG", "MASK"]:
                action_type = cast(Literal["BLOCK", "FLAG", "MASK"], "FLAG")
            else:
                action_type = cast(Literal["BLOCK", "FLAG", "MASK"], env_action)
        super().__init__(action_type=action_type)
        self.patterns: Dict[str, Pattern[str]] = patterns or DEFAULT_PII_PATTERNS

    def _detect_pii(self, text: str) -> Tuple[bool, Counter[str], str, Dict[str, str]]:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text, entities)
        """
        text = self._preprocess(text)  # trim & normalize
        if not text:
            return False, Counter(), "", {}

        spans: Dict[str, List[Tuple[int, int]]] = {}
        counts: Counter[str] = Counter()

        for label, pattern in self.patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            counts[label] = len(matches)
            spans[label] = [m.span() for m in matches]

        has_pii_local = bool(counts)
        masked = text
        entities: dict[str, Any] = {}

        if has_pii_local:
            masked = self._mask_spans(text, spans)

        return has_pii_local, counts, masked, entities


class PresidioPIIDetector(PIIDetector):
    """
    Presidio-based PII detector. Overrides _detect_single_message to
    call Presidio's Analyzer + Anonymizer on a string.

    Examples:
        # Using default configuration
        detector = PresidioPIIDetector()
        result = detector.detect("My email is john@example.com")

        # Using custom hash function
        import hashlib
        def custom_hash(text: str) -> str:
            return hashlib.sha256(text.encode()).hexdigest()[:8]

        detector = PresidioPIIDetector(
            hash_function=custom_hash,
            anonymizer_cache_size=500,
            action_type="MASK",
            score_threshold=0.8
        )

        # Using custom spaCy model configuration
        spacy_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
        detector = PresidioPIIDetector(nlp_configuration=spacy_config)

        # Using Stanza model configuration
        stanza_config = {
            "nlp_engine_name": "stanza",
            "models": [{"lang_code": "en", "model_name": "en"}]
        }
        detector = PresidioPIIDetector(nlp_configuration=stanza_config)

        # Using transformers model configuration
        transformers_config = {
            "nlp_engine_name": "transformers",
            "models": [{
                "lang_code": "en",
                "model_name": {
                    "spacy": "en_core_web_sm",
                    "transformers": "dbmdz/bert-large-cased-finetuned-conll03-english"
                }
            }],
            "ner_model_configuration": {
                "labels_to_ignore": ["O"],
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "LOC": "LOCATION",
                    "ORG": "ORGANIZATION"
                }
            }
        }
        detector = PresidioPIIDetector(nlp_configuration=transformers_config)
    """

    def __init__(
        self,
        entities: Optional[List[str]] = None,
        language: str = "en",
        score_threshold: float = 0.6,
        action_type: Optional[Literal["BLOCK", "FLAG", "MASK"]] = None,
        anonymizer_cache_size: int = 1000,
        hash_function: Optional[Callable[[str], str]] = None,
        nlp_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Presidio PII detector.

        Args:
            entities: List of entity types to detect. If None, uses DEFAULT_ENTITIES.
            language: Language code for detection (default: "en").
            score_threshold: Minimum confidence score for detections (default: 0.6).
            action_type: Action to take when PII is detected ("BLOCK", "FLAG", "MASK").
            anonymizer_cache_size: Size of the anonymizer cache (default: 1000).
            hash_function: Custom hash function for anonymization.
            nlp_configuration: Dictionary containing NLP engine configuration.
                Format: {
                    "nlp_engine_name": "spacy|stanza|transformers",
                    "models": [{"lang_code": "en", "model_name": "model_name"}],
                    "ner_model_configuration": {...}  # Optional, for transformers
                }

                For spaCy and Stanza:
                - model_name should be a string (e.g., "en_core_web_lg", "en")

                For transformers:
                - model_name should be a dict with "spacy" and "transformers" keys
                - Example: {"spacy": "en_core_web_sm", "transformers": "model_path"}

        Raises:
            ImportError: If presidio-analyzer is not installed or required NLP library is missing.
        """
        if action_type is None:
            action_type = "FLAG"
            env_action = os.getenv("NETRA_ACTION_TYPE", "FLAG")
            # Ensure action_type is one of the valid literal values
            if env_action in ["BLOCK", "FLAG", "MASK"]:
                action_type = cast(Literal["BLOCK", "FLAG", "MASK"], env_action)
        super().__init__(action_type=action_type)

        # Import presidio-analyzer
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: F401
        except ImportError as exc:
            raise ImportError("Presidio-based PII detection requires: presidio-analyzer. Install via pip.") from exc

        self.language: str = language
        self.entities: Optional[List[str]] = entities if entities else DEFAULT_ENTITIES
        self.score_threshold: float = score_threshold

        # Initialize AnalyzerEngine with custom or default NLP engine
        if nlp_configuration is not None:
            self.analyzer = self._create_analyzer_with_custom_nlp(nlp_configuration)
        else:
            # Use default AnalyzerEngine
            self.analyzer = AnalyzerEngine()

        self.anonymizer = Anonymizer(hash_function=hash_function, cache_size=anonymizer_cache_size)

    def _create_analyzer_with_custom_nlp(self, nlp_configuration: Dict[str, Any]) -> Any:
        """
        Create an AnalyzerEngine with custom NLP configuration.

        Args:
            nlp_configuration: Dictionary containing NLP engine configuration.

        Returns:
            AnalyzerEngine instance with custom NLP engine.

        Raises:
            ImportError: If required NLP library is not available.
        """
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as exc:
            raise ImportError("Presidio-based PII detection requires: presidio-analyzer. Install via pip.") from exc

        # Validate and prepare configuration
        engine_name = nlp_configuration.get("nlp_engine_name", "").lower()

        # Perform lazy imports based on engine type
        if engine_name == "spacy":
            self._ensure_spacy_available()
        elif engine_name == "stanza":
            self._ensure_stanza_available()
        elif engine_name == "transformers":
            self._ensure_transformers_available()
        else:
            # Default behavior - let Presidio handle it
            pass

        # Create NLP engine from configuration
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        custom_nlp_engine = provider.create_engine()

        # Extract supported languages from configuration
        supported_languages = [self.language]
        if "models" in nlp_configuration:
            supported_languages = [model["lang_code"] for model in nlp_configuration["models"]]

        return AnalyzerEngine(nlp_engine=custom_nlp_engine, supported_languages=supported_languages)

    def _ensure_spacy_available(self) -> None:
        """Ensure spaCy is available when needed."""
        try:
            import spacy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for spaCy-based PII detection. Install via: pip install spacy"
            ) from exc

    def _ensure_stanza_available(self) -> None:
        """Ensure Stanza is available when needed."""
        try:
            import stanza  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Stanza is required for Stanza-based PII detection. Install via: pip install stanza"
            ) from exc

    def _ensure_transformers_available(self) -> None:
        """Ensure transformers is available when needed."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Transformers and PyTorch are required for transformers-based PII detection. "
                "Install via: pip install transformers torch"
            ) from exc

    def _detect_pii(self, text: str) -> Tuple[bool, Counter[str], str, Dict[str, str]]:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text, entities)
        """
        text = self._preprocess(text)
        if not text:
            return False, Counter(), "", {}

        analyzer_results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

        counts = Counter([res.entity_type for res in analyzer_results])
        has_pii = bool(counts)
        masked = text
        entities: Dict[str, str] = {}

        if has_pii:
            try:
                anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
                masked = anonymized_result.masked_text
                entities = anonymized_result.entities
            except Exception:
                spans: Dict[str, List[Tuple[int, int]]] = {}
                for res in analyzer_results:
                    spans.setdefault(res.entity_type, []).append((res.start, res.end))
                masked = self._mask_spans(text, spans)

        return has_pii, counts, masked, entities


def get_default_detector(
    action_type: Optional[Literal["BLOCK", "FLAG", "MASK"]] = None,
    entities: Optional[List[str]] = None,
    hash_function: Optional[Callable[[str], str]] = None,
    nlp_configuration: Optional[Dict[str, Any]] = None,
) -> PIIDetector:
    """
    Returns a default PII detector instance (Presidio-based by default).
    If you want regex-based instead, call `set_default_detector(RegexPIIDetector(...))`.

    Args:
        action_type: Action to take when PII is detected. Options are:
            - "BLOCK": Raise PIIBlockedException when PII is detected
            - "FLAG": Detect PII but don't block or mask
            - "MASK": Replace PII with mask tokens (default)
        entities: Optional list of entity types to detect. If None, uses Presidio's default entities
        hash_function: Optional custom hash function for anonymization. If None, uses default hash function.
        nlp_configuration: Dictionary containing NLP engine configuration for custom models.
    """
    return PresidioPIIDetector(
        action_type=action_type, entities=entities, hash_function=hash_function, nlp_configuration=nlp_configuration
    )


# ---------------------------------------------------------------------------- #
#                                EXAMPLE USAGE                                  #
# ---------------------------------------------------------------------------- #
# from netra.pii import RegexPIIDetector, get_default_detector
# from netra.exceptions.pii import PIIBlockedException
#
# # Create a regex-based detector that blocks on any PII found:
# regex_detector = RegexPIIDetector(action_type="BLOCK")
# try:
#     result = regex_detector.detect("My email is sooraj@example.com")
#     # If block_on_pii=True and PII is found, code won't reach here
#     print(f"PII detected: {result.has_pii}, Entities: {result.entity_counts}")
# except PIIBlockedException as e:
#     # Access structured information about the PII detection
#     print(f"Blocked: {e.blocked}, Entities found: {e.entity_counts}")
#     print(f"Masked version: {e.masked_text}")
#
# # Or get the default (Presidio-based) detector:
# default_detector = get_default_detector(action_type="FLAG")
# pii_info = default_detector.detect("Call me at 123-456-7890")
# print(f"Found PII types: {list(pii_info.entity_counts.keys())}")
#
# # Using custom hash function with get_default_detector:
# import hashlib
# def custom_hash(text: str) -> str:
#     return hashlib.sha256(text.encode()).hexdigest()[:8]
#
# detector_with_custom_hash = get_default_detector(
#     action_type="MASK",
#     hash_function=custom_hash
# )
# result = detector_with_custom_hash.detect("My email is john@example.com")
# print(f"Masked text: {result.masked_text}")
# print(f"Entity mappings: {result.entities}")
#
# # Using PresidioPIIDetector directly with custom hash function:
# from combat.pii import PresidioPIIDetector
#
# def md5_hash(text: str) -> str:
#     return hashlib.md5(text.encode()).hexdigest()[:10]
#
# presidio_detector = PresidioPIIDetector(
#     hash_function=md5_hash,
#     anonymizer_cache_size=500,
#     action_type="MASK",
#     score_threshold=0.8
# )
# result = presidio_detector.detect("Contact me at john.doe@company.com or 555-123-4567")
# print(f"Masked: {result.masked_text}")
# print(f"Entities: {result.entities}")
#
# # You can also manually raise the exception if needed
# if pii_info.has_pii:
#     raise PIIBlockedException(
#         message="Custom blocking message",
#         has_pii=pii_info.has_pii,
#         entity_counts=pii_info.entity_counts,
#         masked_text=pii_info.masked_text,
#         is_blocked=True
#         original_text=pii_info.original_text,
#     )
