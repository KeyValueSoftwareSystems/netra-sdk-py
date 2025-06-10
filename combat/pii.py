# File: combat/pii.py
import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Union, Literal, Tuple, cast

try:
    from opentelemetry.trace import get_current_span
except ImportError:
    # Stub for when opentelemetry is not available
    def get_current_span():
        return None

from combat.exceptions import PIIBlockedException

EMAIL_PATTERN: Pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN: Pattern = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN: Pattern = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_PATTERN: Pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

DEFAULT_PII_PATTERNS: Dict[str, Pattern] = {
    "EMAIL": EMAIL_PATTERN,
    "PHONE": PHONE_PATTERN,
    "CREDIT_CARD": CREDIT_CARD_PATTERN,
    "SSN": SSN_PATTERN,
}

DEFAULT_ENTITIES = [
    "CREDIT_CARD",
    "CRYPTO",
    "DATE_TIME",
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
    "ES_NIF",
    "ES_NIE",
    "IT_FISCAL_CODE",
    "IT_DRIVER_LICENSE",
    "IT_VAT_CODE",
    "IT_PASSPORT",
    "IT_IDENTITY_CARD",
    "PL_PESEL",
    "SG_NRIC_FIN",
    "SG_UEN",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    "IN_PAN",
    "IN_AADHAAR",
    "IN_VEHICLE_REGISTRATION",
    "IN_VOTER",
    "IN_PASSPORT",
    "FI_PERSONAL_IDENTITY_CODE"
]


@dataclass(frozen=True)
class PIIDetectionResult:
    """
    Result of running PII detection on input text.
    Attributes:
        has_pii: True if any PII matches were found.
        entity_counts: Dictionary mapping PII label -> count of occurrences.
        masked_text: Input text with PII spans replaced/masked.
            Can be a string for simple inputs, a list of dicts for chat messages,
            or a list of BaseMessage objects for LangChain inputs.
        is_blocked: True if block_on_pii is enabled and has_pii is True.
        is_masked: True if any text was replaced to mask PII.
    """

    has_pii: bool = False
    entity_counts: Dict[str, int] = field(default_factory=dict)
    masked_text: Optional[Union[str, List[Dict[str, str]], List[Any]]] = None
    is_blocked: bool = False
    is_masked: bool = False


class PIIDetector(ABC):
    """
    Abstract base for all PII detectors. Provides common iteration/
    aggregation logic, while requiring subclasses to implement _detect_single_message().
    """

    def __init__(self, action_type: Literal["BLOCK", "FLAG", "MASK"] = "MASK") -> None:
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
    def _detect_pii(self, text: str) -> Tuple[bool, Counter, str]:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text)
        """
        pass

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

    def _mask_spans(self, text: str, spans: Dict[str, List[tuple]]) -> str:
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

    def _record_detection_trace(self, has_pii: bool, counts: Counter, masked_text: str) -> None:
        """
        Record PII detection results as a trace event if tracing is enabled.

        Args:
            has_pii: Whether PII was detected
            counts: Counter of PII entity types found
            masked_text: The masked version of the text
        """
        try:
            span = get_current_span()
            if span:
                attributes = {
                    "has_pii": has_pii,
                    "entity_counts": json.dumps(dict(counts)),
                    "is_blocked": self._action_type == "BLOCK" and has_pii,
                    "is_masked": self._action_type == "MASK",
                }

                # Add masked_text to attributes only for MASK action type
                if self._action_type == "MASK" and has_pii:
                    if isinstance(masked_text, dict):
                        attributes["masked_text"] = json.dumps(masked_text)
                    elif isinstance(masked_text, list):
                        attributes["masked_text"] = json.dumps(masked_text)
                    else:
                        attributes["masked_text"] = str(masked_text)

                span.add_event("pii_detection_result", attributes)
        except (NameError, AttributeError):
            # Tracing is not available or not configured
            pass

    def detect(
            self, input_data: Union[str, List[Dict[str, str]], List[str], List[Any]]
    ) -> PIIDetectionResult:
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
            # Handle different input types
            if isinstance(input_data, str):
                return self._detect_single_message(input_data)
            elif isinstance(input_data, list):
                if not input_data:
                    return PIIDetectionResult()

                # Check first item to determine list type
                first_item = input_data[0]

                # Case: List of dictionaries (chat messages)
                if isinstance(first_item, dict):
                    return self._detect_chat_messages(input_data)
                # Case: List of strings
                elif isinstance(first_item, str):
                    return self._detect_string_list(input_data)
                # Case: List of LangChain BaseMessage-like objects (duck typing)
                elif hasattr(first_item, "content") and hasattr(first_item, "type"):
                    # Extract content from BaseMessage-like objects
                    contents = [msg.content for msg in input_data if hasattr(msg, "content")]
                    return self._detect_string_list(contents)
                else:
                    raise ValueError(
                        f"Unsupported input type in list: {type(first_item).__name__}"
                    )
            else:
                raise ValueError(f"Unsupported input type: {type(input_data).__name__}")
        except PIIBlockedException as e:
            # Catch the exception to add it to the span
            span = get_current_span()
            if span:
                attributes = {
                    "has_pii": e.has_pii,
                    "entity_counts": json.dumps(e.entity_counts),
                    "is_blocked": self._action_type == "BLOCK",
                    "is_masked": self._action_type == "MASK",
                }

                # Add masked_text to attributes only for MASK action type
                if self._action_type == "MASK":
                    if isinstance(e.masked_text, dict):
                        attributes["masked_text"] = json.dumps(e.masked_text)
                    elif isinstance(e.masked_text, list):
                        attributes["masked_text"] = json.dumps(e.masked_text)
                    else:
                        attributes["masked_text"] = str(e.masked_text)

                span.add_event("pii_detection_result", attributes)

            # Re-raise the exception if action_type is BLOCK
            if self._action_type == "BLOCK":
                raise

            # For MASK action type, return the masked text
            if self._action_type == "MASK":
                return PIIDetectionResult(
                    has_pii=e.has_pii,
                    entity_counts=e.entity_counts,
                    masked_text=e.masked_text,
                    is_blocked=False,
                    is_masked=True,
                )

            # For FLAG action type, return without masked_text
            return PIIDetectionResult(
                has_pii=e.has_pii,
                entity_counts=e.entity_counts,
                masked_text=None,
                is_blocked=False,
                is_masked=False,
            )

    def _detect_single_message(self, text: str) -> PIIDetectionResult:
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            PIIDetectionResult: The detection result containing PII information
        """
        has_pii, counts, masked_text = self._detect_pii(text)

        if has_pii:
            raise PIIBlockedException(
                message="PII detected; blocking enabled.",
                has_pii=has_pii,
                entity_counts=dict(counts),
                masked_text=masked_text,
                blocked=True,
            )

        return PIIDetectionResult(
            has_pii=has_pii,
            entity_counts=dict(counts),
            masked_text=None,  # No PII detected, so no masked text needed
            is_blocked=False,
            is_masked=False,
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
        total_counts: Counter = Counter()
        masked_list: List[Dict[str, str]] = []

        for message in chat_messages:
            role = message.get("role", "unknown")
            text = message.get("content", "")

            try:
                result = self._detect_single_message(text)
                # If we get here, no PII was detected
                masked_list.append({"role": role, "message": text})
            except PIIBlockedException as e:
                # PII was detected
                overall_has_pii = True
                total_counts.update(e.entity_counts)
                masked_list.append({"role": role, "message": e.masked_text})

        if overall_has_pii:
            raise PIIBlockedException(
                message="PII detected in one or more messages; blocking enabled.",
                has_pii=overall_has_pii,
                entity_counts=dict(total_counts),
                masked_text=masked_list,
                blocked=True,
            )

        return PIIDetectionResult(
            has_pii=False,
            entity_counts={},
            masked_text=None,
            is_blocked=False,
            is_masked=False,
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
        total_counts: Counter = Counter()
        masked_list: List[str] = []

        for text in string_list:
            try:
                result = self._detect_single_message(text)
                # If we get here, no PII was detected
                masked_list.append(text)
            except PIIBlockedException as e:
                # PII was detected
                overall_has_pii = True
                total_counts.update(e.entity_counts)
                masked_list.append(e.masked_text)

        if overall_has_pii:
            raise PIIBlockedException(
                message="PII detected in one or more messages; blocking enabled.",
                has_pii=overall_has_pii,
                entity_counts=dict(total_counts),
                masked_text=masked_list,
                blocked=True,
            )

        return PIIDetectionResult(
            has_pii=False,
            entity_counts={},
            masked_text=None,
            is_blocked=False,
            is_masked=False,
        )


class RegexPIIDetector(PIIDetector):
    """
    Regex-based PII detector. Overrides _detect_single_message to handle a plain string.
    """

    def __init__(
            self,
            patterns: Optional[Dict[str, Pattern]] = None,
            action_type: Literal["BLOCK", "FLAG", "MASK"] = "MASK",
    ) -> None:
        super().__init__(action_type=action_type)
        self.patterns: Dict[str, Pattern] = patterns or DEFAULT_PII_PATTERNS

    def _detect_pii(self, text: str) -> (bool, Counter, str):
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text)
        """
        text = self._preprocess(text)  # trim & normalize
        if not text:
            return False, Counter(), ""

        spans: Dict[str, List[tuple]] = {}
        counts: Counter = Counter()

        for label, pattern in self.patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            counts[label] = len(matches)
            spans[label] = [m.span() for m in matches]

        has_pii_local = bool(counts)
        masked = text
        if has_pii_local:
            masked = self._mask_spans(text, spans)

        return has_pii_local, counts, masked


class PresidioPIIDetector(PIIDetector):
    """
    Presidio-based PII detector. Overrides _detect_single_message to
    call Presidio's Analyzer + Anonymizer on a string.
    """

    def __init__(
            self,
            entities: Optional[list] = None,
            language: str = "en",
            score_threshold: float = 0.6,
            action_type: Optional[Literal["BLOCK", "FLAG", "MASK"]] = None,
    ) -> None:
        if action_type is None:
            env_action = os.getenv("COMBAT_ACTION_TYPE", "MASK")
            # Ensure action_type is one of the valid literal values
            if env_action not in ["BLOCK", "FLAG", "MASK"]:
                action_type = cast(Literal["BLOCK", "FLAG", "MASK"], "MASK")
            else:
                action_type = cast(Literal["BLOCK", "FLAG", "MASK"], env_action)
        super().__init__(action_type=action_type)
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: F401
            from presidio_anonymizer import AnonymizerEngine
            import flair  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Presidio-based PII detection requires: presidio-analyzer, "
                "presidio-anonymizer, flair. Install via pip."
            ) from exc

        self.language: str = language
        self.entities: Optional[list] = entities if entities else DEFAULT_ENTITIES
        self.score_threshold: float = score_threshold

        # self.analyzer = self._create_flair_analyzer_engine()
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def _detect_pii(self, text: str) -> (bool, Counter, str):
        """
        Detect PII in a single message.

        Args:
            text: The text to detect PII in

        Returns:
            Tuple of (has_pii, counts, masked_text)
        """
        text = self._preprocess(text)
        if not text:
            return False, Counter(), ""

        analyzer_results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

        counts: Counter = Counter({res.entity_type: 1 for res in analyzer_results})
        has_pii_local = bool(counts)
        masked = text

        if has_pii_local:
            try:
                anonymized = self.anonymizer.anonymize(
                    text=text, analyzer_results=analyzer_results
                )
                masked = anonymized.text
            except Exception:
                spans: Dict[str, List[tuple]] = {}
                for res in analyzer_results:
                    spans.setdefault(res.entity_type, []).append((res.start, res.end))
                masked = self._mask_spans(text, spans)

        return has_pii_local, counts, masked

    def _create_flair_analyzer_engine(
            self, model_path: str = "flair/ner-english-large"
    ):
        """
        Private helper: set up Presidio AnalyzerEngine with FlairNLP + spaCy.
        """
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        import spacy
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from combat.flair_recognizer import FlairRecognizer

        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        flair_recognizer = FlairRecognizer(model_path=model_path)
        registry.add_recognizer(flair_recognizer)
        registry.remove_recognizer("SpacyRecognizer")

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": self.language, "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


def get_default_detector(action_type: Literal["BLOCK", "FLAG", "MASK"] = "MASK", entities: Optional[List[str]] = None) -> PIIDetector:
    """
    Returns a default PII detector instance (Presidio-based by default).
    If you want regex-based instead, call `set_default_detector(RegexPIIDetector(...))`.

    Args:
        action_type: Action to take when PII is detected. Options are:
            - "BLOCK": Raise PIIBlockedException when PII is detected
            - "FLAG": Detect PII but don't block or mask
            - "MASK": Replace PII with mask tokens (default)
        entities: Optional list of entity types to detect. If None, uses Presidio's default entities
    """
    return PresidioPIIDetector(action_type=action_type, entities=entities)

# ---------------------------------------------------------------------------- #
#                                EXAMPLE USAGE                                  #
# ---------------------------------------------------------------------------- #
# from combat.pii import RegexPIIDetector, get_default_detector
# from combat.exceptions.pii import PIIBlockedException
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
# # You can also manually raise the exception if needed
# if pii_info.has_pii:
#     raise PIIBlockedException(
#         message="Custom blocking message",
#         has_pii=pii_info.has_pii,
#         entity_counts=pii_info.entity_counts,
#         masked_text=pii_info.masked_text,
#         blocked=True
#     )
