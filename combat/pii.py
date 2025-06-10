# File: combat/pii.py

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Pattern, Any, List, Union

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


@dataclass(frozen=True)
class PIIDetectionResult:
    """
    Result of running PII detection on input text.
    Attributes:
        has_pii: True if any PII matches were found.
        entity_counts: Dictionary mapping PII label -> count of occurrences.
        masked_text: Input text with PII spans replaced/masked.
        blocked: True if block_on_pii is enabled and has_pii is True.
    """

    has_pii: bool = False
    entity_counts: Dict[str, int] = field(default_factory=dict)
    masked_text: Optional[str] = None
    blocked: bool = False


class PIIDetector(ABC):
    """
    Abstract base for all PII detectors. Provides common iteration/
    aggregation logic, while requiring subclasses to implement _detect_single_message().
    """

    @abstractmethod
    def _detect_single_message(self, message: Any) -> (bool, Counter, Any):
        """
        Analyze a single "message unit" and return a tuple:
          - has_pii: bool
          - counts: Counter[str, int]
          - masked_message: same type as input (str, dict, or BaseMessage) but with content masked.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement _detect_single_message")

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #
    def _preprocess(self, text: str) -> str:
        """Trim and normalise whitespace; subclasses may override."""
        return text.strip()

    def _mask_spans(self, text: str, spans: Dict[str, List[tuple]]) -> str:
        """Return text with every span replaced by '***'. Non-overlapping assumed."""
        char_list = list(text)
        for span_group in spans.values():
            for start, end in span_group:
                char_list[start:end] = '*' * (end - start)
        return ''.join(char_list)


    def detect(
        self, input_data: Union[str, List[Dict[str, str]]]
    ) -> PIIDetectionResult:
        """
        Public entry point. Accepts either:
          - a plain string
          - a list of {"role":..., "message":...} dicts
          - a list of LangChain `BaseMessage` objects

        Returns PIIDetectionResult with:
          - has_pii: True if any message had PII
          - entity_counts: aggregated counts across all messages
          - masked_text:
              * str, if input was a single string
              * List[Dict[str,str]], if input was a list of dicts
              * List[BaseMessage], if input was a list of BaseMessage
          - blocked: True if block_on_pii is True and has_pii is True
        """
        # Helper to unify a single plain string:
        if isinstance(input_data, str):
            has_pii, counts, masked_obj = self._detect_single_message(input_data)
            if has_pii and self._block_on_pii:
                raise PIIBlockedException(
                    message="PII detected; blocking enabled.",
                    has_pii=has_pii,
                    entity_counts=dict(counts),
                    masked_text=masked_obj,
                    blocked=True,
                )
            return PIIDetectionResult(
                has_pii=has_pii,
                entity_counts=dict(counts),
                masked_text=masked_obj,
                blocked=False,
            )

        # If it’s a list of simple dicts ({"role":..., "message": ...}):
        if (
            isinstance(input_data, list)
            and input_data
            and isinstance(input_data[0], dict)
        ):
            overall_has_pii = False
            total_counts: Counter = Counter()
            masked_list: List[Dict[str, str]] = []

            for item in input_data:
                role = item.get("role", "")
                text = item.get("message", "")
                has_pii_local, counts_local, masked_msg = self._detect_single_message(
                    text
                )
                if has_pii_local:
                    overall_has_pii = True
                    total_counts.update(counts_local)
                masked_list.append({"role": role, "message": masked_msg})

            if overall_has_pii and self._block_on_pii:
                raise PIIBlockedException(
                    message="PII detected in one or more messages; blocking enabled.",
                    has_pii=overall_has_pii,
                    entity_counts=dict(total_counts),
                    masked_text=masked_list,
                    blocked=True,
                )

            return PIIDetectionResult(
                has_pii=overall_has_pii,
                entity_counts=dict(total_counts),
                masked_text=masked_list,
                blocked=False,
            )

        # If none of the above, it’s a misuse
        raise TypeError(
            "PIIDetector.detect() expects either a str, "
            "List[Dict[str,str]], or List[BaseMessage]."
        )


class RegexPIIDetector(PIIDetector):
    """
    Regex-based PII detector. Overrides _detect_single_message to handle a plain string.
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, Pattern]] = None,
        block_on_pii: bool = False,
    ) -> None:
        self.patterns: Dict[str, Pattern] = patterns or DEFAULT_PII_PATTERNS
        self._block_on_pii: bool = block_on_pii

    def _detect_single_message(self, message: Any) -> (bool, Counter, Any):
        """
        Called by the base class. `message` is always a string.
        Returns (has_pii, counts, masked_string).
        """
        text = self._preprocess(message)  # trim & normalize
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
    call Presidio’s Analyzer + Anonymizer on a string.
    """

    def __init__(
        self,
        entities: Optional[list] = None,
        language: str = "en",
        score_threshold: float = 0.5,
        block_on_pii: bool = False,
    ) -> None:
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
        self.entities: Optional[list] = entities
        self.score_threshold: float = score_threshold
        self._block_on_pii: bool = block_on_pii

        self.analyzer = self._create_flair_analyzer_engine()
        self.anonymizer = AnonymizerEngine()

    def _detect_single_message(self, message: Any) -> (bool, Counter, Any):
        """
        Called by the base class. `message` is always a string.
        Returns (has_pii, counts, masked_str) after using Presidio.
        """
        text = self._preprocess(message)
        if not text:
            return False, Counter(), ""

        analyzer_results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

        # Count each occurrence of entity types
        counts = Counter(res.entity_type for res in analyzer_results)
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
        nlp_engine = NlpEngineProvider(nlp_config).create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


def get_default_detector(block_on_pii: bool = False) -> PIIDetector:
    """
    Returns a default PII detector instance (Presidio-based by default).
    If you want regex-based instead, call `set_default_detector(RegexPIIDetector(...))`.
    """
    return PresidioPIIDetector(block_on_pii=block_on_pii)


# ---------------------------------------------------------------------------- #
#                                EXAMPLE USAGE                                  #
# ---------------------------------------------------------------------------- #
# from combat.pii import RegexPIIDetector, get_default_detector
# from combat.exceptions.pii import PIIBlockedException
#
# # Create a regex-based detector that blocks on any PII found:
# regex_detector = RegexPIIDetector(block_on_pii=True)
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
# default_detector = get_default_detector(block_on_pii=False)
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
