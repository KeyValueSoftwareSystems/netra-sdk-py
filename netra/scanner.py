"""
Scanner module for Netra SDK to implement various scanning capabilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

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
    Supports custom model configuration for enhanced detection capabilities.

    Examples:
        # Using default configuration
        scanner = PromptInjection()

        # Using custom threshold
        scanner = PromptInjection(threshold=0.8)

        # Using custom model configuration
        model_config = {
            "model": "deepset/deberta-v3-base-injection",
            "tokenizer": "deepset/deberta-v3-base-injection",
            "device": "cpu",
            "max_length": 512
        }
        scanner = PromptInjection(model_configuration=model_config)

        # Using custom model with specific match type
        from llm_guard.input_scanners.prompt_injection import MatchType
        scanner = PromptInjection(
            threshold=0.7,
            match_type=MatchType.SENTENCE,
            model_configuration=model_config
        )
    """

    def __init__(
        self,
        threshold: float = 0.5,
        match_type: Optional[str] = None,
        model_configuration: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PromptInjection scanner.

        Args:
            threshold: The threshold value (between 0.0 and 1.0) above which a prompt is considered risky
            match_type: The type of matching to use
                (from llm_guard.input_scanners.prompt_injection.MatchType)
            model_configuration: Dictionary containing custom model configuration.
                Format: {
                    "model": "model_name_or_path",  # HuggingFace model name or local path
                    "device": "cpu|cuda",  # Optional, defaults to "cpu"
                    "max_length": 512,  # Optional, max sequence length
                    "use_onnx": False,  # Optional, use ONNX runtime
                    "onnx_model_path": "/path/to/model.onnx",  # Required if use_onnx=True
                    "torch_dtype": "float16"  # Optional, torch data type
                }

        Raises:
            ImportError: If required dependencies are not installed.
            ValueError: If model configuration is invalid.
        """
        self.threshold = threshold
        self.model_configuration = model_configuration
        self.scanner = None
        self.llm_guard_available = False

        try:
            from llm_guard.input_scanners import PromptInjection as LLMGuardPromptInjection
            from llm_guard.input_scanners.prompt_injection import MatchType

            if match_type is None:
                match_type = MatchType.FULL

            # Create scanner with custom model configuration if provided
            if model_configuration is not None:
                self.scanner = self._create_scanner_with_custom_model(
                    LLMGuardPromptInjection, threshold, match_type, model_configuration
                )
            else:
                self.scanner = LLMGuardPromptInjection(threshold=threshold, match_type=match_type)

            self.llm_guard_available = True
        except ImportError:
            logger.warning(
                "llm-guard package is not installed. Prompt injection scanning will be limited. "
                "To enable full functionality, install with: pip install 'netra-sdk[llm_guard]'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PromptInjection scanner: {e}")
            raise

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

    def _create_scanner_with_custom_model(
        self, scanner_class: Any, threshold: float, match_type: Any, model_config: Dict[str, Any]
    ) -> Any:
        """
        Create a PromptInjection scanner with custom model configuration.

        Args:
            scanner_class: The LLMGuardPromptInjection class
            threshold: Detection threshold
            match_type: Type of matching to use
            model_config: Dictionary containing model configuration

        Returns:
            Configured PromptInjection scanner instance

        Raises:
            ImportError: If required dependencies are not available
            ValueError: If model configuration is invalid
        """
        # Validate model configuration
        self._validate_model_configuration(model_config)

        # Check if using ONNX runtime
        if model_config.get("use_onnx", False):
            return self._create_onnx_scanner(scanner_class, threshold, match_type, model_config)
        else:
            return self._create_transformers_scanner(scanner_class, threshold, match_type, model_config)

    def _validate_model_configuration(self, model_config: Dict[str, Any]) -> None:
        """
        Validate the model configuration dictionary.

        Args:
            model_config: Dictionary containing model configuration

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["model"]

        # Check for required fields
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"Missing required field '{field}' in model configuration")

        # Validate ONNX-specific requirements
        if model_config.get("use_onnx", False):
            if "onnx_model_path" not in model_config:
                raise ValueError("'onnx_model_path' is required when use_onnx=True")

        # Validate device
        device = model_config.get("device", "cpu")
        if device not in ["cpu", "cuda"]:
            logger.warning(f"Unknown device '{device}', defaulting to 'cpu'")
            model_config["device"] = "cpu"

    def _create_transformers_scanner(
        self, scanner_class: Any, threshold: float, match_type: Any, model_config: Dict[str, Any]
    ) -> Any:
        """
        Create scanner with transformers-based model.

        Args:
            scanner_class: The LLMGuardPromptInjection class
            threshold: Detection threshold
            match_type: Type of matching to use
            model_config: Dictionary containing model configuration

        Returns:
            Configured scanner instance
        """
        try:
            from llm_guard.model import Model
        except ImportError as exc:
            raise ImportError(
                "Custom model configuration requires llm-guard. " "Install with: pip install llm-guard"
            ) from exc

        # Extract configuration parameters
        model_name = model_config["model"]
        device = model_config.get("device", "cpu")
        max_length = model_config.get("max_length", 512)
        torch_dtype = model_config.get("torch_dtype")

        logger.info(f"Loading custom model: {model_name}")

        # Prepare model kwargs for transformers
        model_kwargs = {}
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype

        # Prepare pipeline kwargs
        pipeline_kwargs = {
            "device": device,
            "max_length": max_length,
            "truncation": True,
            "return_token_type_ids": False,
        }

        # Create llm-guard Model object
        custom_model = Model(path=model_name, kwargs=model_kwargs, pipeline_kwargs=pipeline_kwargs)

        # Create scanner with custom model
        return scanner_class(model=custom_model, threshold=threshold, match_type=match_type)

    def _create_onnx_scanner(
        self, scanner_class: Any, threshold: float, match_type: Any, model_config: Dict[str, Any]
    ) -> Any:
        """
        Create scanner with ONNX runtime model.

        Args:
            scanner_class: The LLMGuardPromptInjection class
            threshold: Detection threshold
            match_type: Type of matching to use
            model_config: Dictionary containing model configuration

        Returns:
            Configured scanner instance
        """
        try:
            from llm_guard.model import Model
        except ImportError as exc:
            raise ImportError(
                "ONNX model configuration requires llm-guard. " "Install with: pip install llm-guard"
            ) from exc

        # Extract ONNX configuration
        onnx_model_path = model_config["onnx_model_path"]
        model_name = model_config["model"]
        max_length = model_config.get("max_length", 512)
        device = model_config.get("device", "cpu")

        logger.info(f"Loading ONNX model: {onnx_model_path}")

        # Prepare pipeline kwargs
        pipeline_kwargs = {
            "device": device,
            "max_length": max_length,
            "truncation": True,
            "return_token_type_ids": False,
        }

        # Create llm-guard Model object with ONNX configuration
        custom_model = Model(path=model_name, onnx_path=onnx_model_path, pipeline_kwargs=pipeline_kwargs)

        # Create scanner with ONNX model
        return scanner_class(model=custom_model, threshold=threshold, match_type=match_type, use_onnx=True)
