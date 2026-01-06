"""
Base evaluator class for Netra evaluation framework.

This module provides the abstract base class that all custom evaluators
should inherit from when implementing local evaluators for run_test_suite().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from netra.evaluation.models import EvaluatorConfig, EvaluatorOutput


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Subclasses must implement:
        - get(): Returns evaluator configuration (name, label, scoreType).
        - evaluate(): Runs the evaluation logic and returns a result.

    Example:
        class MyEvaluator(BaseEvaluator):
            def get(self) -> EvaluatorConfig:
                return EvaluatorConfig(
                    name="my_evaluator",
                    label="My Custom Evaluator",
                    score_type=ScoreType.BOOLEAN,
                )

            def evaluate(
                self,
                input: Any,
                task_output: Any,
                expected_output: Any,
                metadata: Optional[Dict[str, Any]] = None,
            ) -> EvaluatorResult:
                is_correct = task_output == expected_output
                return EvaluatorResult(
                    evaluatorName="my_evaluator",
                    result=is_correct,
                    isPassed=is_correct,
                    reason="Output matches expected" if is_correct else "Mismatch",
                )
    """

    @abstractmethod
    def get(self) -> EvaluatorConfig:
        """
        Return the evaluator configuration.

        The returned dictionary must contain:
            - name (str): Unique identifier for the evaluator.
            - label (str): Human-readable display name.
            - scoreType (str): One of "boolean", "numeric", "percentage".

        Returns:
            Dict[str, Any]: Evaluator configuration dictionary.
        """

    @abstractmethod
    def evaluate(
        self,
        input: Any,
        task_output: Any,
        expected_output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorOutput:
        """
        Run the evaluation logic.

        This method can be sync or async. If async, the framework will
        await the coroutine automatically.

        Args:
            input: The original input passed to the task function.
            task_output: The output returned by the task function.
            expected_output: The expected output from the dataset item.
            metadata: Optional metadata from the dataset item.

        Returns:
            EvaluatorOutput: The evaluation result containing:
                - evaluator_name (str): Must match the name from get().
                - result (Any): The evaluation score/value.
                - is_passed (bool): Whether the evaluation passed.
                - reason (Optional[str]): Explanation for the result.
        """
