"""
Base evaluator class for Netra evaluation framework.

This module provides the abstract base class that all custom evaluators
should inherit from when implementing local evaluators for run_test_suite().
"""

from abc import ABC, abstractmethod

from netra.evaluation.models import EvaluatorConfig, EvaluatorContext, EvaluatorOutput


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Subclasses must:
        - Implement evaluate(): Runs the evaluation logic and returns a result.

    The configuration is passed when instantiating the evaluator.

    Example:
        class MyEvaluator(BaseEvaluator):
            def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
                is_correct = context.task_output == context.expected_output
                return EvaluatorOutput(
                    evaluator_name=self.config.name,
                    result=is_correct,
                    is_passed=is_correct,
                    reason="Output matches expected" if is_correct else "Mismatch",
                )

        # Usage:
        result = Netra.evaluation.run_test_suite(
            name="Copywriting Assistant v1",
            data=dataset,
            task=get_copywriting_agent_response,
            evaluators=[
                MyEvaluator(
                    EvaluatorConfig(
                        name="my_evaluator",
                        label="My Custom Evaluator",
                        score_type=ScoreType.BOOLEAN,
                    )
                )
            ]
        )
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        """
        Initialize the evaluator with its configuration.

        Args:
            config: EvaluatorConfig containing:
                - name (str): Unique identifier for the evaluator.
                - label (str): Human-readable display name.
                - score_type (ScoreType): One of BOOLEAN, NUMERICAL, CATEGORICAL.
        """
        self.config = config

    @abstractmethod
    def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
        """
        Run the evaluation logic.

        This method can be sync or async. If async, the framework will
        await the coroutine automatically.

        Args:
            context: EvaluatorContext containing:
                - input: The original input passed to the task function.
                - task_output: The output returned by the task function.
                - expected_output: The expected output from the dataset item.
                - metadata: Optional metadata from the dataset item.

        Returns:
            EvaluatorOutput: The evaluation result containing:
                - evaluator_name (str): Must match the name from config.
                - result (Any): The evaluation score/value.
                - is_passed (bool): Whether the evaluation passed.
                - reason (Optional[str]): Explanation for the result.
        """
